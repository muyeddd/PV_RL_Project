from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights

from experiments import run_convnext_tiny_image_only_pilot as runner
from experiments import train_convnext_tiny_image_only_date_grouped as training
from experiments import train_resnet50_image_only_date_grouped as baseline
from models.convnext_tiny_image_only import (
    CLASSIFIER_INPUT_FEATURES,
    OFFICIAL_CHECKPOINT_FILENAME,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_WEIGHTS,
    SolarConvNeXtTinyImageOnly,
    verify_official_checkpoint,
)


class ConvNeXtTinyWeightsTests(unittest.TestCase):
    def test_torchvision_official_enum_is_imagenet1k_v1(self):
        self.assertIs(OFFICIAL_WEIGHTS, ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.assertEqual(OFFICIAL_WEIGHTS, ConvNeXt_Tiny_Weights.DEFAULT)

    def test_exact_local_checkpoint_filename_and_sha256(self):
        provenance = verify_official_checkpoint()
        self.assertEqual(provenance["filename"], OFFICIAL_CHECKPOINT_FILENAME)
        self.assertEqual(provenance["sha256"], OFFICIAL_CHECKPOINT_SHA256)
        self.assertEqual(provenance["size_bytes"], 114_419_221)

    def test_pretrained_model_loads_only_verified_local_checkpoint(self):
        model = SolarConvNeXtTinyImageOnly(use_pretrained=True)
        self.assertEqual(
            model.pretrained_provenance["weights_enum"],
            "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        )
        self.assertEqual(
            model.pretrained_provenance["sha256"], OFFICIAL_CHECKPOINT_SHA256
        )


class ConvNeXtTinyModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SolarConvNeXtTinyImageOnly(use_pretrained=False)

    def test_preserves_norm_and_flatten(self):
        classifier = self.model.backbone.classifier
        self.assertEqual(type(classifier[0]).__name__, "LayerNorm2d")
        self.assertIsInstance(classifier[1], nn.Flatten)

    def test_head_is_strictly_768_128_1(self):
        head = self.model.regression_head
        self.assertEqual(CLASSIFIER_INPUT_FEATURES, 768)
        self.assertIsInstance(head, nn.Sequential)
        self.assertEqual(len(head), 5)
        self.assertIsInstance(head[0], nn.Linear)
        self.assertEqual((head[0].in_features, head[0].out_features), (768, 128))
        self.assertIsInstance(head[1], nn.ReLU)
        self.assertIsInstance(head[2], nn.Dropout)
        self.assertEqual(head[2].p, 0.3)
        self.assertIsInstance(head[3], nn.Linear)
        self.assertEqual((head[3].in_features, head[3].out_features), (128, 1))
        self.assertIsInstance(head[4], nn.Sigmoid)

    def test_head_has_no_batchnorm_attention_or_extra_hidden_layer(self):
        head = self.model.regression_head
        self.assertFalse(any(isinstance(module, nn.BatchNorm1d) for module in head))
        self.assertEqual(sum(isinstance(module, nn.Linear) for module in head), 2)
        self.assertFalse(any("attention" in type(module).__name__.lower() for module in head))

    def test_rgb_224_input_and_batch_by_one_output(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 1))
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))

    def test_entire_backbone_is_trainable(self):
        parameters = list(self.model.backbone.parameters())
        self.assertTrue(parameters)
        self.assertTrue(all(parameter.requires_grad for parameter in parameters))

    def test_forward_accepts_images_only(self):
        signature = inspect.signature(SolarConvNeXtTinyImageOnly.forward)
        self.assertEqual(list(signature.parameters), ["self", "images"])


class ConvNeXtTinyTransformTests(unittest.TestCase):
    def test_transform_function_is_exact_formal_baseline_source(self):
        self.assertIs(training.build_transforms, baseline.build_transforms)
        challenger = training.build_transforms()
        reference = baseline.build_transforms()
        self.assertEqual(repr(challenger), repr(reference))

    def test_train_transform_order_and_parameters(self):
        train_transform, validation_transform = training.build_transforms()
        names = [type(item).__name__ for item in train_transform.transforms]
        self.assertEqual(
            names,
            [
                "Resize",
                "RandomResizedCrop",
                "RandomHorizontalFlip",
                "RandomRotation",
                "ColorJitter",
                "ToTensor",
                "Normalize",
            ],
        )
        self.assertEqual(train_transform.transforms[0].size, (256, 256))
        crop = train_transform.transforms[1]
        self.assertEqual(crop.size, (224, 224))
        self.assertEqual(crop.scale, (0.85, 1.0))
        self.assertEqual(train_transform.transforms[2].p, 0.5)
        self.assertEqual(train_transform.transforms[3].degrees, [-7.0, 7.0])
        jitter = train_transform.transforms[4]
        self.assertIsInstance(jitter, transforms.ColorJitter)
        self.assertEqual(jitter.brightness, (0.92, 1.08))
        self.assertEqual(jitter.contrast, (0.92, 1.08))
        self.assertEqual(jitter.saturation, (0.95, 1.05))
        self.assertEqual(jitter.hue, (-0.02, 0.02))
        self.assertEqual(
            [type(item).__name__ for item in validation_transform.transforms],
            ["Resize", "ToTensor", "Normalize"],
        )
        self.assertEqual(validation_transform.transforms[0].size, (224, 224))

    def test_no_photometric_v1_gamma_or_stronger_jitter(self):
        train_transform, _ = training.build_transforms()
        representation = repr(train_transform).lower()
        self.assertNotIn("gamma", representation)
        self.assertNotIn("randomgamma", representation)
        self.assertEqual(
            repr(train_transform.transforms[4]),
            repr(baseline.build_transforms()[0].transforms[4]),
        )


class ConvNeXtTinyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = training.load_training_config()

    def test_frozen_training_protocol(self):
        training.validate_frozen_protocol(self.config)
        expected = {
            "seed": 42,
            "epochs": 50,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "loss": "MSELoss",
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": 0.5,
            "scheduler_patience": 2,
            "early_stopping_patience": 8,
            "selection_metric": "validation_rmse",
            "amp": True,
            "pretrained": True,
            "dropout": 0.3,
            "allowed_pilot_folds": [1, 2, 3, 4],
        }
        for key, value in expected.items():
            self.assertEqual(self.config[key], value, key)

    def test_loss_optimizer_and_uniform_lr_implementation(self):
        source = inspect.getsource(training.run_training)
        self.assertIn("nn.MSELoss", source)
        self.assertIn("torch.optim.AdamW", source)
        self.assertIn('lr=config["learning_rate"]', source)
        self.assertNotIn("differential", source.lower())
        self.assertNotIn("layer_wise", source.lower())

    def test_amp_and_full_backprop_are_implemented(self):
        source = inspect.getsource(training.run_epoch)
        self.assertIn("torch.amp.autocast", source)
        self.assertIn("backward()", source)
        self.assertIn("optimizer.step()", source)

    def test_memory_smoke_rejects_batch64_before_loading_data(self):
        with self.assertRaises(ValueError):
            training.run_single_batch_memory_smoke(64)

    def test_dataset_returns_only_image_l_and_identity_metadata(self):
        source = inspect.getsource(training.ModelDevelopmentImageOnlyDataset.__getitem__)
        self.assertIn('convert("RGB")', source)
        self.assertIn("target_l", source)
        self.assertNotIn("time_feat", source)
        self.assertNotIn("irradiance", source.lower())

    def test_all_four_folds_share_one_model_and_training_protocol(self):
        config = training.load_training_config()
        frozen_keys = (
            "seed",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "loss",
            "optimizer",
            "scheduler",
            "scheduler_factor",
            "scheduler_patience",
            "early_stopping_patience",
            "selection_metric",
            "amp",
            "dropout",
            "pretrained",
        )
        expected_protocol = {key: config[key] for key in frozen_keys}
        model_names = set()
        protocols = []
        for fold in training.ALLOWED_PILOT_FOLDS:
            args = training.parse_args(["--fold", str(fold)])
            model_names.add(config["model_name"])
            protocols.append(
                {
                    **expected_protocol,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": (
                        args.gradient_accumulation_steps
                    ),
                }
            )
        self.assertEqual(model_names, {"SolarConvNeXtTinyImageOnly"})
        self.assertTrue(all(protocol == expected_protocol for protocol in protocols))
        run_source = inspect.getsource(training.run_training)
        self.assertEqual(run_source.count("SolarConvNeXtTinyImageOnly("), 1)

    def test_all_four_folds_share_exactly_one_transform_protocol(self):
        transform_pairs = [
            tuple(repr(transform) for transform in training.build_transforms())
            for _fold in training.ALLOWED_PILOT_FOLDS
        ]
        self.assertTrue(
            all(pair == transform_pairs[0] for pair in transform_pairs[1:])
        )
        source = inspect.getsource(training.build_fold_datasets)
        self.assertEqual(source.count("build_transforms()"), 1)


class ConvNeXtTinySplitAndIsolationTests(unittest.TestCase):
    def test_frozen_manifest_path_and_sha256_are_locked(self):
        audit = training.preflight_fold(4)
        self.assertEqual(audit["manifest_sha256"], training.EXPECTED_MANIFEST_SHA256)
        with self.assertRaises(ValueError):
            training.preflight_fold(4, manifest_path=Path("alternate_manifest.csv"))

    def test_all_fold_counts_and_roles_are_frozen(self):
        expected = {
            1: (19_805, 5_911),
            2: (18_524, 7_192),
            3: (19_216, 6_500),
            4: (19_603, 6_113),
        }
        for fold, (train_count, validation_count) in expected.items():
            with self.subTest(fold=fold):
                audit = training.preflight_fold(fold)
                self.assertEqual(audit["train_count"], train_count)
                self.assertEqual(audit["validation_count"], validation_count)
                self.assertEqual(audit["selected_role"], "model_development")
                self.assertEqual(
                    set(audit["train_records"]["top_level_role"]),
                    {"model_development"},
                )
                self.assertEqual(
                    set(audit["validation_records"]["top_level_role"]),
                    {"model_development"},
                )
                self.assertEqual(audit["forbidden_roles_accessed"], [])
                self.assertFalse(audit["final_test_accessed"])
                self.assertEqual(
                    len(audit["train_records"])
                    + len(audit["validation_records"]),
                    25_716,
                )

    def test_all_validation_dates_are_frozen(self):
        expected = {
            1: ["2017-06-13", "2017-06-28"],
            2: ["2017-06-14", "2017-06-29"],
            3: ["2017-06-16", "2017-06-26"],
            4: ["2017-06-20", "2017-06-21"],
        }
        for fold, validation_dates in expected.items():
            with self.subTest(fold=fold):
                audit = training.preflight_fold(fold)
                self.assertEqual(audit["validation_dates"], validation_dates)

    def test_folds_1_2_3_4_are_allowed_and_other_folds_are_rejected(self):
        self.assertEqual(training.ALLOWED_PILOT_FOLDS, (1, 2, 3, 4))
        for fold in (0, 5, -1, True):
            with self.subTest(fold=fold):
                with self.assertRaises(ValueError):
                    training.validate_pilot_fold(fold)
                with self.assertRaises(ValueError):
                    runner.validate_pilot_fold(fold)
        for fold in (1, 2, 3, 4):
            with self.subTest(fold=fold):
                training.validate_pilot_fold(fold)
                runner.validate_pilot_fold(fold)

    def test_output_namespace_is_fully_isolated(self):
        root = (
            training.PROJECT_ROOT
            / "outputs"
            / "date_grouped_v1"
            / "convnext_tiny_image_only_v1"
            / "pilot"
        )
        output_dirs = []
        for fold in (1, 2, 3, 4):
            expected = root / f"fold_{fold}_seed_42"
            output_dirs.append(training.expected_output_dir(fold).resolve())
            self.assertEqual(
                training.expected_output_dir(fold).resolve(), expected.resolve()
            )
            self.assertNotIn("resnet50", str(expected).lower())
            self.assertNotIn("dinov2", str(expected).lower())
            self.assertNotIn("photometric", str(expected).lower())
        self.assertEqual(len(set(output_dirs)), 4)

    def test_existing_fold3_and_fold4_outputs_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            isolated_root = Path(temporary_directory) / "pilot"
            with mock.patch.object(training, "OUTPUT_ROOT", isolated_root):
                for fold in (3, 4):
                    output_dir = training.expected_output_dir(fold)
                    output_dir.mkdir(parents=True)
                    marker = output_dir / "existing_result.marker"
                    marker.write_text("preserve", encoding="utf-8")
                    with self.assertRaises(FileExistsError):
                        training._prepare_output_directory(output_dir, fold, 42)
                    self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_config_declares_fold1_and_fold2_counts(self):
        config = training.load_training_config()
        self.assertEqual(config["fold1_train_count"], 19_805)
        self.assertEqual(config["fold1_validation_count"], 5_911)
        self.assertEqual(config["fold2_train_count"], 18_524)
        self.assertEqual(config["fold2_validation_count"], 7_192)

    def test_future_required_outputs_and_prediction_schema(self):
        self.assertEqual(
            set(training.REQUIRED_OUTPUT_FILES),
            {
                "best_model.pth",
                "final_metrics.json",
                "history.csv",
                "predictions.csv",
                "run_metadata.json",
                "config_snapshot.json",
            },
        )
        self.assertEqual(
            training.PREDICTION_FIELDS,
            (
                "filename",
                "date",
                "fold",
                "true_L",
                "pred_L",
                "error",
                "abs_error",
            ),
        )

    def test_all_fold_runners_route_without_starting_training_in_test(self):
        for fold in (1, 2, 3, 4):
            with self.subTest(fold=fold):
                with mock.patch.object(training, "run_training") as run_training:
                    runner.main(["--fold", str(fold)])
                run_training.assert_called_once()
                args = run_training.call_args.args[0]
                self.assertEqual(args.fold, fold)
                self.assertEqual(
                    args.output_dir.resolve(),
                    training.expected_output_dir(fold).resolve(),
                )


if __name__ == "__main__":
    unittest.main()
