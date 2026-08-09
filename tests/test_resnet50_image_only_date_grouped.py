import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from experiments import run_resnet50_image_only_date_grouped_cv as image_runner
from experiments import run_resnet50_with_i_date_grouped_cv as reference_runner
from experiments import train_resnet50_image_only_date_grouped as image_training
from experiments import train_resnet50_with_i_date_grouped as reference_training
from models.resnet50_image_only import SolarResNet50ImageOnly


class ResNet50ImageOnlyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SolarResNet50ImageOnly(
            dropout=0.3,
            use_pretrained=False,
        ).eval()

    def test_forward_accepts_images_only_and_has_no_irradiance_branch(self):
        parameters = list(inspect.signature(SolarResNet50ImageOnly.forward).parameters)
        self.assertEqual(parameters, ["self", "images"])
        self.assertFalse(hasattr(self.model, "i_branch"))
        self.assertIsInstance(self.model.backbone.fc, nn.Identity)
        self.assertEqual(self.model.regressor[0].in_features, 2048)
        self.assertNotEqual(self.model.regressor[0].in_features, 2064)
        self.assertEqual(self.model.regressor[0].out_features, 128)
        self.assertIsInstance(self.model.regressor[1], nn.ReLU)
        self.assertIsInstance(self.model.regressor[2], nn.Dropout)
        self.assertEqual(self.model.regressor[2].p, 0.3)
        self.assertEqual(self.model.regressor[3].out_features, 1)

    def test_output_shape_and_sigmoid_range(self):
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            predictions = self.model(images)
        self.assertEqual(tuple(predictions.shape), (2, 1))
        self.assertTrue(torch.all(predictions >= 0.0))
        self.assertTrue(torch.all(predictions <= 1.0))

    def test_irradiance_placeholders_cannot_affect_predictions(self):
        images = torch.randn(2, 3, 64, 64)
        labels = torch.tensor([0.1, 0.9])
        time_features = torch.tensor([10.0, 11.0])
        first_batch = (images, labels, torch.zeros(2), time_features)
        second_batch = (images, labels, torch.full((2,), 999.0), time_features)

        first_images, first_labels = image_training.unpack_batch(first_batch)
        second_images, second_labels = image_training.unpack_batch(second_batch)
        self.assertIs(first_images, images)
        self.assertIs(second_images, images)
        self.assertIs(first_labels, labels)
        self.assertIs(second_labels, labels)

        with torch.no_grad():
            first_predictions = self.model(first_images)
            second_predictions = self.model(second_images)
        torch.testing.assert_close(first_predictions, second_predictions)


class ResNet50ImageOnlyFairnessTests(unittest.TestCase):
    def test_frozen_configuration_matches_resnet50_with_i(self):
        image_config = image_training.load_training_config()
        reference_config = reference_training.load_training_config()
        frozen_fields = (
            "schema_version",
            "split_version",
            "seed",
            "epochs",
            "batch_size",
            "num_workers",
            "learning_rate",
            "weight_decay",
            "patience",
            "amp",
            "pretrained",
        )
        for field in frozen_fields:
            self.assertEqual(image_config[field], reference_config[field], field)
        self.assertEqual(image_config["model_name"], "SolarResNet50ImageOnly")
        self.assertEqual(
            image_config["training_version"],
            "resnet50_image_only_date_grouped_v1",
        )
        image_runner.validate_frozen_config()

    def test_transforms_and_normalization_match_resnet50_with_i(self):
        self.assertEqual(
            image_training.PREPROCESSING_DESCRIPTION,
            reference_training.PREPROCESSING_DESCRIPTION,
        )
        image_transforms = image_training.build_transforms()
        reference_transforms = reference_training.build_transforms()
        self.assertEqual(repr(image_transforms), repr(reference_transforms))

    def test_metric_implementation_is_identical(self):
        self.assertEqual(
            inspect.getsource(image_training.MetricAccumulator),
            inspect.getsource(reference_training.MetricAccumulator),
        )
        self.assertEqual(
            inspect.getsource(image_training.sample_weighted_average),
            inspect.getsource(reference_training.sample_weighted_average),
        )

    def test_optimizer_scheduler_loss_and_early_stopping_semantics(self):
        source = inspect.getsource(image_training.run_training)
        required_fragments = (
            "nn.MSELoss(reduction=\"mean\")",
            "torch.optim.AdamW(",
            "ReduceLROnPlateau(",
            "mode=\"min\"",
            "factor=0.5",
            "patience=2",
            'scheduler.step(validation_metrics["rmse"])',
            'improved = validation_metrics["rmse"] < best_validation_rmse',
            "if epochs_without_improvement >= args.patience:",
            'checkpoint_kind="best"',
        )
        for fragment in required_fragments:
            self.assertIn(fragment, source)

    def test_training_epoch_has_image_only_model_call(self):
        source = inspect.getsource(image_training.run_epoch)
        self.assertIn("images, labels = unpack_batch(batch)", source)
        self.assertIn("predictions = model(images)", source)
        self.assertNotIn("irradiance", source)
        self.assertNotIn("batch[2]", inspect.getsource(image_training.unpack_batch))
        images = torch.randn(2, 3, 8, 8)
        labels = torch.tensor([0.2, 0.8])
        unpacked_images, unpacked_labels = image_training.unpack_batch(
            (images, labels)
        )
        self.assertIs(unpacked_images, images)
        self.assertIs(unpacked_labels, labels)


class ResNet50ImageOnlySplitSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audits = {
            fold: image_training.preflight_fold(
                image_training.DEFAULT_MANIFEST,
                image_training.DEFAULT_IMAGE_ROOT,
                fold,
            )
            for fold in range(1, 5)
        }

    def test_fold_counts_match_frozen_resnet50_with_i_protocol(self):
        self.assertEqual(
            image_training.EXPECTED_FOLD_COUNTS,
            reference_training.EXPECTED_FOLD_COUNTS,
        )
        for fold, expected in image_training.EXPECTED_FOLD_COUNTS.items():
            audit = self.audits[fold]
            self.assertEqual(
                (len(audit["train_records"]), len(audit["validation_records"])),
                expected,
            )

    def test_each_fold_uses_only_model_development_and_is_date_isolated(self):
        for audit in self.audits.values():
            self.assertEqual(set(audit["train_records"]["top_level_role"]), {"model_development"})
            self.assertEqual(set(audit["validation_records"]["top_level_role"]), {"model_development"})
            self.assertTrue(
                set(audit["train_dates"]).isdisjoint(audit["validation_dates"])
            )
            self.assertEqual(
                audit["forbidden_role_counts"],
                {
                    "cp_calibration": 0,
                    "decision_development": 0,
                    "final_test": 0,
                },
            )

    def test_fold_dates_and_manifest_match_resnet50_with_i(self):
        for fold, image_audit in self.audits.items():
            reference_audit = reference_training.preflight_fold(
                reference_training.DEFAULT_MANIFEST,
                reference_training.DEFAULT_IMAGE_ROOT,
                fold,
            )
            self.assertEqual(image_audit["train_dates"], reference_audit["train_dates"])
            self.assertEqual(
                image_audit["validation_dates"],
                reference_audit["validation_dates"],
            )
            self.assertEqual(
                image_audit["manifest_sha256"],
                reference_audit["manifest_sha256"],
            )


class ResNet50ImageOnlyOutputIsolationTests(unittest.TestCase):
    def test_formal_output_root_is_image_only(self):
        expected = (
            image_runner.PROJECT_ROOT
            / "outputs"
            / "date_grouped_v1"
            / "resnet50_image_only"
            / "formal_cv"
        ).resolve()
        self.assertEqual(image_runner.FORMAL_OUTPUT_ROOT.resolve(), expected)
        self.assertNotIn("resnet50_with_i", image_runner.FORMAL_OUTPUT_ROOT.parts)
        image_runner.validate_output_root_isolation()

    def test_output_guard_rejects_resnet50_with_i_root(self):
        with mock.patch.object(
            image_runner,
            "FORMAL_OUTPUT_ROOT",
            reference_runner.FORMAL_OUTPUT_ROOT,
        ):
            with self.assertRaises(RuntimeError):
                image_runner.validate_output_root_isolation()

    def test_runner_exposes_no_overwrite_or_resume_option(self):
        parser_source = inspect.getsource(image_runner.parse_args)
        self.assertNotIn("overwrite", parser_source)
        self.assertNotIn("resume", parser_source)
        runner_source = inspect.getsource(image_runner)
        self.assertNotIn("torch.load", runner_source)
        args = image_runner.parse_args(["--start-fold", "1", "--end-fold", "4"])
        self.assertEqual((args.start_fold, args.end_fold), (1, 4))

    def test_existing_fold_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            existing = Path(temp_dir) / "fold_1_seed_42"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                image_runner.ensure_new_fold_directory(existing)


if __name__ == "__main__":
    unittest.main()
