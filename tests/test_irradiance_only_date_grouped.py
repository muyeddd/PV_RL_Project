import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments import run_irradiance_only_date_grouped_cv as runner
from experiments import run_resnet50_image_only_date_grouped_cv as image_runner
from experiments import run_resnet50_with_i_date_grouped_cv as with_i_runner
from experiments import train_irradiance_only_date_grouped as training
from experiments import train_resnet50_with_i_date_grouped as reference_training
from models.irradiance_only_mlp import IrradianceOnlyMLP
from utils.irradiance_only_dataset import (
    IrradianceOnlyDataset,
    load_irradiance_only_fold_records,
)
from utils.parser import parse_filename


class IrradianceOnlyModelTests(unittest.TestCase):
    def test_model_structure_and_parameter_count(self):
        model = IrradianceOnlyMLP(dropout=0.3)
        layers = list(model.regressor)
        self.assertEqual(len(layers), 6)
        self.assertIsInstance(layers[0], nn.Linear)
        self.assertEqual((layers[0].in_features, layers[0].out_features), (1, 16))
        self.assertIsInstance(layers[1], nn.ReLU)
        self.assertIsInstance(layers[2], nn.Dropout)
        self.assertEqual(layers[2].p, 0.3)
        self.assertIsInstance(layers[3], nn.Linear)
        self.assertEqual((layers[3].in_features, layers[3].out_features), (16, 16))
        self.assertIsInstance(layers[4], nn.ReLU)
        self.assertIsInstance(layers[5], nn.Linear)
        self.assertEqual((layers[5].in_features, layers[5].out_features), (16, 1))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 321)

    def test_forward_shape_and_sigmoid_range(self):
        model = IrradianceOnlyMLP(dropout=0.3).eval()
        irradiance = torch.tensor([0.0, 0.5, 1.0061254902])
        with torch.no_grad():
            predictions = model(irradiance)
        self.assertEqual(tuple(predictions.shape), (3, 1))
        self.assertTrue(torch.all(predictions >= 0.0))
        self.assertTrue(torch.all(predictions <= 1.0))

    def test_forward_rejects_non_vector_input(self):
        model = IrradianceOnlyMLP()
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 1))


class IrradianceOnlyDatasetTests(unittest.TestCase):
    def setUp(self):
        self.filenames = [
            "solar_Tue_Jun_13_10__0__0_2017_L_0.25_I_1.0061254902.jpg",
            "solar_Tue_Jun_13_10__0__1_2017_L_0.75_I_0.00268235294118.jpg",
        ]

    def test_dataset_preserves_order_and_returns_only_label_and_irradiance(self):
        dataset = IrradianceOnlyDataset(self.filenames)
        self.assertEqual(dataset.files, self.filenames)
        self.assertEqual(len(dataset), 2)
        first = dataset[0]
        second = dataset[1]
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertEqual(first[0].dtype, torch.float32)
        self.assertEqual(first[1].dtype, torch.float32)
        self.assertAlmostEqual(first[0].item(), 0.25)
        self.assertAlmostEqual(first[1].item(), 1.0061254902, places=6)
        self.assertAlmostEqual(second[0].item(), 0.75)
        self.assertAlmostEqual(second[1].item(), 0.00268235294118, places=8)

    def test_dataset_reuses_parse_filename(self):
        with mock.patch(
            "utils.irradiance_only_dataset.parse_filename",
            wraps=parse_filename,
        ) as parser:
            dataset = IrradianceOnlyDataset(self.filenames)
            dataset[0]
        parser.assert_called_once_with(self.filenames[0])

    def test_dataset_has_no_image_decoder_or_directory_scan(self):
        source = inspect.getsource(
            __import__(
                "utils.irradiance_only_dataset",
                fromlist=["IrradianceOnlyDataset"],
            )
        )
        forbidden = (
            "from PIL",
            "import PIL",
            "Image.open",
            "os.listdir",
            "iterdir(",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_raw_irradiance_is_not_standardized_clipped_or_unit_constrained(self):
        dataset = IrradianceOnlyDataset(self.filenames)
        _, irradiance = dataset[0]
        self.assertGreater(irradiance.item(), 1.0)
        self.assertEqual(training.INPUT_PREPROCESSING["standardized"], False)
        self.assertEqual(training.INPUT_PREPROCESSING["clipped"], False)
        self.assertEqual(
            training.INPUT_PREPROCESSING["strict_unit_interval_assumed"],
            False,
        )
        self.assertIsNone(training.INPUT_PREPROCESSING["physical_unit"])


class IrradianceOnlyTrainingProtocolTests(unittest.TestCase):
    def test_label_and_prediction_shapes_match_exactly(self):
        class RecordingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
                self.seen_shape = None

            def forward(self, irradiance):
                self.seen_shape = tuple(irradiance.shape)
                return torch.sigmoid(self.linear(irradiance.unsqueeze(1)))

        labels = torch.tensor([0.2, 0.8], dtype=torch.float32)
        irradiance = torch.tensor([0.1, 1.0], dtype=torch.float32)
        loader = DataLoader(TensorDataset(labels, irradiance), batch_size=2)
        model = RecordingModel()
        metrics = training.run_epoch(
            model=model,
            loader=loader,
            criterion=nn.MSELoss(reduction="mean"),
            device=torch.device("cpu"),
            amp_enabled=False,
            phase="validation",
        )
        self.assertEqual(model.seen_shape, (2,))
        self.assertEqual(metrics["sample_count"], 2)
        source = inspect.getsource(training.run_epoch)
        self.assertIn("labels, irradiance = unpack_batch(batch)", source)
        self.assertIn(".unsqueeze(1)", source)
        self.assertIn("predictions.shape != labels.shape", source)

    def test_shape_guard_rejects_broadcasting(self):
        class BadShapeModel(nn.Module):
            def forward(self, irradiance):
                return irradiance

        loader = DataLoader(
            TensorDataset(torch.tensor([0.2, 0.8]), torch.tensor([0.1, 1.0])),
            batch_size=2,
        )
        with self.assertRaisesRegex(RuntimeError, "broadcasting"):
            training.run_epoch(
                model=BadShapeModel(),
                loader=loader,
                criterion=nn.MSELoss(reduction="mean"),
                device=torch.device("cpu"),
                amp_enabled=False,
                phase="validation",
            )

    def test_frozen_hyperparameters_match_formal_resnet_protocol(self):
        config = training.load_training_config()
        reference = reference_training.load_training_config()
        common_fields = (
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
        )
        for field in common_fields:
            self.assertEqual(config[field], reference[field], field)
        self.assertFalse(config["pretrained"])
        self.assertEqual(config["optimizer"], "AdamW")
        self.assertEqual(
            config["scheduler"],
            "ReduceLROnPlateau(mode=min,factor=0.5,patience=2)",
        )
        self.assertEqual(config["loss"], "MSELoss(reduction=mean)")
        self.assertEqual(config["early_stopping_metric"], "validation_rmse")
        self.assertEqual(config["checkpoint_selection"], "minimum_validation_rmse")
        runner.validate_frozen_config()

    def test_metrics_are_identical_to_formal_resnet_protocol(self):
        self.assertEqual(
            inspect.getsource(training.MetricAccumulator),
            inspect.getsource(reference_training.MetricAccumulator),
        )
        self.assertEqual(
            inspect.getsource(training.sample_weighted_average),
            inspect.getsource(reference_training.sample_weighted_average),
        )

    def test_optimizer_scheduler_loss_and_early_stopping_semantics(self):
        source = inspect.getsource(training.run_training)
        required = (
            'nn.MSELoss(reduction="mean")',
            "torch.optim.AdamW(",
            "ReduceLROnPlateau(",
            'mode="min"',
            "factor=0.5",
            "patience=2",
            'scheduler.step(validation_metrics["rmse"])',
            'improved = validation_metrics["rmse"] < best_validation_rmse',
            "if epochs_without_improvement >= args.patience:",
            'checkpoint_kind="best"',
        )
        for fragment in required:
            self.assertIn(fragment, source)

    def test_trainer_has_no_image_or_checkpoint_input_path(self):
        source = inspect.getsource(training)
        forbidden = (
            "from PIL",
            "import PIL",
            "Image.open",
            "torchvision",
            "resnet",
            "transforms",
            "ImageNet",
            "torch.load",
            "--image-root",
            "--pretrained",
            "--overwrite",
            "--resume",
        )
        for token in forbidden:
            self.assertNotIn(token, source)


class IrradianceOnlySplitSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audits = {
            fold: training.preflight_fold(training.DEFAULT_MANIFEST, fold)
            for fold in range(1, 5)
        }

    def test_fold_counts_dates_and_manifest_match_formal_protocol(self):
        self.assertEqual(
            training.EXPECTED_FOLD_COUNTS,
            reference_training.EXPECTED_FOLD_COUNTS,
        )
        for fold, expected in training.EXPECTED_FOLD_COUNTS.items():
            audit = self.audits[fold]
            self.assertEqual(
                (len(audit["train_records"]), len(audit["validation_records"])),
                expected,
            )
            reference_audit = reference_training.preflight_fold(
                reference_training.DEFAULT_MANIFEST,
                reference_training.DEFAULT_IMAGE_ROOT,
                fold,
            )
            self.assertEqual(audit["train_dates"], reference_audit["train_dates"])
            self.assertEqual(
                audit["validation_dates"],
                reference_audit["validation_dates"],
            )
            self.assertEqual(
                audit["manifest_sha256"],
                reference_audit["manifest_sha256"],
            )
            self.assertEqual(
                audit["train_records"]["filename"].tolist(),
                reference_audit["train_records"]["filename"].tolist(),
            )
            self.assertEqual(
                audit["validation_records"]["filename"].tolist(),
                reference_audit["validation_records"]["filename"].tolist(),
            )

    def test_only_model_development_enters_datasets(self):
        for audit in self.audits.values():
            self.assertEqual(
                set(audit["train_records"]["top_level_role"]),
                {"model_development"},
            )
            self.assertEqual(
                set(audit["validation_records"]["top_level_role"]),
                {"model_development"},
            )
            self.assertEqual(
                audit["forbidden_role_counts"],
                {
                    "cp_calibration": 0,
                    "decision_development": 0,
                    "final_test": 0,
                },
            )
            self.assertTrue(
                set(audit["train_dates"]).isdisjoint(audit["validation_dates"])
            )

    def test_preflight_does_not_parse_any_filename_values(self):
        with mock.patch(
            "utils.irradiance_only_dataset.parse_filename",
            side_effect=AssertionError("preflight must not parse L or I"),
        ):
            training.preflight_fold(training.DEFAULT_MANIFEST, 1)

    def test_record_loader_does_not_add_image_paths(self):
        train, validation = load_irradiance_only_fold_records(
            training.DEFAULT_MANIFEST,
            1,
        )
        self.assertNotIn("image_path", train.columns)
        self.assertNotIn("image_path", validation.columns)


class IrradianceOnlyOutputIsolationTests(unittest.TestCase):
    def test_formal_output_root_is_exact_and_isolated(self):
        expected = (
            runner.PROJECT_ROOT
            / "outputs"
            / "date_grouped_v1"
            / "irradiance_only"
            / "formal_cv"
        ).resolve()
        self.assertEqual(runner.FORMAL_OUTPUT_ROOT.resolve(), expected)
        runner.validate_output_root_isolation()

    def test_output_guard_rejects_both_resnet_roots(self):
        for forbidden_root in (
            with_i_runner.FORMAL_OUTPUT_ROOT,
            image_runner.FORMAL_OUTPUT_ROOT,
        ):
            with self.subTest(forbidden_root=forbidden_root):
                with mock.patch.object(runner, "FORMAL_OUTPUT_ROOT", forbidden_root):
                    with self.assertRaises(RuntimeError):
                        runner.validate_output_root_isolation()

    def test_runner_has_no_overwrite_resume_or_checkpoint_loading_option(self):
        parser_source = inspect.getsource(runner.parse_args)
        self.assertNotIn("overwrite", parser_source)
        self.assertNotIn("resume", parser_source)
        source = inspect.getsource(runner)
        self.assertNotIn("torch.load", source)
        args = runner.parse_args(["--start-fold", "1", "--end-fold", "4"])
        self.assertEqual((args.start_fold, args.end_fold), (1, 4))

    def test_existing_fold_directory_and_summary_are_refused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            existing = Path(temp_dir) / "fold_1_seed_42"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                runner.ensure_new_fold_directory(existing)
        summary_source = inspect.getsource(runner.write_cv_summary)
        self.assertIn("csv_path.exists() or json_path.exists()", summary_source)
        self.assertIn("FileExistsError", summary_source)


if __name__ == "__main__":
    unittest.main()
