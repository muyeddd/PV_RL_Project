from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import run_dinov2_vits14_frozen_regression_pilot as runner
from experiments import train_dinov2_vits14_frozen_regression_date_grouped as training
from models.dinov2_frozen_feature_regressor import DINOv2FrozenFeatureRegressor


class FormalFeatureCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = training.load_config()
        cls.cache = training.load_and_validate_feature_cache(cls.config)

    def test_feature_shape_is_25716_by_384(self):
        self.assertEqual(self.cache.features.shape, (25716, 384))

    def test_feature_dtype_and_finiteness(self):
        self.assertEqual(self.cache.features.dtype, np.float32)
        self.assertTrue(np.isfinite(self.cache.features).all())

    def test_metadata_count_and_columns(self):
        self.assertEqual(len(self.cache.metadata), 25716)
        self.assertEqual(tuple(self.cache.metadata.columns), training.METADATA_COLUMNS)

    def test_features_and_metadata_rows_are_strictly_aligned(self):
        self.assertTrue(
            np.array_equal(
                self.cache.metadata["row_index"].to_numpy(), np.arange(25716)
            )
        )
        filenames = self.cache.metadata["filename"].astype(str).tolist()
        self.assertEqual(filenames, sorted(filenames))
        self.assertEqual(
            training.sha256_ordered_strings(filenames),
            self.cache.dataset_fingerprint["ordered_filenames_sha256"],
        )

    def test_feature_cache_sha256_matches_formal_manifest(self):
        self.assertEqual(
            self.cache.file_sha256["features.npy"],
            self.cache.feature_manifest["features_sha256"],
        )
        self.assertEqual(
            self.cache.file_sha256["metadata.csv"],
            self.cache.feature_manifest["metadata_sha256"],
        )
        self.assertEqual(
            self.cache.file_sha256["dataset_fingerprint.json"],
            self.cache.feature_manifest["dataset_fingerprint_sha256"],
        )

    def test_cache_is_model_development_only(self):
        self.assertEqual(
            self.cache.feature_manifest["top_level_role"], "model_development"
        )
        self.assertEqual(
            self.cache.dataset_fingerprint["top_level_role"], "model_development"
        )

    def test_cache_does_not_use_irradiance(self):
        self.assertFalse(
            self.cache.feature_manifest["irradiance_used_as_model_input"]
        )
        self.assertNotIn("I", self.cache.metadata.columns)
        self.assertNotIn("time", self.cache.metadata.columns)

    def test_fold_definition_matches_date_grouped_v1(self):
        split_config = json.loads(
            (ROOT / "configs" / "splits" / "date_grouped_v1.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {
            int(item["fold"]): sorted(item["validation_dates"])
            for item in split_config["cv_folds"]
        }
        observed = {
            int(fold): sorted(frame["date"].astype(str).unique().tolist())
            for fold, frame in self.cache.metadata.groupby("cv_validation_fold")
        }
        self.assertEqual(observed, expected)

    def test_fold3_training_and_validation_counts(self):
        train_indices, validation_indices = training.fold_indices(self.cache, 3)
        self.assertEqual((len(train_indices), len(validation_indices)), (19216, 6500))

    def test_fold4_training_and_validation_counts(self):
        train_indices, validation_indices = training.fold_indices(self.cache, 4)
        self.assertEqual((len(train_indices), len(validation_indices)), (19603, 6113))

    def test_feature_dataset_reads_cache_rows_only(self):
        train_indices, _ = training.fold_indices(self.cache, 3)
        dataset = training.FeatureRegressionDataset(self.cache, train_indices[:2])
        feature, label = dataset[0]
        self.assertEqual(tuple(feature.shape), (384,))
        self.assertEqual(feature.dtype, torch.float32)
        self.assertEqual(label.ndim, 0)


class RegressionModelContractTests(unittest.TestCase):
    def setUp(self):
        self.model = DINOv2FrozenFeatureRegressor()

    def test_model_input_and_output_dimensions(self):
        output = self.model(torch.zeros(5, 384))
        self.assertEqual(tuple(output.shape), (5, 1))

    def test_regression_head_is_exactly_384_to_128_to_1(self):
        layers = list(self.model.regressor)
        self.assertEqual(
            [type(layer) for layer in layers],
            [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.Sigmoid],
        )
        self.assertEqual((layers[0].in_features, layers[0].out_features), (384, 128))
        self.assertEqual((layers[3].in_features, layers[3].out_features), (128, 1))
        self.assertEqual(layers[2].p, 0.3)

    def test_sigmoid_is_final_activation(self):
        self.assertIsInstance(list(self.model.regressor)[-1], nn.Sigmoid)

    def test_model_accepts_only_feature_argument(self):
        parameters = list(inspect.signature(self.model.forward).parameters)
        self.assertEqual(parameters, ["features"])


class TrainingConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = training.load_config()

    def test_loss_is_mse(self):
        self.assertIsInstance(training.build_loss(self.config), nn.MSELoss)

    def test_optimizer_is_adamw_with_frozen_values(self):
        model = training.build_model(self.config)
        optimizer = training.build_optimizer(model, self.config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-4)

    def test_scheduler_and_early_stopping_values(self):
        model = training.build_model(self.config)
        optimizer = training.build_optimizer(model, self.config)
        scheduler = training.build_scheduler(optimizer, self.config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(scheduler.patience, 2)
        self.assertEqual(self.config["early_stopping_patience"], 8)
        self.assertEqual(self.config["selection_metric"], "validation_rmse")

    def test_batch_seed_epoch_worker_and_feature_standardization_values(self):
        self.assertEqual(self.config["seed"], 42)
        self.assertEqual(self.config["batch_size"], 256)
        self.assertEqual(self.config["num_workers"], 0)
        self.assertEqual(self.config["max_epochs"], 50)
        self.assertEqual(self.config["feature_standardization"], "none")
        self.assertFalse(self.config["amp"])

    def test_metrics_include_prediction_compression_diagnostics(self):
        metrics = training.compute_metrics(
            np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.4, 0.8])
        )
        required = {
            "r2",
            "rmse",
            "mae",
            "prediction_mean",
            "prediction_std",
            "true_mean",
            "true_std",
            "bias",
            "pred_min",
            "pred_max",
            "true_min",
            "true_max",
        }
        self.assertTrue(required.issubset(metrics))


class PilotSafetyTests(unittest.TestCase):
    def test_fold1_and_fold2_are_rejected(self):
        for fold in (1, 2):
            with self.subTest(fold=fold), self.assertRaisesRegex(ValueError, "only Fold 3"):
                training.validate_pilot_fold(fold)

    def test_fold3_and_fold4_are_allowed(self):
        training.validate_pilot_fold(3)
        training.validate_pilot_fold(4)

    def test_runner_parser_rejects_fold1_and_fold2(self):
        for fold in (1, 2):
            with self.subTest(fold=fold), self.assertRaises(SystemExit):
                runner.parse_args(["--fold", str(fold)])

    def test_runner_allows_fold3_and_fold4_without_starting_training(self):
        for fold in (3, 4):
            with self.subTest(fold=fold):
                with mock.patch.object(training, "run_training", return_value={"fold": fold}) as mocked:
                    self.assertEqual(runner.run_pilot(fold), {"fold": fold})
                    mocked.assert_called_once_with(fold)

    def test_output_paths_are_isolated(self):
        for fold in (3, 4):
            path = training.expected_output_dir(fold)
            self.assertEqual(
                path,
                ROOT
                / "outputs"
                / "date_grouped_v1"
                / "dinov2_vits14_frozen_regression_v1"
                / "pilot"
                / f"fold_{fold}_seed_42",
            )
            lowered = str(path).lower()
            self.assertNotIn("resnet50_image_only", lowered)
            self.assertNotIn("photometric", lowered)
            self.assertNotIn("features\\dinov2", lowered)

    def test_training_source_has_no_raw_image_access(self):
        source = inspect.getsource(training)
        forbidden_tokens = ("Image.open", "from PIL", "import PIL", "torchvision.transforms")
        self.assertFalse(any(token in source for token in forbidden_tokens))

    def test_training_source_has_no_feature_extraction_or_auxiliary_model_input(self):
        source = inspect.getsource(training)
        self.assertNotIn("torch.hub.load", source)
        model_parameters = list(
            inspect.signature(DINOv2FrozenFeatureRegressor.forward).parameters
        )
        self.assertEqual(model_parameters, ["self", "features"])

    def test_required_output_contract(self):
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


if __name__ == "__main__":
    unittest.main()
