from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import train_dinov2_vits14_regression_diagnostic_fold4 as diagnostic
from models.dinov2_regression_diagnostic_head import DINOv2DiagnosticRegressionHead


class DiagnosticDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = diagnostic.load_config()
        cls.data = diagnostic.prepare_diagnostic_data(cls.config)

    def test_feature_cache_is_formal_model_development_only(self):
        self.assertEqual(
            self.data.cache.feature_manifest["top_level_role"], "model_development"
        )
        self.assertFalse(
            self.data.cache.feature_manifest["irradiance_used_as_model_input"]
        )

    def test_only_fold4_indices_are_used(self):
        folds = self.data.cache.metadata["cv_validation_fold"].to_numpy()
        self.assertTrue(np.all(folds[self.data.validation_indices] == 4))
        self.assertTrue(np.all(folds[self.data.training_indices] != 4))

    def test_fold4_counts_are_exact(self):
        self.assertEqual(len(self.data.training_indices), 19603)
        self.assertEqual(len(self.data.validation_indices), 6113)

    def test_standardization_is_fitted_on_training_count_only(self):
        self.assertEqual(self.data.standardization.fitted_row_count, 19603)

    def test_standardized_training_dimensions_have_zero_mean_and_unit_std(self):
        training = self.data.standardized_features[self.data.training_indices]
        active = ~self.data.standardization.near_zero_mask
        means = training[:, active].mean(axis=0, dtype=np.float64)
        stds = training[:, active].std(axis=0, dtype=np.float64)
        self.assertLess(float(np.max(np.abs(means))), 1e-5)
        self.assertLess(float(np.max(np.abs(stds - 1.0))), 1e-5)

    def test_all_standardized_features_are_float32_and_finite(self):
        self.assertEqual(self.data.standardized_features.dtype, np.float32)
        self.assertEqual(self.data.standardized_features.shape, (25716, 384))
        self.assertTrue(np.isfinite(self.data.standardized_features).all())

    def test_validation_uses_training_statistics_without_refitting(self):
        validation_raw = np.asarray(
            self.data.cache.features[self.data.validation_indices], dtype=np.float32
        )
        expected = (
            validation_raw - self.data.standardization.mean[None, :]
        ) / self.data.standardization.safe_std[None, :]
        observed = self.data.standardized_features[self.data.validation_indices]
        self.assertTrue(np.array_equal(observed, expected.astype(np.float32)))

    def test_training_only_fit_ignores_validation_outlier(self):
        features = np.zeros((3, 384), dtype=np.float32)
        features[0, 0] = 0.0
        features[1, 0] = 2.0
        features[2, 0] = 1000.0
        stats = diagnostic.compute_training_standardization(
            features, np.array([0, 1]), epsilon=1e-6
        )
        self.assertEqual(float(stats.mean[0]), 1.0)
        self.assertEqual(float(stats.std[0]), 1.0)

    def test_near_zero_dimensions_are_safely_handled(self):
        features = np.ones((2, 384), dtype=np.float32)
        stats = diagnostic.compute_training_standardization(
            features, np.array([0, 1]), epsilon=1e-6
        )
        self.assertTrue(stats.near_zero_mask.all())
        self.assertTrue(np.equal(stats.safe_std, 1.0).all())
        standardized = diagnostic.apply_standardization(features, stats)
        self.assertTrue(np.equal(standardized, 0.0).all())


class VariantContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = diagnostic.load_config()

    def test_exactly_three_variants_are_defined(self):
        self.assertEqual(tuple(self.config["variants"]), ("A", "B", "C"))

    def test_variant_a_is_standardized_sigmoid_lr1e4(self):
        variant = self.config["variants"]["A"]
        self.assertEqual(variant["name"], "standardized_sigmoid_lr1e4")
        self.assertEqual(variant["learning_rate"], 1e-4)
        self.assertEqual(variant["output_activation"], "Sigmoid")

    def test_variant_b_differs_from_a_only_by_learning_rate_and_identity(self):
        a = self.config["variants"]["A"]
        b = self.config["variants"]["B"]
        self.assertEqual(b["name"], "standardized_sigmoid_lr1e3")
        self.assertEqual(b["learning_rate"], 1e-3)
        self.assertEqual(b["output_activation"], "Sigmoid")
        self.assertEqual(a["output_activation"], b["output_activation"])

    def test_variant_c_is_linear_lr1e3(self):
        variant = self.config["variants"]["C"]
        self.assertEqual(variant["name"], "standardized_linear_lr1e3")
        self.assertEqual(variant["learning_rate"], 1e-3)
        self.assertEqual(variant["output_activation"], "Linear")

    def test_sigmoid_head_has_exact_structure(self):
        model = diagnostic.build_model(self.config, "A")
        layers = list(model.regressor)
        self.assertEqual(
            [type(layer) for layer in layers],
            [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.Sigmoid],
        )
        self.assertEqual((layers[0].in_features, layers[0].out_features), (384, 128))
        self.assertEqual((layers[3].in_features, layers[3].out_features), (128, 1))

    def test_linear_head_has_no_sigmoid_or_other_final_activation(self):
        model = diagnostic.build_model(self.config, "C")
        layers = list(model.regressor)
        self.assertEqual(
            [type(layer) for layer in layers],
            [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear],
        )

    def test_optimizer_learning_rates_and_weight_decay(self):
        for variant, expected_lr in (("A", 1e-4), ("B", 1e-3), ("C", 1e-3)):
            with self.subTest(variant=variant):
                model = diagnostic.build_model(self.config, variant)
                optimizer = diagnostic.build_optimizer(model, self.config, variant)
                self.assertIsInstance(optimizer, torch.optim.AdamW)
                self.assertEqual(optimizer.param_groups[0]["lr"], expected_lr)
                self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-4)

    def test_loss_scheduler_and_patience_contract(self):
        criterion = diagnostic.build_loss(self.config)
        self.assertIsInstance(criterion, nn.MSELoss)
        model = diagnostic.build_model(self.config, "A")
        optimizer = diagnostic.build_optimizer(model, self.config, "A")
        scheduler = diagnostic.build_scheduler(optimizer, self.config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(scheduler.patience, 2)
        self.assertEqual(self.config["early_stopping_patience"], 8)
        self.assertEqual(self.config["max_epochs"], 50)

    def test_batch_seed_and_selection_contract(self):
        self.assertEqual(self.config["seed"], 42)
        self.assertEqual(self.config["batch_size"], 256)
        self.assertEqual(self.config["num_workers"], 0)
        self.assertEqual(self.config["selection_metric"], "validation_rmse")

    def test_predictions_are_not_clipped(self):
        metrics = diagnostic.base.compute_metrics(
            np.array([0.0, 1.0]), np.array([-2.0, 3.0])
        )
        self.assertEqual(metrics["pred_min"], -2.0)
        self.assertEqual(metrics["pred_max"], 3.0)
        self.assertFalse(self.config["prediction_clipping"])


class DiagnosticSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = diagnostic.load_config()

    def test_fold1_fold2_fold3_are_rejected(self):
        for fold in (1, 2, 3):
            with self.subTest(fold=fold), self.assertRaisesRegex(ValueError, "Fold 4"):
                diagnostic.reject_non_fold4(fold)

    def test_fold4_is_the_only_accepted_fold(self):
        diagnostic.reject_non_fold4(4)
        self.assertEqual(self.config["fold"], 4)

    def test_variant_output_directories_are_isolated(self):
        expected_root = (
            ROOT
            / "outputs"
            / "date_grouped_v1"
            / "dinov2_vits14_frozen_regression_diagnostic_v1"
            / "fold_4"
        )
        for variant in diagnostic.ALLOWED_VARIANTS:
            with self.subTest(variant=variant):
                path = diagnostic.variant_output_dir(self.config, variant)
                self.assertEqual(path.parent, expected_root)
                self.assertEqual(path.name, f"variant_{variant}")
                self.assertNotIn("resnet50", str(path).lower())

    def test_runner_all_order_is_a_then_b_then_c(self):
        args = diagnostic.parse_args(["--all"])
        self.assertTrue(args.all)
        self.assertEqual(diagnostic.ALLOWED_VARIANTS, ("A", "B", "C"))

    def test_training_source_has_no_raw_image_or_feature_extraction_access(self):
        source = inspect.getsource(diagnostic)
        forbidden = (
            "Image.open",
            "from PIL",
            "import PIL",
            "torchvision.transforms",
            "torch.hub.load",
        )
        self.assertFalse(any(token in source for token in forbidden))

    def test_model_has_no_irradiance_or_time_input(self):
        parameters = list(inspect.signature(DINOv2DiagnosticRegressionHead.forward).parameters)
        self.assertEqual(parameters, ["self", "features"])

    def test_required_outputs_include_train_and_validation_diagnostics(self):
        self.assertIn("final_metrics.json", diagnostic.REQUIRED_OUTPUT_FILES)
        self.assertIn("predictions.csv", diagnostic.REQUIRED_OUTPUT_FILES)
        self.assertIn("training_predictions.csv", diagnostic.REQUIRED_OUTPUT_FILES)
        self.assertIn("standardization.npz", diagnostic.REQUIRED_OUTPUT_FILES)


if __name__ == "__main__":
    unittest.main()
