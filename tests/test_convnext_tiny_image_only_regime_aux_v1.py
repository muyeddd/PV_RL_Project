"""Contract tests for the Fold3 ConvNeXt regime-aware auxiliary pilot."""

from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset

from experiments import run_convnext_tiny_image_only_regime_aux_v1_pilot as runner
from experiments import train_convnext_tiny_image_only_date_grouped as baseline
from experiments import train_convnext_tiny_image_only_regime_aux_date_grouped as training
from models import convnext_tiny_image_only as baseline_model
from models.convnext_tiny_image_only_regime_aux import (
    RegimeAuxiliaryHead,
    SolarConvNeXtTinyImageOnlyRegimeAux,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SOURCE_HASHES = {
    "models/convnext_tiny_image_only.py": (
        "08a8646bbbc7332346aac6a2df30793acebcd59484781adfd780df4bbdbb771d"
    ),
    "experiments/train_convnext_tiny_image_only_date_grouped.py": (
        "f849c5d348a04a9ce96354cf9763bd24330edccf6031e335c795162bfc73db1e"
    ),
    "experiments/run_convnext_tiny_image_only_pilot.py": (
        "e8b0fa62d9dc79b1360bf94a9e0b9f627f22b7fb31f6d1be22b2e5554991c322"
    ),
    "configs/training/convnext_tiny_image_only_v1_date_grouped.json": (
        "22471d4b8e9170a070e6526cc75ce414142263af03e9a86320a7b869d4b2321c"
    ),
    "tests/test_convnext_tiny_image_only_pilot.py": (
        "be5f62057881816fc6a3be342be9228b5404a17f67d55b2e3660f50b76947a79"
    ),
}


class RegimeBoundaryTests(unittest.TestCase):
    def test_exact_boundaries(self):
        labels = np.asarray([0.099, 0.1, 0.499, 0.5], dtype=np.float64)
        np.testing.assert_array_equal(
            training.assign_regime_ids(labels), np.asarray([0, 1, 1, 2])
        )

    def test_tensor_boundaries_match_numpy(self):
        labels = torch.tensor([0.0, 0.099, 0.1, 0.499, 0.5, 1.0])
        self.assertEqual(
            training.regime_targets_from_tensor(labels).tolist(), [0, 0, 1, 1, 2, 2]
        )

    def test_each_sample_gets_exactly_one_class(self):
        labels = np.linspace(0.0, 1.0, 10_001)
        ids = training.assign_regime_ids(labels)
        self.assertEqual(ids.shape, labels.shape)
        self.assertTrue(np.isin(ids, [0, 1, 2]).all())


class FoldAndDataProtectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = training.preflight_fold(3)

    def test_fold3_train_regime_counts(self):
        counts = self.audit["train_regime_audit"]["regime_counts"]
        self.assertEqual(counts, [6_549, 8_261, 4_406])
        self.assertEqual(sum(counts), 19_216)

    def test_validation_does_not_participate_in_train_counts(self):
        regime_audit = self.audit["train_regime_audit"]
        self.assertEqual(regime_audit["train_count"], self.audit["train_count"])
        self.assertEqual(self.audit["validation_count"], 6_500)
        self.assertFalse(regime_audit["validation_participated"])
        self.assertEqual(regime_audit["weights_source"], "none")

    def test_model_development_is_the_only_selected_role(self):
        self.assertEqual(self.audit["selected_role"], "model_development")
        self.assertEqual(
            set(self.audit["train_records"]["top_level_role"]), {"model_development"}
        )
        self.assertEqual(
            set(self.audit["validation_records"]["top_level_role"]),
            {"model_development"},
        )

    def test_protected_role_audit_is_false(self):
        self.assertEqual(self.audit["forbidden_roles_accessed"], [])
        self.assertFalse(self.audit["final_test_accessed"])


class LoaderAndLossTests(unittest.TestCase):
    def test_loaders_are_ordinary_shuffle_and_sequential(self):
        train_dataset = TensorDataset(torch.arange(20))
        validation_dataset = TensorDataset(torch.arange(7))
        train_loader, validation_loader = training.build_data_loaders(
            train_dataset,
            validation_dataset,
            batch_size=4,
            num_workers=0,
            device=torch.device("cpu"),
            seed=42,
        )
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertFalse(train_loader.sampler.replacement)
        self.assertIsInstance(validation_loader.sampler, SequentialSampler)
        self.assertEqual(len(validation_loader.dataset), 7)

    def test_no_weighted_sampler_is_imported_or_used(self):
        source = Path(training.__file__).read_text(encoding="utf-8")
        self.assertNotIn("WeightedRandomSampler", source)
        self.assertNotIn("sampler=", source.replace(" ", ""))

    def test_losses_are_plain_unweighted_pytorch_losses(self):
        regression = nn.MSELoss(reduction="mean")
        auxiliary = nn.CrossEntropyLoss(reduction="mean")
        self.assertIsInstance(regression, nn.MSELoss)
        self.assertIsInstance(auxiliary, nn.CrossEntropyLoss)
        self.assertIsNone(auxiliary.weight)

    def test_lambda_and_total_loss_formula(self):
        self.assertEqual(training.LAMBDA_REGIME, 0.01)
        regression_mse = torch.tensor(0.02)
        regime_ce = torch.tensor(1.2)
        total = regression_mse + training.LAMBDA_REGIME * regime_ce
        self.assertAlmostEqual(total.item(), 0.032, places=7)


class ModelAndProtocolTests(unittest.TestCase):
    def test_model_has_baseline_width_shared_layer_and_two_heads(self):
        model = SolarConvNeXtTinyImageOnlyRegimeAux(use_pretrained=False)
        head = model.auxiliary_head
        self.assertIsInstance(head, RegimeAuxiliaryHead)
        self.assertEqual(head.shared[0].in_features, 768)
        self.assertEqual(head.shared[0].out_features, 128)
        self.assertIsInstance(head.shared[1], nn.ReLU)
        self.assertEqual(head.shared[2].p, 0.3)
        self.assertEqual(head.regression[0].in_features, 128)
        self.assertEqual(head.regression[0].out_features, 1)
        self.assertIsInstance(head.regression[1], nn.Sigmoid)
        self.assertEqual(head.regime.in_features, 128)
        self.assertEqual(head.regime.out_features, 3)
        model.eval()
        with torch.inference_mode():
            regression, regime_logits = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(regression.shape), (1, 1))
        self.assertEqual(tuple(regime_logits.shape), (1, 3))
        self.assertGreaterEqual(regression.item(), 0.0)
        self.assertLessEqual(regression.item(), 1.0)

    def test_pretrained_constants_and_verifier_are_reused(self):
        self.assertIs(training.verify_official_checkpoint, baseline_model.verify_official_checkpoint)
        self.assertEqual(
            training.OFFICIAL_CHECKPOINT_FILENAME,
            baseline_model.OFFICIAL_CHECKPOINT_FILENAME,
        )
        self.assertEqual(
            training.OFFICIAL_CHECKPOINT_SHA256,
            baseline_model.OFFICIAL_CHECKPOINT_SHA256,
        )
        self.assertIs(training.OFFICIAL_WEIGHTS, baseline_model.OFFICIAL_WEIGHTS)

    def test_transform_implementation_is_identical_to_baseline(self):
        self.assertIs(training.build_transforms, baseline.build_transforms)
        self.assertEqual(
            training.PREPROCESSING_DESCRIPTION, baseline.PREPROCESSING_DESCRIPTION
        )

    def test_all_frozen_training_fields_match_baseline(self):
        config = training.load_training_config()
        baseline_config = baseline.load_training_config()
        training.validate_single_variable_contract(config, baseline_config)
        for key in (
            "seed",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "optimizer",
            "scheduler",
            "scheduler_factor",
            "scheduler_patience",
            "early_stopping_patience",
            "selection_metric",
            "amp",
            "pretrained",
            "dropout",
            "loss",
        ):
            self.assertEqual(config[key], baseline_config[key], key)

    def test_scheduler_checkpoint_and_early_stopping_use_regression_rmse(self):
        source = inspect.getsource(training.run_training)
        self.assertIn("scheduler.step(validation_regression_rmse)", source)
        self.assertIn("improved = validation_regression_rmse < best_rmse", source)
        self.assertIn(
            'epochs_without_improvement >= config["early_stopping_patience"]', source
        )
        self.assertIn('"selection_metric": "validation_rmse"', source)

    def test_history_records_three_losses_separately(self):
        source = inspect.getsource(training.run_training)
        for field in (
            "train_regression_mse",
            "train_regime_ce",
            "train_total_loss",
            "validation_regression_mse",
            "validation_regime_ce",
            "validation_total_loss",
        ):
            self.assertIn(field, source)


class IsolationAndGitProtectionTests(unittest.TestCase):
    def test_output_namespace_is_independent(self):
        config = training.load_training_config()
        new_namespace = config["pilot_output_namespace"]
        self.assertNotEqual(new_namespace, baseline.load_training_config()["pilot_output_namespace"])
        self.assertNotIn("convnext_tiny_image_only_v1/pilot", new_namespace)
        self.assertNotIn("convnext_tiny_image_only_lbalanced_v1", new_namespace)

    def test_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "pilot"
            existing = root / "fold_3_seed_42"
            existing.mkdir(parents=True)
            with mock.patch.object(training, "OUTPUT_ROOT", root):
                with self.assertRaises(FileExistsError):
                    training._prepare_output_directory(existing, fold=3, seed=42)

    def test_pilot_accepts_only_fold3(self):
        runner.validate_pilot_fold(3)
        for fold in (1, 2, 4):
            with self.assertRaises(ValueError):
                runner.validate_pilot_fold(fold)

    def test_no_formal_training_output_exists(self):
        self.assertFalse(training.expected_output_dir(3).exists())

    def test_baseline_sources_retain_recorded_sha256(self):
        for relative_path, expected in BASELINE_SOURCE_HASHES.items():
            actual = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative_path)


if __name__ == "__main__":
    unittest.main()
