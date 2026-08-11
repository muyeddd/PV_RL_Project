from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import SequentialSampler, TensorDataset, WeightedRandomSampler

from experiments import run_convnext_tiny_image_only_lbalanced_v1_pilot as runner
from experiments import train_convnext_tiny_image_only_date_grouped as baseline
from experiments import train_convnext_tiny_image_only_lbalanced_date_grouped as training
from models.convnext_tiny_image_only import SolarConvNeXtTinyImageOnly


class ConvNeXtLBalancedFoldAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audits = {fold: training.preflight_fold(fold) for fold in (3, 4)}

    def test_fold3_train_bin_counts_are_frozen(self):
        provenance = self.audits[3]["sampling_provenance"]
        actual = tuple(
            provenance["train_bin_counts"][label] for label in training.BIN_LABELS
        )
        self.assertEqual(actual, (6_549, 6_321, 1_940, 4_154, 252))
        self.assertEqual(sum(actual), 19_216)

    def test_fold4_train_bin_counts_are_frozen(self):
        provenance = self.audits[4]["sampling_provenance"]
        actual = tuple(
            provenance["train_bin_counts"][label] for label in training.BIN_LABELS
        )
        self.assertEqual(actual, (7_404, 4_981, 994, 5_111, 1_113))
        self.assertEqual(sum(actual), 19_603)

    def test_fixed_bin_edges_and_right_side_boundary_membership(self):
        self.assertEqual(
            training.BIN_EDGES, (-np.inf, 0.1, 0.3, 0.5, 0.7, np.inf)
        )
        values = np.asarray(
            [-1.0, 0.099999, 0.1, 0.299999, 0.3, 0.499999, 0.5, 0.699999, 0.7, 9.0]
        )
        self.assertEqual(
            training.assign_l_bin_ids(values).tolist(),
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        )
        self.assertIn('side="right"', inspect.getsource(training.assign_l_bin_ids))

    def test_each_sample_weight_is_exact_count_to_minus_one_half(self):
        for fold, audit in self.audits.items():
            with self.subTest(fold=fold):
                plan = audit["sampling_plan"]
                counts = torch.as_tensor(
                    training.FROZEN_TRAIN_BIN_COUNTS[fold], dtype=torch.double
                )
                expected = counts[
                    torch.as_tensor(plan["bin_ids"], dtype=torch.long)
                ].pow(-0.5)
                torch.testing.assert_close(
                    plan["weights"], expected, rtol=0.0, atol=0.0
                )
                self.assertEqual(plan["weights"].dtype, torch.double)

    def test_fold3_high_l_sampler_probability_is_about_5_574_percent(self):
        probability = self.audits[3]["sampling_provenance"][
            "expected_sampling_probability_per_bin"
        ]["[0.7,+inf)"]
        self.assertAlmostEqual(probability * 100.0, 5.573874, places=5)
        self.assertAlmostEqual(252 / 19_216, 0.013114, places=5)

    def test_sampler_plan_accepts_only_current_fold_training_records(self):
        self.assertEqual(
            tuple(inspect.signature(training.build_sampling_plan).parameters),
            ("train_records", "fold"),
        )
        validation_record = self.audits[3]["validation_records"].iloc[[0]].copy()
        with mock.patch.object(baseline, "parse_loss_label") as parse_label:
            with self.assertRaises(ValueError):
                training.labels_from_training_records(validation_record, 3)
            parse_label.assert_not_called()

    def test_only_model_development_is_selected(self):
        for fold, audit in self.audits.items():
            with self.subTest(fold=fold):
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
                provenance = audit["sampling_provenance"]
                self.assertFalse(provenance["protected_roles_accessed"])
                self.assertFalse(provenance["cp_calibration_accessed"])
                self.assertFalse(provenance["decision_development_accessed"])
                self.assertFalse(provenance["final_test_accessed"])


class ConvNeXtLBalancedLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = training.preflight_fold(3)
        cls.factory = training.ConvNeXtLBalancedDataLoaderFactory(
            cls.audit["train_records"],
            cls.audit["validation_records"],
            cls.audit["sampling_plan"],
        )
        train_data = TensorDataset(torch.zeros(cls.audit["train_count"], 1))
        train_data.records = cls.audit["train_records"].copy()
        cls.train_loader = cls.factory(
            train_data,
            batch_size=32,
            shuffle=True,
            generator=torch.Generator().manual_seed(999),
            num_workers=0,
        )
        validation_data = TensorDataset(
            torch.zeros(cls.audit["validation_count"], 1)
        )
        validation_data.records = cls.audit["validation_records"].copy()
        cls.validation_loader = cls.factory(
            validation_data,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )

    def test_training_loader_uses_weighted_replacement_sampling(self):
        sampler = self.train_loader.sampler
        self.assertIsInstance(sampler, WeightedRandomSampler)
        self.assertTrue(sampler.replacement)
        self.assertEqual(sampler.num_samples, self.audit["train_count"])
        self.assertIs(sampler, self.factory.train_sampler)

    def test_sampler_and_worker_generators_are_independent_seed_42(self):
        self.assertIsNot(
            self.factory.sampler_generator, self.factory.worker_generator
        )
        self.assertEqual(self.factory.sampler_generator.initial_seed(), 42)
        self.assertEqual(self.factory.worker_generator.initial_seed(), 42)
        self.assertIs(self.train_loader.generator, self.factory.worker_generator)
        self.assertIs(
            self.train_loader.sampler.generator, self.factory.sampler_generator
        )

    def test_validation_is_sequential_unweighted_and_full_length(self):
        self.assertIsInstance(self.validation_loader.sampler, SequentialSampler)
        self.assertNotIsInstance(
            self.validation_loader.sampler, WeightedRandomSampler
        )
        self.assertEqual(
            len(self.validation_loader.dataset), self.audit["validation_count"]
        )
        self.assertEqual(len(self.validation_loader.dataset), 6_500)

    def test_training_record_order_mismatch_fails_immediately(self):
        factory = training.ConvNeXtLBalancedDataLoaderFactory(
            self.audit["train_records"],
            self.audit["validation_records"],
            self.audit["sampling_plan"],
        )
        dataset = TensorDataset(torch.zeros(self.audit["train_count"], 1))
        dataset.records = self.audit["train_records"].iloc[::-1].reset_index(drop=True)
        with self.assertRaises(RuntimeError):
            factory(dataset, batch_size=32, shuffle=True, num_workers=0)

    def test_factory_allows_exactly_train_and_validation_loader_calls(self):
        self.assertEqual(self.factory.calls, 2)
        extra = TensorDataset(torch.zeros(1, 1))
        extra.records = self.audit["validation_records"].iloc[[0]].copy()
        with self.assertRaises(RuntimeError):
            self.factory(extra, batch_size=1, shuffle=False, num_workers=0)


class ConvNeXtLBalancedProtocolTests(unittest.TestCase):
    def test_variant_is_an_exact_single_factor_protocol(self):
        training.validate_single_factor_contract()
        current = training.load_training_config()
        reference = baseline.load_training_config()
        frozen_fields = (
            "split_version",
            "manifest_sha256",
            "model_name",
            "weights_enum",
            "weights_filename",
            "weights_sha256",
            "seed",
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "num_workers",
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
            "pretrained",
            "dropout",
            "transform_source",
        )
        for field in frozen_fields:
            self.assertEqual(current[field], reference[field], field)

    def test_model_class_and_training_implementation_remain_baseline(self):
        self.assertIs(baseline.SolarConvNeXtTinyImageOnly, SolarConvNeXtTinyImageOnly)
        adapter_source = inspect.getsource(training)
        self.assertNotIn("class SolarConvNeXtTinyImageOnly", adapter_source)
        run_source = inspect.getsource(baseline.run_training)
        self.assertIn("SolarConvNeXtTinyImageOnly(dropout=0.3, use_pretrained=True)", run_source)

    def test_loss_is_plain_mse_and_no_alternative_method_is_added(self):
        baseline_source = inspect.getsource(baseline.run_training)
        adapter_source = inspect.getsource(training)
        self.assertIn('nn.MSELoss(reduction="mean")', baseline_source)
        self.assertNotIn("HuberLoss", adapter_source)
        self.assertNotIn("SmoothL1Loss", adapter_source)
        self.assertNotIn("weighted_loss", adapter_source.lower())
        for forbidden in (
            "mixup",
            "cutmix",
            "date_weight",
            "temporal_sampler",
            "domain_adaptation",
            "exponentialmovingaverage",
            "differential_lr",
        ):
            self.assertNotIn(forbidden, adapter_source.lower())

    def test_optimizer_scheduler_amp_and_selection_remain_baseline(self):
        source = inspect.getsource(baseline.run_training)
        required = (
            "torch.optim.AdamW(",
            "ReduceLROnPlateau(",
            'scheduler.step(validation_metrics["rmse"])',
            'improved = validation_metrics["rmse"] < best_rmse',
            'config["early_stopping_patience"]',
            'config["amp"]',
        )
        for fragment in required:
            self.assertIn(fragment, source)

    def test_augmentation_is_exactly_the_baseline_augmentation(self):
        self.assertIs(training.baseline.build_transforms, baseline.build_transforms)
        self.assertEqual(
            repr(training.baseline.build_transforms()),
            repr(baseline.build_transforms()),
        )


class ConvNeXtLBalancedRunnerAndOutputTests(unittest.TestCase):
    def test_new_and_baseline_output_namespaces_do_not_overlap(self):
        training.validate_output_namespace()
        new_root = training.OUTPUT_ROOT.resolve()
        baseline_root = baseline.OUTPUT_ROOT.resolve()
        self.assertNotEqual(new_root, baseline_root)
        self.assertNotIn(baseline_root, new_root.parents)
        self.assertEqual(
            training.expected_output_dir(3).resolve(),
            (
                training.PROJECT_ROOT
                / "outputs"
                / "date_grouped_v1"
                / "convnext_tiny_image_only_lbalanced_v1"
                / "pilot"
                / "fold_3_seed_42"
            ).resolve(),
        )

    def test_existing_output_directory_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            isolated_root = Path(temporary_directory) / "pilot"
            with mock.patch.object(training, "OUTPUT_ROOT", isolated_root):
                output_dir = training.expected_output_dir(3)
                output_dir.mkdir(parents=True)
                marker = output_dir / "existing.marker"
                marker.write_text("preserve", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    training.ensure_new_output_target(output_dir, 3, 42)
                self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_fold1_fold2_rejected_and_fold3_fold4_allowed(self):
        for fold in (1, 2):
            with self.subTest(fold=fold):
                with self.assertRaises(ValueError):
                    training.validate_pilot_fold(fold)
                with self.assertRaises(ValueError):
                    runner.validate_pilot_fold(fold)
                with self.assertRaises(ValueError):
                    runner.parse_args(["--fold", str(fold)])
        for fold in (3, 4):
            with self.subTest(fold=fold):
                training.validate_pilot_fold(fold)
                runner.validate_pilot_fold(fold)
                args = runner.parse_args(["--fold", str(fold)])
                self.assertEqual(args.fold, fold)
                self.assertEqual(
                    args.output_dir.resolve(),
                    training.expected_output_dir(fold).resolve(),
                )

    def test_runner_routes_allowed_folds_without_training_in_test(self):
        for fold in (3, 4):
            with self.subTest(fold=fold):
                with mock.patch.object(training, "run_training") as run_training:
                    runner.main(["--fold", str(fold)])
                run_training.assert_called_once()


if __name__ == "__main__":
    unittest.main()
