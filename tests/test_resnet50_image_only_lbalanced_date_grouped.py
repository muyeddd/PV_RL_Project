import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import SequentialSampler, TensorDataset, WeightedRandomSampler

from experiments import run_resnet50_image_only_date_grouped_cv as baseline_runner
from experiments import run_resnet50_image_only_lbalanced_date_grouped_cv as runner
from experiments import train_resnet50_image_only_date_grouped as baseline
from experiments import train_resnet50_image_only_lbalanced_date_grouped as training
from models.resnet50_image_only import SolarResNet50ImageOnly


class LBalancedBinDefinitionTests(unittest.TestCase):
    def test_fixed_edges_and_labels(self):
        self.assertEqual(
            training.BIN_EDGES, (-np.inf, 0.1, 0.3, 0.5, 0.7, np.inf)
        )
        self.assertEqual(
            training.BIN_LABELS,
            (
                "(-inf,0.1)",
                "[0.1,0.3)",
                "[0.3,0.5)",
                "[0.5,0.7)",
                "[0.7,+inf)",
            ),
        )

    def test_right_false_boundary_membership(self):
        values = np.asarray(
            [-1.0, 0.099999, 0.1, 0.299999, 0.3, 0.499999, 0.5, 0.699999, 0.7, 9.0]
        )
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        self.assertEqual(training.assign_l_bin_ids(values).tolist(), expected)

    def test_non_finite_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            training.assign_l_bin_ids([0.1, np.nan])


class LBalancedFoldAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audits = {
            fold: training.preflight_fold(
                training.DEFAULT_MANIFEST, training.DEFAULT_IMAGE_ROOT, fold
            )
            for fold in range(1, 5)
        }

    def test_four_fold_train_bin_counts_match_frozen_audit(self):
        for fold, expected in training.FROZEN_TRAIN_BIN_COUNTS.items():
            provenance = self.audits[fold]["sampling_provenance"]
            actual = tuple(provenance["train_bin_counts"][label] for label in training.BIN_LABELS)
            self.assertEqual(actual, expected, fold)
            self.assertEqual(sum(actual), training.EXPECTED_FOLD_COUNTS[fold][0])

    def test_fold_dates_counts_and_roles_remain_frozen(self):
        expected_dates = {
            1: (
                ["2017-06-14", "2017-06-16", "2017-06-20", "2017-06-21", "2017-06-26", "2017-06-29"],
                ["2017-06-13", "2017-06-28"],
            ),
            2: (
                ["2017-06-13", "2017-06-16", "2017-06-20", "2017-06-21", "2017-06-26", "2017-06-28"],
                ["2017-06-14", "2017-06-29"],
            ),
            3: (
                ["2017-06-13", "2017-06-14", "2017-06-20", "2017-06-21", "2017-06-28", "2017-06-29"],
                ["2017-06-16", "2017-06-26"],
            ),
            4: (
                ["2017-06-13", "2017-06-14", "2017-06-16", "2017-06-26", "2017-06-28", "2017-06-29"],
                ["2017-06-20", "2017-06-21"],
            ),
        }
        for fold, audit in self.audits.items():
            expected_train, expected_validation = training.EXPECTED_FOLD_COUNTS[fold]
            self.assertEqual(len(audit["train_records"]), expected_train)
            self.assertEqual(len(audit["validation_records"]), expected_validation)
            self.assertTrue(set(audit["train_dates"]).isdisjoint(audit["validation_dates"]))
            self.assertEqual(audit["train_dates"], expected_dates[fold][0])
            self.assertEqual(audit["validation_dates"], expected_dates[fold][1])
            self.assertEqual(set(audit["train_records"]["top_level_role"]), {"model_development"})
            self.assertEqual(set(audit["validation_records"]["top_level_role"]), {"model_development"})
            self.assertEqual(
                audit["forbidden_role_counts"],
                {"cp_calibration": 0, "decision_development": 0, "final_test": 0},
            )

    def test_sampling_provenance_contains_split_identity(self):
        for audit in self.audits.values():
            provenance = audit["sampling_provenance"]
            self.assertEqual(provenance["manifest_sha256"], audit["manifest_sha256"])
            self.assertEqual(provenance["dataset_fingerprint"], audit["fingerprint"])
            self.assertEqual(provenance["train_dates"], audit["train_dates"])

    def test_theoretical_probabilities_match_frozen_audit(self):
        expected_percent = {
            1: (30.615, 23.639, 16.229, 17.751, 11.766),
            2: (30.604, 20.033, 16.378, 20.942, 12.044),
            3: (28.415, 27.916, 15.465, 22.630, 5.574),
            4: (29.367, 24.087, 10.760, 24.399, 11.386),
        }
        for fold, expected in expected_percent.items():
            probabilities = self.audits[fold]["sampling_provenance"][
                "expected_sampling_probability_per_bin"
            ]
            actual = tuple(probabilities[label] * 100 for label in training.BIN_LABELS)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0006)

    def test_probability_formula_is_sqrt_count_mass(self):
        for fold, counts in training.FROZEN_TRAIN_BIN_COUNTS.items():
            counts_array = np.asarray(counts, dtype=np.float64)
            expected = np.sqrt(counts_array) / np.sqrt(counts_array).sum()
            actual = training.theoretical_sampling_probabilities(counts_array)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_weight_formula_is_exactly_count_to_minus_one_half(self):
        for fold, audit in self.audits.items():
            plan = audit["sampling_plan"]
            counts = np.asarray(training.FROZEN_TRAIN_BIN_COUNTS[fold])
            expected = torch.as_tensor(counts, dtype=torch.double)[
                torch.as_tensor(plan["bin_ids"], dtype=torch.long)
            ].double().pow(-0.5)
            torch.testing.assert_close(plan["weights"], expected, rtol=0.0, atol=0.0)
            self.assertEqual(plan["weights"].dtype, torch.double)

    def test_alpha_is_strictly_one_half_and_weights_are_not_normalized(self):
        config = training.load_training_config()
        self.assertEqual(config["sampling"]["alpha"], 0.5)
        self.assertEqual(training.ALPHA, 0.5)
        self.assertFalse(config["sampling"]["normalize_weights"])
        for audit in self.audits.values():
            provenance = audit["sampling_provenance"]
            self.assertEqual(provenance["weight_formula"], "n_bin^-0.5")
            self.assertFalse(provenance["weights_normalized"])

    def test_sampling_design_accepts_training_records_only(self):
        parameters = tuple(inspect.signature(training.build_sampling_provenance).parameters)
        self.assertEqual(parameters, ("train_records", "fold"))
        function_source = inspect.getsource(training.build_sampling_provenance)
        self.assertNotIn("validation", function_source.lower())

    def test_non_model_development_records_fail_before_label_parsing(self):
        protected = pd.DataFrame(
            {
                "filename": ["must_not_parse.jpg"],
                "date": ["2099-01-01"],
                "top_level_role": ["final_test"],
            }
        )
        with mock.patch.object(training, "parse_filename") as parser:
            with self.assertRaises(ValueError):
                training.labels_from_training_records(protected)
            parser.assert_not_called()

    def test_sanity_simulation_passes_for_all_folds(self):
        for audit in self.audits.values():
            provenance = audit["sampling_provenance"]
            simulation = provenance["sanity_simulation"]
            self.assertTrue(simulation["passed"])
            self.assertEqual(simulation["seed"], 42)
            self.assertEqual(simulation["draws"], 100000)
            self.assertLess(
                simulation["maximum_absolute_probability_error"], 0.005
            )
            self.assertLess(
                provenance["sampled_total_variation_from_uniform"],
                provenance["baseline_total_variation_from_uniform"],
            )
            probabilities = list(
                provenance["expected_sampling_probability_per_bin"].values()
            )
            self.assertFalse(np.allclose(probabilities, np.full(5, 0.2), atol=1e-12))

    def test_fold3_high_l_risk_is_explicit(self):
        audit = self.audits[3]["sampling_provenance"]["fold3_high_l_audit"]
        self.assertEqual(audit["original_count"], 252)
        self.assertEqual(audit["train_count"], 19216)
        self.assertAlmostEqual(audit["sqrt_sampling_probability"], 0.05573874, places=7)
        self.assertAlmostEqual(audit["expected_draws_per_epoch"], 1071.08, places=1)
        self.assertIn("temporally", audit["risk"])


class LBalancedLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = training.preflight_fold(
            training.DEFAULT_MANIFEST, training.DEFAULT_IMAGE_ROOT, 3
        )
        cls.factory = training.LBalancedDataLoaderFactory(
            3,
            cls.audit["train_records"],
            sampling_plan=cls.audit["sampling_plan"],
        )
        train_data = TensorDataset(torch.zeros(len(cls.audit["train_records"]), 1))
        train_data.files = cls.audit["train_records"]["filename"].tolist()
        cls.train_loader = cls.factory(
            train_data,
            batch_size=32,
            shuffle=True,
            generator=torch.Generator().manual_seed(999),
            num_workers=0,
        )
        validation_data = TensorDataset(torch.zeros(17, 1))
        cls.validation_loader = cls.factory(
            validation_data,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )

    def test_training_uses_weighted_random_sampler(self):
        sampler = self.train_loader.sampler
        self.assertIsInstance(sampler, WeightedRandomSampler)
        self.assertTrue(sampler.replacement)
        self.assertEqual(sampler.num_samples, len(self.audit["train_records"]))
        self.assertIs(sampler, self.factory.train_sampler)

    def test_training_sampler_and_worker_generators_are_independent_seed_42(self):
        self.assertIsNot(
            self.factory.sampler_generator, self.factory.worker_generator
        )
        self.assertEqual(self.factory.sampler_generator.initial_seed(), 42)
        self.assertEqual(self.factory.worker_generator.initial_seed(), 42)
        self.assertIs(self.train_loader.generator, self.factory.worker_generator)
        self.assertIs(
            self.train_loader.sampler.generator, self.factory.sampler_generator
        )

    def test_training_shuffle_is_disabled_by_custom_sampler(self):
        self.assertIsInstance(self.train_loader.sampler, WeightedRandomSampler)
        self.assertNotIsInstance(self.train_loader.sampler, SequentialSampler)

    def test_validation_is_sequential_and_unweighted(self):
        self.assertIsInstance(self.validation_loader.sampler, SequentialSampler)
        self.assertNotIsInstance(self.validation_loader.sampler, WeightedRandomSampler)
        self.assertEqual(len(self.validation_loader.dataset), 17)

    def test_factory_constructs_exactly_train_and_validation_loaders_once(self):
        self.assertEqual(self.factory.calls, 2)
        with self.assertRaises(RuntimeError):
            self.factory(TensorDataset(torch.zeros(1, 1)), batch_size=1, shuffle=False)


class LBalancedSingleFactorTests(unittest.TestCase):
    def test_model_is_exactly_the_baseline_model(self):
        self.assertIs(training.SolarResNet50ImageOnly, SolarResNet50ImageOnly)
        self.assertIs(training.SolarResNet50ImageOnly, baseline.SolarResNet50ImageOnly)

    def test_transforms_and_normalization_are_exactly_baseline(self):
        self.assertEqual(training.PREPROCESSING_DESCRIPTION, baseline.PREPROCESSING_DESCRIPTION)
        self.assertEqual(repr(baseline.build_transforms()), repr(baseline.build_transforms()))
        adapter_source = inspect.getsource(training)
        self.assertNotIn("def build_transforms", adapter_source)

    def test_loss_optimizer_scheduler_and_early_stop_remain_baseline(self):
        source = inspect.getsource(baseline.run_training)
        required = (
            'nn.MSELoss(reduction="mean")',
            "torch.optim.AdamW(",
            "ReduceLROnPlateau(",
            'scheduler.step(validation_metrics["rmse"])',
            'improved = validation_metrics["rmse"] < best_validation_rmse',
            "if epochs_without_improvement >= args.patience:",
        )
        for fragment in required:
            self.assertIn(fragment, source)
        self.assertIs(training.baseline.run_epoch, baseline.run_epoch)
        self.assertIs(training.baseline.MetricAccumulator, baseline.MetricAccumulator)

    def test_no_weighted_or_huber_loss_is_added(self):
        adapter_source = inspect.getsource(training)
        self.assertNotIn("HuberLoss", adapter_source)
        self.assertNotIn("SmoothL1Loss", adapter_source)
        self.assertNotIn("weighted_loss", adapter_source.lower())
        self.assertNotIn("criterion =", adapter_source)

    def test_no_other_sampler_or_new_augmentation_is_implemented(self):
        adapter_source = inspect.getsource(training)
        self.assertNotIn("BatchSampler", adapter_source)
        self.assertNotIn("RandomResizedCrop", adapter_source)
        self.assertNotIn("ColorJitter", adapter_source)
        self.assertNotIn("gamma", adapter_source.lower())
        self.assertNotIn("date_weight", adapter_source.lower())
        self.assertNotIn("capped", adapter_source.lower())

    def test_frozen_training_fields_match_baseline(self):
        current = training.load_training_config()
        reference = baseline.load_training_config()
        fields = (
            "schema_version",
            "split_version",
            "model_name",
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
        for field in fields:
            self.assertEqual(current[field], reference[field], field)
        training.validate_single_factor_contract()

    def test_temporal_risk_and_training_metric_notes_are_frozen(self):
        self.assertIn("highly temporally correlated", training.TEMPORAL_RISK_NOTE)
        self.assertIn("only 252", training.TEMPORAL_RISK_NOTE)
        self.assertIn("does not turn them into independent scenes", training.TEMPORAL_RISK_NOTE)
        self.assertIn("must not simply be increased", training.TEMPORAL_RISK_NOTE)
        self.assertIn("sampled draws", training.TRAINING_METRICS_NOTE)
        self.assertIn("unweighted validation metrics", training.TRAINING_METRICS_NOTE)


class LBalancedRunnerSafetyTests(unittest.TestCase):
    def test_output_root_is_isolated_from_baseline(self):
        runner.validate_output_root_isolation()
        self.assertNotEqual(runner.FORMAL_OUTPUT_ROOT.resolve(), baseline_runner.FORMAL_OUTPUT_ROOT.resolve())
        self.assertIn("resnet50_image_only_lbalanced", runner.FORMAL_OUTPUT_ROOT.parts)

    def test_output_guard_rejects_baseline_root(self):
        with mock.patch.object(runner, "FORMAL_OUTPUT_ROOT", baseline_runner.FORMAL_OUTPUT_ROOT):
            with self.assertRaises(RuntimeError):
                runner.validate_output_root_isolation()

    def test_formal_parsers_expose_no_overwrite_resume_or_pilot_switch(self):
        runner_source = inspect.getsource(runner.parse_args)
        training_source = inspect.getsource(training.parse_args)
        for source in (runner_source, training_source):
            self.assertNotIn('add_argument("--overwrite"', source)
            self.assertNotIn('add_argument("--resume"', source)
            self.assertNotIn('add_argument("--pilot-run"', source)

    def test_formal_sources_cannot_load_old_checkpoints(self):
        self.assertNotIn("torch.load", inspect.getsource(training))
        self.assertNotIn("torch.load", inspect.getsource(runner))

    def test_existing_fold_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fold_1_seed_42"
            path.mkdir()
            with self.assertRaises(FileExistsError):
                runner.ensure_new_fold_directory(path)

    def test_runner_constants_match_baseline_except_output_and_training_version(self):
        for name in (
            "FOLDS",
            "SEED",
            "EPOCHS",
            "BATCH_SIZE",
            "NUM_WORKERS",
            "LEARNING_RATE",
            "WEIGHT_DECAY",
            "PATIENCE",
            "AMP",
            "PRETRAINED",
            "EXPECTED_MANIFEST_SHA256",
        ):
            self.assertEqual(getattr(runner, name), getattr(baseline_runner, name), name)
        runner.validate_frozen_config()


if __name__ == "__main__":
    unittest.main()
