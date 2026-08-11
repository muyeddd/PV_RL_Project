import contextlib
import inspect
import io
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from experiments import run_resnet50_image_only_date_grouped_cv as baseline_runner
from experiments import run_resnet50_image_only_photometric_v1_pilot as runner
from experiments import train_resnet50_image_only_date_grouped as baseline
from experiments import train_resnet50_image_only_photometric_v1_date_grouped as training


class PhotometricV1TransformTests(unittest.TestCase):
    def setUp(self):
        self.baseline_train, self.baseline_validation = baseline.build_transforms()
        self.train, self.validation = training.build_transforms()

    def test_validation_transform_is_exactly_baseline(self):
        self.assertEqual(repr(self.validation), repr(self.baseline_validation))
        self.assertEqual(
            training.PREPROCESSING_DESCRIPTION["validation"],
            baseline.PREPROCESSING_DESCRIPTION["validation"],
        )
        self.assertEqual(
            training.PREPROCESSING_DESCRIPTION["normalization_mean"],
            baseline.PREPROCESSING_DESCRIPTION["normalization_mean"],
        )
        self.assertEqual(
            training.PREPROCESSING_DESCRIPTION["normalization_std"],
            baseline.PREPROCESSING_DESCRIPTION["normalization_std"],
        )

    def test_train_transform_order_and_unchanged_geometric_augmentations(self):
        names = [type(item).__name__ for item in self.train.transforms]
        self.assertEqual(
            names,
            [
                "Resize",
                "RandomResizedCrop",
                "RandomHorizontalFlip",
                "RandomRotation",
                "ColorJitter",
                "RandomGamma",
                "ToTensor",
                "Normalize",
            ],
        )
        for baseline_index, experiment_index in ((0, 0), (1, 1), (2, 2), (3, 3), (5, 6), (6, 7)):
            self.assertEqual(
                repr(self.baseline_train.transforms[baseline_index]),
                repr(self.train.transforms[experiment_index]),
            )
        gamma_index = names.index("RandomGamma")
        self.assertLess(gamma_index, names.index("ToTensor"))
        self.assertEqual(gamma_index, names.index("ColorJitter") + 1)

    def test_color_jitter_changes_only_saturation(self):
        baseline_jitter = self.baseline_train.transforms[4]
        experiment_jitter = self.train.transforms[4]
        self.assertIsInstance(experiment_jitter, transforms.ColorJitter)
        self.assertEqual(experiment_jitter.brightness, baseline_jitter.brightness)
        self.assertEqual(experiment_jitter.contrast, baseline_jitter.contrast)
        self.assertEqual(experiment_jitter.hue, baseline_jitter.hue)
        self.assertEqual(experiment_jitter.brightness, (0.92, 1.08))
        self.assertEqual(experiment_jitter.contrast, (0.92, 1.08))
        self.assertEqual(experiment_jitter.saturation, (0.85, 1.15))
        self.assertEqual(experiment_jitter.hue, (-0.02, 0.02))
        self.assertEqual(baseline_jitter.saturation, (0.95, 1.05))

    def test_random_gamma_range_probability_and_seed_control(self):
        gamma = self.train.transforms[5]
        self.assertIsInstance(gamma, training.RandomGamma)
        self.assertEqual(gamma.gamma, (0.85, 1.15))
        self.assertEqual(gamma.probability, 0.5)
        self.assertEqual(gamma.gain, 1.0)

        torch.manual_seed(42)
        samples = [gamma.sample_gamma() for _ in range(1000)]
        self.assertGreaterEqual(min(samples), 0.85)
        self.assertLessEqual(max(samples), 1.15)

        pixels = (np.arange(256 * 256 * 3, dtype=np.uint32) % 256).astype(np.uint8)
        image = Image.fromarray(pixels.reshape(256, 256, 3), mode="RGB")
        torch.manual_seed(42)
        first = self.train(image)
        torch.manual_seed(42)
        second = self.train(image)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

        gamma_source = inspect.getsource(training.RandomGamma)
        self.assertIn("torch.rand", gamma_source)
        self.assertNotIn("random.random", gamma_source)
        self.assertNotIn("np.random", gamma_source)


class PhotometricV1ControlContractTests(unittest.TestCase):
    def test_frozen_baseline_files_have_not_changed(self):
        training.validate_baseline_integrity()
        expected = training.load_training_config()["baseline_source_sha256_lf"]
        self.assertEqual(
            set(expected),
            {
                "models/resnet50_image_only.py",
                "configs/training/resnet50_image_only_date_grouped_v1.json",
                "experiments/train_resnet50_image_only_date_grouped.py",
                "experiments/run_resnet50_image_only_date_grouped_cv.py",
            },
        )

    def test_model_and_all_training_hyperparameters_match_baseline(self):
        experiment = training.load_training_config()
        control = baseline.load_training_config()
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
            self.assertEqual(experiment[field], control[field], field)
        self.assertIs(training.SolarResNet50ImageOnly, baseline.SolarResNet50ImageOnly)
        training.validate_single_factor_contract()

    def test_loss_optimizer_scheduler_and_early_stopping_are_baseline(self):
        source = inspect.getsource(baseline.run_training)
        for fragment in (
            'nn.MSELoss(reduction="mean")',
            "torch.optim.AdamW(",
            "ReduceLROnPlateau(",
            'scheduler.step(validation_metrics["rmse"])',
            'improved = validation_metrics["rmse"] < best_validation_rmse',
            "if epochs_without_improvement >= args.patience:",
        ):
            self.assertIn(fragment, source)
        adapter_source = inspect.getsource(training)
        self.assertIn("return baseline.run_training(args)", adapter_source)
        self.assertNotIn("WeightedRandomSampler", adapter_source)
        self.assertNotIn("SmoothL1Loss", adapter_source)

    def test_runtime_patch_is_narrow_and_restored(self):
        original_builder = baseline.build_transforms
        original_loader = baseline.load_training_config
        original_preprocessing = baseline.PREPROCESSING_DESCRIPTION
        with training.photometric_runtime():
            self.assertIs(baseline.build_transforms, training.build_transforms)
            self.assertIs(baseline.load_training_config, training.load_training_config)
            self.assertIs(
                baseline.PREPROCESSING_DESCRIPTION,
                training.PREPROCESSING_DESCRIPTION,
            )
        self.assertIs(baseline.build_transforms, original_builder)
        self.assertIs(baseline.load_training_config, original_loader)
        self.assertIs(baseline.PREPROCESSING_DESCRIPTION, original_preprocessing)


class PhotometricV1PilotSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audits = {fold: runner.preflight_pilot_fold(fold) for fold in (3, 4)}

    def test_fold3_and_fold4_are_allowed(self):
        self.assertEqual(runner.ALLOWED_FOLDS, (3, 4))
        for fold in (3, 4):
            runner.validate_pilot_fold(fold)
            args = runner.parse_args(["--fold", str(fold)])
            self.assertEqual(args.fold, fold)
            training_args = training.parse_args(["--fold", str(fold)])
            training.validate_experiment_arguments(training_args)
            self.assertEqual(
                training_args.output_dir.resolve(),
                training.expected_output_dir(fold).resolve(),
            )

    def test_fold1_and_fold2_are_rejected(self):
        for fold in (1, 2):
            with self.assertRaises(ValueError):
                runner.validate_pilot_fold(fold)
            with self.assertRaises(ValueError):
                training.expected_output_dir(fold)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.parse_args(["--fold", str(fold)])

    def test_protected_roles_are_inaccessible(self):
        self.assertEqual(
            runner.PROTECTED_ROLES,
            ("cp_calibration", "decision_development", "final_test"),
        )
        for audit in self.audits.values():
            self.assertEqual(
                audit["forbidden_role_counts"],
                {"cp_calibration": 0, "decision_development": 0, "final_test": 0},
            )
            self.assertEqual(set(audit["train_records"]["top_level_role"]), {"model_development"})
            self.assertEqual(
                set(audit["validation_records"]["top_level_role"]),
                {"model_development"},
            )
            self.assertTrue(set(audit["train_dates"]).isdisjoint(audit["validation_dates"]))

    def test_output_root_is_fully_isolated_from_baseline(self):
        runner.validate_output_root_isolation()
        config = training.load_training_config()
        expected = (
            training.PROJECT_ROOT
            / "outputs"
            / "date_grouped_v1"
            / "resnet50_image_only_photometric_v1"
            / "pilot"
        ).resolve()
        self.assertEqual(runner.PILOT_OUTPUT_ROOT.resolve(), expected)
        self.assertEqual(
            config["pilot_output_namespace"],
            "outputs/date_grouped_v1/resnet50_image_only_photometric_v1/pilot",
        )
        self.assertNotEqual(expected, baseline_runner.FORMAL_OUTPUT_ROOT.resolve())
        self.assertNotIn(baseline_runner.FORMAL_OUTPUT_ROOT.resolve(), expected.parents)
        with self.assertRaises(RuntimeError):
            runner.ensure_new_fold_directory(
                baseline_runner.FORMAL_OUTPUT_ROOT / "fold_3_seed_42"
            )

    def test_boundary_metadata_is_explicit(self):
        metadata = training._add_experiment_metadata({})
        self.assertIs(metadata["final_test_accessed"], False)
        self.assertEqual(metadata["forbidden_roles_accessed"], [])
        self.assertEqual(metadata["experiment_version"], training.TRAINING_VERSION)

    def test_runner_has_no_overwrite_resume_or_checkpoint_loading(self):
        source = inspect.getsource(runner)
        parser_source = inspect.getsource(runner.parse_args)
        self.assertNotIn("--overwrite", parser_source)
        self.assertNotIn("--resume", parser_source)
        self.assertNotIn("torch.load", source)


if __name__ == "__main__":
    unittest.main()
