"""Safety and protocol tests for frozen Image-only OOF reconstruction."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments import build_resnet50_image_only_oof_diagnostics as diagnostics
from experiments import train_resnet50_image_only_date_grouped as training


class TestDefinitionsAndBins(unittest.TestCase):
    def test_error_columns_use_frozen_definitions(self):
        true = np.asarray([0.2, 0.8], dtype=np.float64)
        pred = np.asarray([0.1, 0.95], dtype=np.float64)
        error = pred - true
        self.assertTrue(np.allclose(error, [-0.1, 0.15]))
        self.assertTrue(np.allclose(np.abs(error), [0.1, 0.15]))
        self.assertTrue(np.allclose(error**2, [0.01, 0.0225]))

    def test_l_bins_are_fixed_and_data_independent(self):
        self.assertEqual(
            diagnostics.L_BIN_EDGES,
            (-np.inf, 0.1, 0.3, 0.5, 0.7, np.inf),
        )
        first = diagnostics.assign_l_bins(np.asarray([-10.0, 0.1, 0.7, 10.0]))
        second = diagnostics.assign_l_bins(np.asarray([-1e9, 0.1, 0.7, 1e9]))
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(tuple(first.categories), diagnostics.L_BIN_LABELS)

    def test_every_finite_value_enters_a_bin(self):
        values = np.asarray(
            [-1e12, -1.0, 0.0, 0.09999, 0.1, 0.3, 0.5, 0.7, 1.0, 1e12]
        )
        assigned = diagnostics.assign_l_bins(values)
        self.assertFalse(pd.isna(assigned).any())

    def test_prediction_std_ratio_has_zero_variance_guard(self):
        ratio = diagnostics.safe_std_ratio(
            pd.Series([0.1, 0.2, 0.3]), pd.Series([0.5, 0.5, 0.5])
        )
        self.assertTrue(np.isnan(ratio))


class TestOOFProtocolIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audits = {
            fold: diagnostics.collect_fold_records(fold)
            for fold in diagnostics.FOLD_SPECS
        }
        frames = []
        for fold, audit in cls.audits.items():
            records = audit["validation_records"].reset_index(drop=True)
            count = len(records)
            true = np.linspace(0.0, 1.0, count, dtype=np.float64)
            pred = true.copy()
            frames.append(
                pd.DataFrame(
                    {
                        "filename": records["filename"],
                        "timestamp": records["timestamp"],
                        "date": records["date"],
                        "fold": fold,
                        "L_true": true,
                        "L_pred": pred,
                        "error": pred - true,
                        "abs_error": np.abs(pred - true),
                        "squared_error": (pred - true) ** 2,
                        "irradiance": np.zeros(count),
                    }
                )
            )
        cls.oof = pd.concat(frames, ignore_index=True)[list(diagnostics.OOF_COLUMNS)]

    def test_oof_total_unique_fold_counts_and_dates(self):
        diagnostics.validate_oof_integrity(self.oof)
        self.assertEqual(len(self.oof), 25716)
        self.assertEqual(self.oof["filename"].nunique(), 25716)
        self.assertEqual(
            tuple(sorted(self.oof["date"].unique())), diagnostics.DEVELOPMENT_DATES
        )
        self.assertEqual(
            self.oof.groupby("fold").size().to_dict(),
            {1: 5911, 2: 7192, 3: 6500, 4: 6113},
        )

    def test_duplicate_oof_filename_is_rejected(self):
        duplicated = self.oof.copy()
        duplicated.loc[1, "filename"] = duplicated.loc[0, "filename"]
        with self.assertRaisesRegex(ValueError, "filenames must be unique"):
            diagnostics.validate_oof_integrity(duplicated)

    def test_protected_roles_are_absent_from_all_fold_records(self):
        for audit in self.audits.values():
            self.assertEqual(
                audit["forbidden_role_counts"],
                {role: 0 for role in diagnostics.FORBIDDEN_ROLES},
            )
            roles = set(audit["train_records"]["top_level_role"]) | set(
                audit["validation_records"]["top_level_role"]
            )
            self.assertEqual(roles, {"model_development"})

    def test_protected_record_cannot_construct_inference_dataset(self):
        protected = pd.DataFrame(
            [
                {
                    "filename": "never_open.jpg",
                    "timestamp": "synthetic",
                    "date": diagnostics.DEVELOPMENT_DATES[0],
                    "top_level_role": "final_test",
                }
            ]
        )
        with mock.patch.object(
            diagnostics.SolarDataset,
            "__init__",
            side_effect=AssertionError("SolarDataset must not be reached"),
        ) as constructor:
            with self.assertRaisesRegex(ValueError, "Only model_development"):
                diagnostics.DiagnosticValidationDataset(protected, transform=None)
            constructor.assert_not_called()


class TestCheckpointSafety(unittest.TestCase):
    def make_checkpoint(self, fold=1):
        return {
            "model_name": diagnostics.MODEL_NAME,
            "seed": diagnostics.SEED,
            "fold": fold,
            "manifest_sha256": diagnostics.EXPECTED_MANIFEST_SHA256,
            "checkpoint_kind": "best",
            "pilot_run": False,
            "NOT_FOR_RESEARCH_METRICS": False,
            "validation_dates": list(
                diagnostics.FOLD_SPECS[fold]["validation_dates"]
            ),
            "model_state_dict": {},
        }

    def test_checkpoint_fold_dates_and_hash_are_validated(self):
        checkpoint = self.make_checkpoint(1)
        path = Path("best_model.pth")
        diagnostics.validate_checkpoint_metadata(
            checkpoint, path, 1, require_formal_path=False
        )
        for field, bad_value in (
            ("fold", 2),
            ("manifest_sha256", "bad-hash"),
            ("validation_dates", ["2017-01-01"]),
        ):
            corrupted = dict(checkpoint)
            corrupted[field] = bad_value
            with self.assertRaises(ValueError):
                diagnostics.validate_checkpoint_metadata(
                    corrupted, path, 1, require_formal_path=False
                )

    def test_best_model_filename_is_mandatory(self):
        with self.assertRaisesRegex(ValueError, "Only best_model"):
            diagnostics.validate_checkpoint_metadata(
                self.make_checkpoint(),
                Path("anything_else.pth"),
                1,
                require_formal_path=False,
            )

    def test_final_model_is_rejected(self):
        with self.assertRaises(ValueError):
            diagnostics.validate_checkpoint_metadata(
                self.make_checkpoint(),
                Path("final_model.pth"),
                1,
                require_formal_path=False,
            )


class TestInferenceAndTransforms(unittest.TestCase):
    def test_model_forward_receives_images_only(self):
        class ImageOnlySpy(nn.Module):
            def __init__(self):
                super().__init__()
                self.received = None

            def forward(self, images):
                self.received = images
                return images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

        model = ImageOnlySpy()
        images = torch.rand(2, 3, 16, 16)
        irradiance = torch.tensor([0.0, 999.0])
        predictions = diagnostics.predict_images(model, images)
        self.assertIs(model.received, images)
        self.assertEqual(tuple(predictions.shape), (2, 1))
        self.assertNotIn("irradiance", inspect.signature(diagnostics.predict_images).parameters)
        del irradiance

    def test_inference_source_never_forwards_irradiance(self):
        source = inspect.getsource(diagnostics.run_fold_inference)
        self.assertIn("predict_images(model, images)", source)
        self.assertNotIn("model(images,", source)
        self.assertNotIn("predict_images(model, images,", source)

    def test_validation_transform_matches_frozen_training(self):
        actual = diagnostics.build_validation_transform()
        _, expected = training.build_transforms()
        self.assertEqual(repr(actual), repr(expected))
        self.assertEqual(
            [type(item).__name__ for item in actual.transforms],
            ["Resize", "ToTensor", "Normalize"],
        )
        resize, _, normalize = actual.transforms
        self.assertEqual(resize.size, (224, 224))
        self.assertEqual(list(normalize.mean), [0.485, 0.456, 0.406])
        self.assertEqual(list(normalize.std), [0.229, 0.224, 0.225])


class TestMetricsAndSupport(unittest.TestCase):
    @staticmethod
    def small_oof():
        frames = []
        for fold in diagnostics.FOLD_SPECS:
            true = np.asarray([0.0, 1.0], dtype=np.float32)
            pred = np.asarray([0.1, 0.9], dtype=np.float32)
            frames.append(pd.DataFrame({"fold": fold, "L_true": true, "L_pred": pred}))
        return pd.concat(frames, ignore_index=True)

    def test_reconstructed_metric_tolerance_gate(self):
        oof = self.small_oof()
        actual = diagnostics.compute_regression_metrics(oof[oof["fold"].eq(1)])
        expected = {
            fold: {name: actual[name] for name in ("mae", "rmse", "r2")}
            for fold in diagnostics.FOLD_SPECS
        }
        with mock.patch.object(diagnostics, "EXPECTED_FORMAL_METRICS", expected):
            comparisons = diagnostics.validate_reconstructed_metrics(oof)
            self.assertTrue(all(item["passed"] for item in comparisons.values()))

            mismatched = {fold: dict(values) for fold, values in expected.items()}
            mismatched[3]["rmse"] += 0.01
            with mock.patch.object(
                diagnostics, "EXPECTED_FORMAL_METRICS", mismatched
            ):
                with self.assertRaisesRegex(RuntimeError, "diagnostics generation stopped"):
                    diagnostics.validate_reconstructed_metrics(oof)

    def test_train_support_is_derived_only_from_train_filenames(self):
        filenames = [
            "solar_Tue_Jun_13_10__0__0_2017_L_0.1_I_0.2.jpg",
            "solar_Tue_Jun_13_10__0__1_2017_L_0.4_I_0.2.jpg",
            "solar_Tue_Jun_13_10__0__2_2017_L_0.9_I_0.2.jpg",
        ]
        audits = {1: {"train_records": pd.DataFrame({"filename": filenames})}}
        rows = []
        for date in diagnostics.FOLD_SPECS[1]["validation_dates"]:
            rows.extend(
                [
                    {"date": date, "L_true": 0.2},
                    {"date": date, "L_true": 0.8},
                ]
            )
        first_oof = pd.DataFrame(rows)
        second_oof = first_oof.copy()
        second_oof["L_true"] = [10.0, 20.0, 30.0, 40.0]
        first = diagnostics.build_fold_support_diagnostics(audits, first_oof)
        second = diagnostics.build_fold_support_diagnostics(audits, second_oof)
        train_columns = [column for column in first if column.startswith("train_")]
        pd.testing.assert_frame_equal(first[train_columns], second[train_columns])
        self.assertEqual(first["train_min"].iloc[0], 0.1)
        self.assertEqual(first["train_max"].iloc[0], 0.9)


class TestOutputIsolation(unittest.TestCase):
    def test_output_root_is_exact_and_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            safe = Path(temp_dir) / "resnet50_image_only" / "diagnostics"
            with mock.patch.object(diagnostics, "OUTPUT_ROOT", safe):
                self.assertEqual(diagnostics.validate_output_root(safe), safe.resolve())
                unsafe_paths = (
                    safe.parent / "formal_cv",
                    Path(temp_dir) / "resnet50_with_i" / "diagnostics",
                    Path(temp_dir) / "irradiance_only" / "diagnostics",
                    safe / "nested",
                )
                for unsafe in unsafe_paths:
                    with self.subTest(unsafe=unsafe):
                        with self.assertRaises(RuntimeError):
                            diagnostics.validate_output_root(unsafe)

    def test_declared_outputs_are_exactly_five_data_files_and_four_figures(self):
        self.assertEqual(len(diagnostics.GENERATED_RELATIVE_FILES), 9)
        self.assertEqual(
            sum(name.startswith("figures/") for name in diagnostics.GENERATED_RELATIVE_FILES),
            4,
        )


if __name__ == "__main__":
    unittest.main()
