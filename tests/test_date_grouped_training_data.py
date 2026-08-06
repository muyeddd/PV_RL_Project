import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.date_grouped_training_data import (
    FORBIDDEN_MODEL_ROLES,
    load_fold_records,
    validate_fold_isolation,
)


class DateGroupedTrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.manifest_path = (
            cls.root / "splits" / "date_grouped_v1" / "split_manifest.csv"
        )
        cls.summary_path = (
            cls.root / "splits" / "date_grouped_v1" / "split_summary.csv"
        )
        cls.image_root = cls.root / "data" / "raw" / "PanelImages"
        cls.manifest = pd.read_csv(cls.manifest_path)
        cls.summary = pd.read_csv(cls.summary_path)
        cls.model_manifest = cls.manifest[
            cls.manifest["top_level_role"].eq("model_development")
        ].copy()

    def test_fold_one_full_counts(self):
        train, validation = load_fold_records(
            self.manifest_path, self.image_root, 1
        )
        self.assertEqual(len(train), 19805)
        self.assertEqual(len(validation), 5911)

    def test_all_fold_counts_match_split_summary(self):
        expected = {
            1: (19805, 5911),
            2: (18524, 7192),
            3: (19216, 6500),
            4: (19603, 6113),
        }
        summary_counts = {
            (int(row.cv_fold), row.subset): int(row.sample_count)
            for row in self.summary.itertuples()
            if row.scope == "cv"
        }
        for fold, (expected_train, expected_validation) in expected.items():
            train, validation = load_fold_records(
                self.manifest_path, self.image_root, fold
            )
            self.assertEqual(len(train), expected_train)
            self.assertEqual(len(validation), expected_validation)
            self.assertEqual(len(train), summary_counts[(fold, "train")])
            self.assertEqual(
                len(validation), summary_counts[(fold, "validation")]
            )

    def test_train_and_validation_have_no_filename_or_date_overlap(self):
        for fold in range(1, 5):
            train, validation = load_fold_records(
                self.manifest_path, self.image_root, fold
            )
            validate_fold_isolation(train, validation)
            self.assertTrue(
                set(train["filename"]).isdisjoint(validation["filename"])
            )
            self.assertTrue(set(train["date"]).isdisjoint(validation["date"]))

    def test_each_fold_completely_covers_model_development(self):
        expected_filenames = set(self.model_manifest["filename"])
        expected_dates = set(self.model_manifest["date"])
        for fold in range(1, 5):
            train, validation = load_fold_records(
                self.manifest_path, self.image_root, fold
            )
            self.assertEqual(
                set(train["filename"]) | set(validation["filename"]),
                expected_filenames,
            )
            self.assertEqual(
                set(train["date"]) | set(validation["date"]),
                expected_dates,
            )

    def test_forbidden_top_level_roles_never_enter_model_folds(self):
        forbidden_filenames = {
            role: set(
                self.manifest.loc[
                    self.manifest["top_level_role"].eq(role), "filename"
                ]
            )
            for role in FORBIDDEN_MODEL_ROLES
        }
        forbidden_dates = {
            role: set(
                self.manifest.loc[
                    self.manifest["top_level_role"].eq(role), "date"
                ]
            )
            for role in FORBIDDEN_MODEL_ROLES
        }

        for fold in range(1, 5):
            train, validation = load_fold_records(
                self.manifest_path, self.image_root, fold
            )
            selected_filenames = set(train["filename"]) | set(
                validation["filename"]
            )
            selected_dates = set(train["date"]) | set(validation["date"])
            for role in FORBIDDEN_MODEL_ROLES:
                self.assertTrue(
                    selected_filenames.isdisjoint(forbidden_filenames[role])
                )
                self.assertTrue(selected_dates.isdisjoint(forbidden_dates[role]))

    def test_repeated_fold_load_is_identical(self):
        first_train, first_validation = load_fold_records(
            self.manifest_path, self.image_root, 1
        )
        second_train, second_validation = load_fold_records(
            self.manifest_path, self.image_root, 1
        )
        pd.testing.assert_frame_equal(first_train, second_train)
        pd.testing.assert_frame_equal(first_validation, second_validation)

    def test_selection_is_independent_of_label_and_irradiance_columns(self):
        first_manifest = self.manifest.copy()
        second_manifest = self.manifest.copy()
        first_manifest["true_L"] = 0.0
        first_manifest["irradiance"] = 0.0
        second_manifest["true_L"] = 1.0
        second_manifest["irradiance"] = 999.0

        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "first.csv"
            second_path = Path(temp_dir) / "second.csv"
            first_manifest.to_csv(first_path, index=False)
            second_manifest.to_csv(second_path, index=False)

            first_train, first_validation = load_fold_records(
                first_path, self.image_root, 1
            )
            second_train, second_validation = load_fold_records(
                second_path, self.image_root, 1
            )

        pd.testing.assert_frame_equal(first_train, second_train)
        pd.testing.assert_frame_equal(first_validation, second_validation)
        self.assertNotIn("true_L", first_train.columns)
        self.assertNotIn("irradiance", first_train.columns)


if __name__ == "__main__":
    unittest.main()
