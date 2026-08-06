import inspect
import tempfile
import unittest
from pathlib import Path

from experiments import build_date_grouped_splits as builder


class DateGroupedSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.image_dir = cls.root / "data" / "raw" / "PanelImages"
        cls.config_path = cls.root / "configs" / "splits" / "date_grouped_v1.json"
        cls.output_dir = cls.root / "splits" / "date_grouped_v1"
        cls.config = builder.load_split_config(cls.config_path)
        cls.artifacts = builder.build_split_artifacts(cls.image_dir, cls.config_path)
        cls.rows = cls.artifacts.manifest_rows

    def rows_for_role(self, role):
        return [row for row in self.rows if row["top_level_role"] == role]

    def test_total_count_and_unique_filenames(self):
        self.assertEqual(len(self.rows), 45754)
        filenames = [row["filename"] for row in self.rows]
        self.assertEqual(len(filenames), len(set(filenames)))

    def test_all_fifteen_dates_are_assigned_exactly_once(self):
        dataset_dates = {row["date"] for row in self.rows}
        configured = self.config["top_level_splits"]
        configured_dates = [date for dates in configured.values() for date in dates]

        self.assertEqual(len(dataset_dates), 15)
        self.assertEqual(set(configured_dates), dataset_dates)
        self.assertEqual(len(configured_dates), len(set(configured_dates)))

    def test_top_level_roles_have_no_filename_or_date_overlap(self):
        role_filenames = {}
        role_dates = {}
        for role in builder.TOP_LEVEL_ROLES:
            rows = self.rows_for_role(role)
            role_filenames[role] = {row["filename"] for row in rows}
            role_dates[role] = {row["date"] for row in rows}

        for index, left in enumerate(builder.TOP_LEVEL_ROLES):
            for right in builder.TOP_LEVEL_ROLES[index + 1 :]:
                self.assertTrue(role_filenames[left].isdisjoint(role_filenames[right]))
                self.assertTrue(role_dates[left].isdisjoint(role_dates[right]))

    def test_expected_top_level_counts(self):
        expected = {
            "model_development": 25716,
            "cp_calibration": 4887,
            "decision_development": 6296,
            "final_test": 8855,
        }
        actual = {
            role: len(self.rows_for_role(role)) for role in builder.TOP_LEVEL_ROLES
        }
        self.assertEqual(actual, expected)

    def test_expected_cv_validation_counts(self):
        model_rows = self.rows_for_role("model_development")
        expected = {1: 5911, 2: 7192, 3: 6500, 4: 6113}
        actual = {
            fold: sum(row["cv_validation_fold"] == fold for row in model_rows)
            for fold in expected
        }
        self.assertEqual(actual, expected)

    def test_each_fold_train_and_validation_are_disjoint_and_complete(self):
        model_rows = self.rows_for_role("model_development")
        model_filenames = {row["filename"] for row in model_rows}
        model_dates = {row["date"] for row in model_rows}

        for fold in self.config["cv_folds"]:
            fold_id = fold["fold"]
            validation = [
                row for row in model_rows if row["cv_validation_fold"] == fold_id
            ]
            train = [
                row for row in model_rows if row["cv_validation_fold"] != fold_id
            ]
            validation_filenames = {row["filename"] for row in validation}
            train_filenames = {row["filename"] for row in train}
            validation_dates = {row["date"] for row in validation}
            train_dates = {row["date"] for row in train}

            self.assertTrue(train_filenames.isdisjoint(validation_filenames))
            self.assertTrue(train_dates.isdisjoint(validation_dates))
            self.assertEqual(train_filenames | validation_filenames, model_filenames)
            self.assertEqual(train_dates | validation_dates, model_dates)
            self.assertEqual(train_dates, set(fold["train_dates"]))
            self.assertEqual(validation_dates, set(fold["validation_dates"]))

    def test_final_test_is_absent_from_all_development_sets(self):
        final_rows = self.rows_for_role("final_test")
        final_filenames = {row["filename"] for row in final_rows}
        final_dates = {row["date"] for row in final_rows}

        for role in ("cp_calibration", "decision_development"):
            rows = self.rows_for_role(role)
            self.assertTrue(final_filenames.isdisjoint(row["filename"] for row in rows))
            self.assertTrue(final_dates.isdisjoint(row["date"] for row in rows))

        model_rows = self.rows_for_role("model_development")
        for fold in self.config["cv_folds"]:
            fold_id = fold["fold"]
            train = [row for row in model_rows if row["cv_validation_fold"] != fold_id]
            validation = [
                row for row in model_rows if row["cv_validation_fold"] == fold_id
            ]
            self.assertTrue(final_filenames.isdisjoint(row["filename"] for row in train))
            self.assertTrue(
                final_filenames.isdisjoint(row["filename"] for row in validation)
            )
            self.assertTrue(final_dates.isdisjoint(row["date"] for row in train))
            self.assertTrue(final_dates.isdisjoint(row["date"] for row in validation))

    def test_generated_artifacts_match_a_fresh_build(self):
        self.assertEqual(
            (self.output_dir / "split_manifest.csv").read_bytes(),
            builder.render_manifest_csv(self.artifacts),
        )
        self.assertEqual(
            (self.output_dir / "split_summary.csv").read_bytes(),
            builder.render_summary_csv(self.artifacts),
        )
        self.assertEqual(
            (self.output_dir / "dataset_fingerprint.json").read_bytes(),
            builder.render_fingerprint_json(self.artifacts),
        )

    def test_repeated_builds_have_identical_manifest_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first_dir = Path(temp_dir) / "first"
            second_dir = Path(temp_dir) / "second"
            first = builder.build_and_write(
                self.image_dir, self.config_path, first_dir
            )
            second = builder.build_and_write(
                self.image_dir, self.config_path, second_dir
            )

            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertEqual(first.manifest_rows, second.manifest_rows)
            self.assertEqual(
                (first_dir / "split_manifest.csv").read_bytes(),
                (second_dir / "split_manifest.csv").read_bytes(),
            )
            self.assertEqual(
                (first_dir / "dataset_fingerprint.json").read_bytes(),
                (second_dir / "dataset_fingerprint.json").read_bytes(),
            )

    def test_allocation_does_not_read_label_or_irradiance_values(self):
        prefix = "solar_Tue_Jun_13_10__0__0_2017"
        first = builder.parse_filename_timestamp(
            prefix + "_L_0.01_I_0.25.jpg"
        )
        second = builder.parse_filename_timestamp(
            prefix + "_L_NOT_PARSED_I_NOT_PARSED.jpg"
        )
        self.assertEqual(first, second)
        self.assertEqual(
            builder.ALLOCATION_INPUT_FIELDS,
            ("filename", "timestamp", "date"),
        )

        allocation_source = inspect.getsource(builder.build_split_artifacts)
        self.assertNotIn("true_L", allocation_source)
        self.assertNotIn("irradiance", allocation_source)


if __name__ == "__main__":
    unittest.main()
