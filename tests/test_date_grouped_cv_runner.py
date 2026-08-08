import inspect
import unittest

from experiments import run_resnet50_with_i_date_grouped_cv as runner


class DateGroupedCVRunnerTests(unittest.TestCase):
    def test_formal_configuration_is_frozen(self):
        self.assertEqual(runner.FOLDS, (1, 2, 3, 4))
        self.assertEqual(runner.SEED, 42)
        self.assertEqual(runner.EPOCHS, 50)
        self.assertEqual(runner.BATCH_SIZE, 32)
        self.assertEqual(runner.NUM_WORKERS, 4)
        self.assertEqual(runner.LEARNING_RATE, 0.0001)
        self.assertEqual(runner.WEIGHT_DECAY, 0.0001)
        self.assertEqual(runner.PATIENCE, 8)
        self.assertTrue(runner.AMP)
        self.assertTrue(runner.PRETRAINED)
        runner.validate_frozen_config()

    def test_runner_has_no_checkpoint_loading_or_overwrite_option(self):
        source = inspect.getsource(runner)
        self.assertNotIn("torch.load", source)
        parser_source = inspect.getsource(runner.parse_args)
        self.assertNotIn("overwrite", parser_source)
        args = runner.parse_args(["--start-fold", "2", "--end-fold", "4"])
        self.assertEqual((args.start_fold, args.end_fold), (2, 4))

    def test_each_fold_preflight_is_isolated_and_frozen(self):
        for fold, expected_counts in runner.training.EXPECTED_FOLD_COUNTS.items():
            audit = runner.preflight_formal_fold(fold)
            self.assertEqual(
                (len(audit["train_records"]), len(audit["validation_records"])),
                expected_counts,
            )
            self.assertEqual(
                audit["forbidden_role_counts"],
                {
                    "cp_calibration": 0,
                    "decision_development": 0,
                    "final_test": 0,
                },
            )
            self.assertEqual(
                audit["manifest_sha256"], runner.EXPECTED_MANIFEST_SHA256
            )
            self.assertTrue(
                set(audit["train_dates"]).isdisjoint(audit["validation_dates"])
            )

    def test_throughput_decline_requires_three_late_low_epochs(self):
        stable = [
            {"train_samples_per_second": value, "train_sample_count": 100}
            for value in (50, 100, 101, 99, 95, 96, 94)
        ]
        declining = [
            {"train_samples_per_second": value, "train_sample_count": 100}
            for value in (50, 100, 101, 99, 70, 72, 71)
        ]
        self.assertFalse(runner.throughput_stability(stable)[
            "sustained_throughput_decline"
        ])
        self.assertTrue(runner.throughput_stability(declining)[
            "sustained_throughput_decline"
        ])

    def test_aggregate_uses_sample_standard_deviation(self):
        rows = []
        for fold, value in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
            row = {field: value for field in runner.SUMMARY_NUMERIC_FIELDS}
            row["fold"] = fold
            rows.append(row)
        aggregate = runner.aggregate_rows(rows)
        self.assertAlmostEqual(aggregate["validation_rmse"]["mean"], 2.5)
        self.assertAlmostEqual(
            aggregate["validation_rmse"]["std"], 1.2909944487358056
        )


if __name__ == "__main__":
    unittest.main()
