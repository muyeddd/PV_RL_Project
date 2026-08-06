import json
import math
import unittest
from pathlib import Path

import pandas as pd

from decision.rule_based import Decision, ReasonCode, apply_cleaning_rules


def configured_rules():
    return {
        "schema_version": 1,
        "config_version": "unit-test-v1",
        "interval_method": "pred_l_mondrian_std_mc",
        "confidence": 0.9,
        "loss_threshold": 0.2,
        "min_irradiance": 0.1,
        "max_interval_width": 0.2,
    }


def input_row(**overrides):
    row = {
        "filename": "sample.jpg",
        "pred_L": 0.2,
        "pred_std": 0.02,
        "irradiance": 0.5,
        "pred_l_mondrian_std_mc_lower": 0.15,
        "pred_l_mondrian_std_mc_upper": 0.25,
        "pred_l_mondrian_std_mc_width": 0.10,
    }
    row.update(overrides)
    return row


class RuleBasedCleaningTests(unittest.TestCase):
    def test_all_six_rule_branches_and_priority_order(self):
        df = pd.DataFrame(
            [
                input_row(filename="invalid.jpg", pred_L=math.nan),
                input_row(
                    filename="low_i.jpg",
                    irradiance=0.05,
                    pred_l_mondrian_std_mc_lower=0.22,
                    pred_l_mondrian_std_mc_upper=0.32,
                ),
                input_row(
                    filename="uncertain.jpg",
                    pred_l_mondrian_std_mc_lower=0.10,
                    pred_l_mondrian_std_mc_upper=0.35,
                    pred_l_mondrian_std_mc_width=0.25,
                ),
                input_row(
                    filename="clean.jpg",
                    pred_L=0.25,
                    pred_l_mondrian_std_mc_lower=0.20,
                    pred_l_mondrian_std_mc_upper=0.30,
                ),
                input_row(
                    filename="wait.jpg",
                    pred_L=0.14,
                    pred_l_mondrian_std_mc_lower=0.10,
                    pred_l_mondrian_std_mc_upper=0.19,
                    pred_l_mondrian_std_mc_width=0.09,
                ),
                input_row(filename="crossing.jpg"),
            ]
        )

        result = apply_cleaning_rules(df, configured_rules())

        self.assertEqual(
            result["decision"].tolist(),
            [
                Decision.REVIEW.value,
                Decision.MONITOR.value,
                Decision.REVIEW.value,
                Decision.CLEAN.value,
                Decision.WAIT.value,
                Decision.MONITOR.value,
            ],
        )
        self.assertEqual(
            result["reason_code"].tolist(),
            [
                ReasonCode.REVIEW_INVALID_INPUT.value,
                ReasonCode.MONITOR_LOW_IRRADIANCE.value,
                ReasonCode.REVIEW_HIGH_UNCERTAINTY.value,
                ReasonCode.CLEAN_CONFIDENT_LOSS.value,
                ReasonCode.WAIT_CONFIDENT_LOW_LOSS.value,
                ReasonCode.MONITOR_THRESHOLD_CROSSING.value,
            ],
        )

    def test_lower_equal_to_threshold_triggers_clean(self):
        df = pd.DataFrame(
            [
                input_row(
                    pred_L=0.25,
                    pred_l_mondrian_std_mc_lower=0.20,
                    pred_l_mondrian_std_mc_upper=0.30,
                )
            ]
        )
        result = apply_cleaning_rules(df, configured_rules())

        self.assertEqual(result.loc[0, "decision"], Decision.CLEAN.value)
        self.assertEqual(result.loc[0, "clean_flag"], 1)

    def test_upper_below_threshold_triggers_wait(self):
        df = pd.DataFrame(
            [
                input_row(
                    pred_L=0.15,
                    pred_l_mondrian_std_mc_lower=0.10,
                    pred_l_mondrian_std_mc_upper=0.19,
                    pred_l_mondrian_std_mc_width=0.09,
                )
            ]
        )
        result = apply_cleaning_rules(df, configured_rules())

        self.assertEqual(result.loc[0, "decision"], Decision.WAIT.value)
        self.assertEqual(result.loc[0, "clean_flag"], 0)

    def test_invalid_inputs_trigger_review(self):
        rows = [
            input_row(filename=""),
            input_row(pred_std=math.inf),
            input_row(pred_L=-0.01),
            input_row(irradiance=-0.01),
            input_row(
                pred_l_mondrian_std_mc_lower=0.30,
                pred_l_mondrian_std_mc_upper=0.20,
                pred_l_mondrian_std_mc_width=-0.10,
            ),
            input_row(pred_l_mondrian_std_mc_width=0.11),
        ]
        result = apply_cleaning_rules(pd.DataFrame(rows), configured_rules())

        self.assertTrue((result["decision"] == Decision.REVIEW.value).all())
        self.assertTrue(
            (result["reason_code"] == ReasonCode.REVIEW_INVALID_INPUT.value).all()
        )

    def test_label_and_coverage_columns_cannot_change_decisions(self):
        base = pd.DataFrame(
            [
                input_row(filename="clean.jpg", pred_L=0.3, pred_l_mondrian_std_mc_lower=0.25, pred_l_mondrian_std_mc_upper=0.35),
                input_row(filename="wait.jpg", pred_L=0.1, pred_l_mondrian_std_mc_lower=0.05, pred_l_mondrian_std_mc_upper=0.15),
                input_row(filename="monitor.jpg"),
            ]
        )
        base["true_L"] = [0.0, 1.0, 0.5]
        base["abs_error"] = [0.9, 0.8, 0.7]
        base["covered_mc_90"] = [0, 0, 0]
        base["pred_l_mondrian_std_mc_covered"] = [0, 1, 0]

        changed = base.copy()
        changed["true_L"] = [1.0, 0.0, 0.0]
        changed["abs_error"] = [0.0, 0.0, 0.0]
        changed["covered_mc_90"] = [1, 1, 1]
        changed["pred_l_mondrian_std_mc_covered"] = [1, 0, 1]

        first = apply_cleaning_rules(base, configured_rules())
        second = apply_cleaning_rules(changed, configured_rules())

        pd.testing.assert_frame_equal(first, second)
        self.assertNotIn("true_L", first.columns)
        self.assertNotIn("abs_error", first.columns)
        self.assertFalse(any("covered" in name for name in first.columns))

    def test_same_input_and_config_are_deterministic(self):
        df = pd.DataFrame(
            [
                input_row(filename="one.jpg"),
                input_row(
                    filename="two.jpg",
                    pred_L=0.3,
                    pred_l_mondrian_std_mc_lower=0.25,
                    pred_l_mondrian_std_mc_upper=0.35,
                ),
            ]
        )

        first = apply_cleaning_rules(df, configured_rules())
        second = apply_cleaning_rules(df.copy(deep=True), dict(configured_rules()))

        pd.testing.assert_frame_equal(first, second)

    def test_null_threshold_config_is_rejected_without_defaults(self):
        config_path = Path(__file__).parents[1] / "configs" / "rule_based_cleaning.json"
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        with self.assertRaisesRegex(
            ValueError,
            "loss_threshold, min_irradiance, max_interval_width",
        ):
            apply_cleaning_rules(pd.DataFrame([input_row()]), config)


if __name__ == "__main__":
    unittest.main()
