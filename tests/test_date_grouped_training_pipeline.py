import inspect
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from experiments import train_resnet50_with_i_date_grouped as training


class DateGroupedTrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.manifest = cls.root / "splits" / "date_grouped_v1" / "split_manifest.csv"
        cls.image_root = cls.root / "data" / "raw" / "PanelImages"

    def test_training_source_does_not_load_old_checkpoints(self):
        source = inspect.getsource(training)
        self.assertNotIn("torch.load", source)
        self.assertNotIn("load_state_dict(torch.load", source)
        parser_source = inspect.getsource(training.parse_args)
        self.assertNotIn("checkpoint", parser_source)
        self.assertNotIn("resume", parser_source)

    def test_only_folds_one_to_four_are_allowed(self):
        for fold in range(1, 5):
            training.validate_fold_number(fold)
        for invalid in (0, 5, -1, True):
            with self.assertRaises(ValueError):
                training.validate_fold_number(invalid)

    def test_existing_output_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                training.prepare_output_directory(output, overwrite=False)
            self.assertEqual(
                training.prepare_output_directory(output, overwrite=True),
                output.resolve(),
            )

    def test_checkpoint_metadata_fields_are_complete(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        context = {
            "fold": 1,
            "seed": 42,
            "split_version": "date_grouped_v1",
            "manifest_sha256": "abc",
            "dataset_fingerprint": {"manifest_sha256": "abc"},
            "train_dates": ["2017-06-14"],
            "validation_dates": ["2017-06-13"],
            "pretrained_used": True,
            "pretrained_source": "torchvision:test",
            "pretrained_load_success": True,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "training_version": "test",
            "pilot_run": True,
        }
        payload = training.build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=1,
            run_context=context,
            best_validation_metrics={"rmse": 0.1},
            current_metrics={"validation": {"rmse": 0.1}},
        )
        self.assertTrue(training.REQUIRED_CHECKPOINT_FIELDS.issubset(payload))
        self.assertTrue(payload["NOT_FOR_RESEARCH_METRICS"])

    def test_loss_average_is_weighted_by_sample_count(self):
        result = training.sample_weighted_average([0.25, 0.81], [8, 2])
        self.assertAlmostEqual(result, (0.25 * 8 + 0.81 * 2) / 10)
        self.assertNotAlmostEqual(result, (0.25 + 0.81) / 2)

        accumulator = training.MetricAccumulator()
        accumulator.update(
            0.25,
            torch.tensor([[0.0], [0.0]]),
            torch.tensor([[0.5], [0.5]]),
        )
        accumulator.update(
            1.0,
            torch.tensor([[0.0]]),
            torch.tensor([[1.0]]),
        )
        self.assertAlmostEqual(accumulator.compute()["loss"], 0.5)

    def test_final_test_and_other_forbidden_roles_are_absent(self):
        audit = training.preflight_fold(self.manifest, self.image_root, 1)
        self.assertEqual(
            audit["forbidden_role_counts"],
            {
                "cp_calibration": 0,
                "decision_development": 0,
                "final_test": 0,
            },
        )

    def test_manifest_hash_is_platform_independent(self):
        expected = "a354afc2b691719bf0cc3c3982033da833795006e3e3b0122cae07810bd83e02"
        self.assertEqual(training.canonical_manifest_sha256(self.manifest), expected)
        with tempfile.TemporaryDirectory() as temp_dir:
            crlf_copy = Path(temp_dir) / "manifest.csv"
            normalized = self.manifest.read_bytes().replace(b"\r\n", b"\n")
            crlf_copy.write_bytes(normalized.replace(b"\n", b"\r\n"))
            self.assertEqual(training.canonical_manifest_sha256(crlf_copy), expected)

    def test_training_config_load_is_stable(self):
        first = training.load_training_config()
        second = training.load_training_config()
        self.assertEqual(first, second)
        self.assertEqual(first["split_version"], "date_grouped_v1")
        self.assertEqual(first["batch_size"], 32)
        self.assertEqual(first["num_workers"], 4)
        self.assertTrue(first["amp"])
        self.assertTrue(first["pretrained"])

    def test_same_seed_has_same_initial_data_order(self):
        first = torch.randperm(
            100, generator=training.make_data_loader_generator(42)
        ).tolist()
        second = torch.randperm(
            100, generator=training.make_data_loader_generator(42)
        ).tolist()
        different = torch.randperm(
            100, generator=training.make_data_loader_generator(43)
        ).tolist()
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)


if __name__ == "__main__":
    unittest.main()
