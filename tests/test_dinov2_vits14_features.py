from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import extract_dinov2_vits14_features as extraction


class FeatureTestFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.image_root = self.base / "images"
        self.image_root.mkdir()
        rows = []
        for fold, suffix in ((1, "d"), (2, "b"), (3, "a"), (4, "c")):
            filename = f"solar_{suffix}_L_0.{fold}_I_0.{fold}.jpg"
            rows.append(
                {
                    "filename": filename,
                    "date": f"2017-06-{12 + fold:02d}",
                    "top_level_role": "model_development",
                    "cv_validation_fold": fold,
                }
            )
            Image.new("L", (20, 16), color=30 * fold).save(self.image_root / filename)
        for index, role in enumerate(
            ("cp_calibration", "decision_development", "final_test"), start=1
        ):
            # Deliberately do not create protected-role image files. Selection must
            # discard these records before resolving a path or parsing a label.
            rows.append(
                {
                    "filename": f"protected_{index}_L_0.5_I_0.5.jpg",
                    "date": f"2017-06-{22 + index:02d}",
                    "top_level_role": role,
                    "cv_validation_fold": np.nan,
                }
            )
        self.manifest = self.base / "manifest.csv"
        pd.DataFrame(rows).to_csv(self.manifest, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def records(self):
        return extraction.load_model_development_records(
            self.manifest, self.image_root, expected_count=4
        )


class DataBoundaryTests(FeatureTestFixture):
    def test_only_model_development_is_selected(self):
        self.assertEqual(set(self.records()["top_level_role"]), {"model_development"})

    def test_protected_roles_are_not_resolved(self):
        records = self.records()
        self.assertEqual(len(records), 4)
        self.assertFalse(records["filename"].str.startswith("protected_").any())

    def test_metadata_order_is_stable_filename_ascending(self):
        first = self.records()["filename"].tolist()
        second = self.records()["filename"].tolist()
        self.assertEqual(first, sorted(first))
        self.assertEqual(first, second)

    def test_cv_validation_fold_is_preserved(self):
        records = self.records()
        observed = dict(zip(records["filename"], records["cv_validation_fold"]))
        self.assertEqual(set(observed.values()), {1, 2, 3, 4})

    def test_metadata_excludes_irradiance(self):
        self.assertNotIn("I", extraction.METADATA_COLUMNS)
        self.assertEqual(
            extraction.METADATA_COLUMNS,
            ("row_index", "filename", "date", "cv_validation_fold", "L"),
        )


class TransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = extraction.load_config()
        cls.transform = extraction.build_deterministic_transform(cls.config)

    def test_transform_has_exact_deterministic_sequence(self):
        self.assertEqual(
            [type(item) for item in self.transform.transforms],
            [transforms.Resize, transforms.ToTensor, transforms.Normalize],
        )
        self.assertEqual(self.transform.transforms[0].size, (224, 224))

    def test_transform_contains_no_random_augmentation(self):
        self.assertFalse(
            any(type(item).__name__.startswith("Random") for item in self.transform.transforms)
        )

    def test_transform_is_deterministic_and_rgb_compatible(self):
        image = Image.new("RGB", (31, 19), color=(10, 100, 240))
        first = self.transform(image)
        second = self.transform(image)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (3, 224, 224))

    def test_normalization_is_imagenet(self):
        normalization = self.transform.transforms[-1]
        self.assertEqual(list(normalization.mean), [0.485, 0.456, 0.406])
        self.assertEqual(list(normalization.std), [0.229, 0.224, 0.225])


class RecordingBackbone(nn.Module):
    def __init__(self, non_finite: bool = False):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.non_finite = non_finite
        self.grad_enabled_during_forward: list[bool] = []

    def forward(self, images):
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        output = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).repeat(1, 384)
        if self.non_finite:
            output[0, 0] = torch.nan
        return output


class ExtractionContractTests(FeatureTestFixture):
    def run_fake_extraction(self, non_finite: bool = False):
        config = extraction.load_config()
        model = extraction.freeze_backbone(RecordingBackbone(non_finite=non_finite))
        result = extraction.extract_features(
            model=model,
            records=self.records(),
            transform=extraction.build_deterministic_transform(config),
            device=torch.device("cpu"),
            batch_size=32,
            num_workers=0,
            amp=False,
        )
        return model, result

    def test_feature_shape_dtype_finiteness_and_alignment(self):
        _, result = self.run_fake_extraction()
        self.assertEqual(result.features.shape, (4, 384))
        self.assertEqual(result.features.dtype, np.float32)
        self.assertTrue(np.isfinite(result.features).all())
        self.assertEqual(result.metadata["row_index"].tolist(), list(range(4)))
        self.assertEqual(result.metadata["filename"].tolist(), sorted(result.metadata["filename"]))

    def test_extraction_runs_without_gradient_computation(self):
        model, _ = self.run_fake_extraction()
        self.assertTrue(model.grad_enabled_during_forward)
        self.assertFalse(any(model.grad_enabled_during_forward))

    def test_backbone_is_fully_frozen_and_eval(self):
        model = extraction.freeze_backbone(RecordingBackbone())
        self.assertFalse(model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_nan_or_inf_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            self.run_fake_extraction(non_finite=True)


class FrozenConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = extraction.load_config()

    def test_batch_size_feature_dimension_and_sample_count_are_frozen(self):
        self.assertEqual(self.config["batch_size"], 32)
        self.assertEqual(self.config["feature_dimension"], 384)
        self.assertEqual(self.config["expected_sample_count"], 25716)

    def test_output_path_is_fully_isolated(self):
        output = extraction.validate_output_path(self.config)
        self.assertEqual(output, ROOT / "features" / "dinov2_vits14_frozen_v1")
        self.assertNotIn("outputs", output.parts)
        self.assertNotIn("resnet50_image_only", str(output))

    def test_date_grouped_v1_source_hashes_are_unchanged(self):
        actual = extraction.verify_source_artifacts(self.config)
        self.assertEqual(set(actual), set(self.config["source_artifact_sha256"]))

    def test_official_cached_model_and_weight_are_required(self):
        repository, weight = extraction.torch_hub_paths(self.config)
        self.assertTrue((repository / "hubconf.py").is_file())
        self.assertEqual(weight.name, "dinov2_vits14_pretrain.pth")


@unittest.skipUnless(
    os.environ.get("RUN_DINOV2_CACHED_INTEGRATION") == "1",
    "cached DINOv2 integration test is opt-in",
)
class CachedBackboneIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = extraction.load_config()
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.model, cls.provenance = extraction.load_cached_official_backbone(
            cls.config, cls.device
        )

    def test_official_dinov2_vits14_dimension_is_384(self):
        self.assertEqual(getattr(self.model, "embed_dim"), 384)

    def test_official_backbone_has_zero_trainable_parameters(self):
        self.assertGreater(self.provenance["total_parameters"], 0)
        self.assertEqual(self.provenance["trainable_parameters"], 0)
        self.assertFalse(self.model.training)


if __name__ == "__main__":
    unittest.main()
