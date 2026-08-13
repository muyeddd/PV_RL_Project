from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

import experiments.train_paper1_resnet50_with_i_v1 as train


def frame(role: str, date: str = "2017-06-13", sample: str = "a") -> pd.DataFrame:
    return pd.DataFrame([{"sample_id":sample,"image_path":f"data/raw/PanelImages/{sample}_L_0.2_I_0.4.jpg",
                          "date":date,"timestamp":date+"T10:00:00","role":role}])


def test_role_guard():
    with pytest.raises(PermissionError):
        train.validate_role_isolation(frame("CP_CALIBRATION"),frame("MODEL_VALIDATION",sample="b"))


@pytest.mark.parametrize("name",["random_test.csv","cp_calibration.csv","decision_development.csv"])
def test_forbidden_manifest_rejection(tmp_path,name):
    path=tmp_path/name; frame("RANDOM_TEST").to_csv(path,index=False)
    with pytest.raises(PermissionError): train.load_role_manifest(path,"TRAIN")


def test_sealed_date_rejection(tmp_path,monkeypatch):
    monkeypatch.setitem(train.EXPECTED_N,"TRAIN",1)
    path=tmp_path/"train.csv"; frame("TRAIN","2017-06-15").to_csv(path,index=False)
    with pytest.raises(PermissionError,match="Sealed"): train.load_role_manifest(path,"TRAIN")


def test_old_checkpoint_rejection():
    with pytest.raises(ValueError,match="checkpoints are forbidden"):
        train.reject_legacy_checkpoint(train.LEGACY_CHECKPOINT)
    train.reject_legacy_checkpoint(None)


def test_train_only_irradiance_statistics():
    data=pd.concat([frame("TRAIN","2017-06-13","a"),frame("TRAIN","2017-06-13","b")])
    data["irradiance_raw"]=[0.2,0.6]
    stats=train.compute_train_irradiance_stats(data)
    assert stats["mean"]==pytest.approx(.4); assert stats["std_ddof0"]==pytest.approx(.2)
    with pytest.raises(PermissionError):
        train.compute_train_irradiance_stats(data.assign(role="MODEL_VALIDATION"))


def test_deterministic_validation_preprocessing():
    _,validation=train.build_transforms()
    assert train.validation_transform_is_deterministic(validation)
    assert "Random" not in repr(validation)


def test_output_collision(tmp_path):
    target=tmp_path/"out"; target.mkdir(); (target/"keep").write_text("x")
    with pytest.raises(FileExistsError): train.ensure_output_available(target)


def test_seed_setting():
    train.set_seed(42); first=torch.rand(4)
    train.set_seed(42); second=torch.rand(4)
    assert torch.equal(first,second)


def test_regime_boundaries():
    assert train.regime_name(.099999)=="LOW"
    assert train.regime_name(.1)=="MEDIUM"
    assert train.regime_name(.499999)=="MEDIUM"
    assert train.regime_name(.5)=="HIGH"


def test_model_requires_imagenet_v2(monkeypatch):
    with pytest.raises(ValueError): train.Paper1ResNet50WithI(weights=None)


def test_no_forbidden_default_manifest_paths():
    assert train.TRAIN_MANIFEST.name=="train.csv"
    assert train.VALIDATION_MANIFEST.name=="model_validation.csv"
    assert not ({train.TRAIN_MANIFEST.name,train.VALIDATION_MANIFEST.name}&train.FORBIDDEN_MANIFEST_NAMES)
