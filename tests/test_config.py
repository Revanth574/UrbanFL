import json

import pytest

from src.fl import config


def test_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("URBANFL_XBD_ROOT", str(tmp_path))
    monkeypatch.setenv("URBANFL_CONFIG", str(tmp_path / "nope.json"))
    assert config.data_path("xbd_root") == tmp_path


def test_user_config_file(tmp_path, monkeypatch):
    root = tmp_path / "xbd"
    for s in ("images", "labels", "targets"):
        (root / "train" / s).mkdir(parents=True)
    cfg = tmp_path / "paths.json"
    cfg.write_text(json.dumps({"xbd_root": str(root)}), encoding="utf-8")
    monkeypatch.delenv("URBANFL_XBD_ROOT", raising=False)
    monkeypatch.setenv("URBANFL_CONFIG", str(cfg))
    assert config.xbd_split_dir("train") == root / "train"


def test_unconfigured_or_missing_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("URBANFL_XBD_ROOT", raising=False)
    monkeypatch.setenv("URBANFL_CONFIG", str(tmp_path / "absent.json"))
    with pytest.raises(config.ConfigError, match="not configured"):
        config.data_path("xbd_root")
    monkeypatch.setenv("URBANFL_XBD_ROOT", str(tmp_path / "does-not-exist"))
    with pytest.raises(config.ConfigError, match="does not exist"):
        config.data_path("xbd_root")
