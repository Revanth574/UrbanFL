"""
Machine-specific data locations. Paths are configuration, never committed and never hardcoded.

Resolution order for a setting such as "xbd_root":
  1. environment variable URBANFL_<NAME> (e.g. URBANFL_XBD_ROOT=C:/data/xbd);
  2. the user config file: $URBANFL_CONFIG if set, else ~/.urbanfl/paths.json — a JSON object such as
     {"xbd_root": "C:/data/xbd"}. It lives in the user's home directory, outside the repository;
  3. otherwise a ConfigError that says how to set it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

USER_CONFIG = Path.home() / ".urbanfl" / "paths.json"


class ConfigError(RuntimeError):
    pass


def user_config_path() -> Path:
    return Path(os.environ["URBANFL_CONFIG"]) if os.environ.get("URBANFL_CONFIG") else USER_CONFIG


def data_path(name: str, must_exist: bool = True) -> Path:
    env = f"URBANFL_{name.upper()}"
    value = os.environ.get(env)
    if not value:
        cfg = user_config_path()
        if cfg.exists():
            value = json.loads(cfg.read_text(encoding="utf-8")).get(name)
    if not value:
        raise ConfigError(f"Data path '{name}' is not configured. Set the environment variable {env}, or add "
                          f'{{"{name}": "<path>"}} to {user_config_path()} (outside the repository).')
    path = Path(value).expanduser()
    if must_exist and not path.exists():
        raise ConfigError(f"Configured path for '{name}' does not exist: {path}")
    return path


def xbd_split_dir(split: str = "train") -> Path:
    """<xbd_root>/<split>, expected to contain images/, labels/ and targets/."""
    d = data_path("xbd_root") / split
    missing = [s for s in ("images", "labels", "targets") if not (d / s).is_dir()]
    if missing:
        raise ConfigError(f"xBD split folder {d} is missing {missing}")
    return d
