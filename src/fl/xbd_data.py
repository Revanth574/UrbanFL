"""
Silo 2 (xBD) data loading for the two-stage xView2 task (decision X1, docs/AUDIT_XBD.md).

  stage "loc"    : input = pre image; target = building footprint from the PRE target (0 background, 1 building).
                   Un-classified buildings ARE buildings here (the pre target paints them as 1).
  stage "damage" : input = pre + post images; target = 4-class damage on building pixels from the POST target
                   (post value 1..4 -> class 0..3 = no / minor / major / destroyed). Background pixels and
                   UN-CLASSIFIED buildings get IGNORE_INDEX (X2): the post target paints un-classified buildings as
                   background (0), so they are re-rasterised from the label polygons and masked out.

Training samples are random crop x crop windows (default 512 of 1024) with random flips / 90-degree rotations applied
identically to every array; evaluation samples are full images. Images are ImageNet-normalised float tensors (CHW).
The data location comes from src/fl/config.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely import wkt
from torch.utils.data import Dataset

IGNORE_INDEX = 255
DAMAGE_CLASSES = ("no-damage", "minor-damage", "major-damage", "destroyed")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
STAGES = ("loc", "damage")


def rasterize_unclassified(label_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask of the 'un-classified' building polygons of a post-disaster label file."""
    feats = json.loads(label_path.read_text(encoding="utf-8"))["features"]["xy"]
    mask = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    for f in feats:
        if f["properties"].get("subtype") != "un-classified":
            continue
        geom = wkt.loads(f["wkt"])
        for poly in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
            draw.polygon([(x, y) for x, y in poly.exterior.coords], fill=1)
    return np.asarray(mask, dtype=bool)


def loc_target(pre_target: np.ndarray) -> np.ndarray:
    return (pre_target > 0).astype(np.int64)


def damage_target(post_target: np.ndarray, unclassified: np.ndarray) -> np.ndarray:
    t = np.full(post_target.shape, IGNORE_INDEX, dtype=np.int64)
    building = (post_target >= 1) & (post_target <= 4)
    t[building] = post_target[building].astype(np.int64) - 1
    t[unclassified] = IGNORE_INDEX
    return t


def random_crop_flip(arrays: Sequence[np.ndarray], crop: int | None, rng: np.random.Generator) -> list[np.ndarray]:
    """Same random crop / flips / rot90 for every array (H, W, ...). crop=None keeps the full size."""
    h, w = arrays[0].shape[:2]
    if crop is not None and crop < min(h, w):
        y, x = int(rng.integers(0, h - crop + 1)), int(rng.integers(0, w - crop + 1))
        arrays = [a[y:y + crop, x:x + crop] for a in arrays]
    if rng.random() < 0.5:
        arrays = [a[:, ::-1] for a in arrays]
    if rng.random() < 0.5:
        arrays = [a[::-1, :] for a in arrays]
    k = int(rng.integers(0, 4))
    return [np.ascontiguousarray(np.rot90(a, k, axes=(0, 1))) for a in arrays]


def _to_tensor_image(img: np.ndarray) -> torch.Tensor:
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class XBDPairDataset(Dataset):
    """
    One item per pre/post pair id ("<event>_<8 digits>"). Returns a dict with
      pre (3,H,W) float, post (3,H,W) float [damage stage only], target (H,W) int64,
      building (H,W) bool = ground-truth building pixels of the scored mask, pair_id, event.
    """

    def __init__(self, split_dir: Path, pair_ids: Sequence[str], stage: str, *, train: bool = False,
                 crop: int | None = 512, seed: int = 0):
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        self.dir, self.ids, self.stage = Path(split_dir), list(pair_ids), stage
        self.train, self.crop, self.seed, self.epoch = train, crop, seed, 0

    def set_epoch(self, epoch: int) -> None:
        """Augmentations depend on (seed, epoch, index): reproducible and different every epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.ids)

    def _img(self, pid: str, phase: str) -> np.ndarray:
        return np.asarray(Image.open(self.dir / "images" / f"{pid}_{phase}_disaster.png").convert("RGB"))

    def _target(self, pid: str, phase: str) -> np.ndarray:
        return np.asarray(Image.open(self.dir / "targets" / f"{pid}_{phase}_disaster_target.png"))

    def __getitem__(self, i: int) -> dict:
        return self.get(i, self.epoch)

    def get(self, i: int, epoch: int) -> dict:
        """Item i with the augmentation of `epoch`. Persistent DataLoader workers hold their own dataset copy, so the
        epoch travels with the index (src/fl/client_loader.py) instead of via set_epoch."""
        pid = self.ids[i]
        pre = self._img(pid, "pre")
        if self.stage == "loc":
            target = loc_target(self._target(pid, "pre"))
            arrays = [pre, target]
        else:
            post = self._img(pid, "post")
            post_t = self._target(pid, "post")
            uncl = rasterize_unclassified(self.dir / "labels" / f"{pid}_post_disaster.json", post_t.shape)
            target = damage_target(post_t, uncl)
            arrays = [pre, post, target]
        if self.train:
            rng = np.random.default_rng([self.seed, epoch, i])
            arrays = random_crop_flip(arrays, self.crop, rng)
        target = torch.from_numpy(np.ascontiguousarray(arrays[-1]))
        item = {"pre": _to_tensor_image(arrays[0]), "target": target,
                "building": target == 1 if self.stage == "loc" else target != IGNORE_INDEX,
                "pair_id": pid, "event": pid.rsplit("_", 1)[0]}
        if self.stage == "damage":
            item["post"] = _to_tensor_image(arrays[1])
        return item


def client_datasets(split_dir: Path, part: dict, split: str, stage: str, **kw) -> dict[str, XBDPairDataset]:
    """One dataset per event client for split in {'train','val','test'}; FL clients for Silo 2."""
    return {e: XBDPairDataset(split_dir, c[split], stage, train=(split == "train"), **kw)
            for e, c in part["clients"].items()}
