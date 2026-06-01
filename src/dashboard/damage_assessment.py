import logging
import json
import os
import re
from pathlib import Path
import numpy as np
from typing import Dict, Any, List
from enum import Enum

logger = logging.getLogger("urbanfl.dashboard.damage")

class DamageClass(Enum):
    DESTROYED = "destroyed"
    MAJOR = "major_damage"
    MINOR = "minor_damage"
    INTACT = "intact"

class BuildingReport:
    def __init__(self, uid: str, damage_class: DamageClass, lat: float, lon: float, img_prefix: str, split: str, disaster: str):
        self.uid = uid
        self.damage_class = damage_class
        self.lat = lat
        self.lon = lon
        self.img_prefix = img_prefix
        self.split = split
        self.disaster = disaster

    def to_dict(self):
        return {
            "id": self.uid,
            "damage_class": self.damage_class.value,
            "lat": self.lat,
            "lon": self.lon,
            "img_prefix": self.img_prefix,
            "split": self.split,
            "disaster": self.disaster
        }

class DamageAssessor:
    """
    Evaluates building and infrastructure damage based on visual/satellite changes.
    Translates raw pixel change detections into structural damage categories.
    """
    def __init__(self):
        self.stats = {
            "destroyed": 0,
            "major_damage": 0,
            "minor_damage": 0,
            "intact": 0
        }
        self.all_reports: List[BuildingReport] = []
        self.geotransforms = {}
        self.xbd_root = Path(__file__).resolve().parent.parent.parent / "data" / "real" / "xbd" / "geotiffs"

    def load_real_data(self, max_samples: int = 1500):
        """Loads real xBD labels and converts pixel coords to lat/lon via geotransforms."""
        gt_path = Path(__file__).resolve().parent.parent.parent / "data" / "real" / "xbd_geotransforms.json"
        if not gt_path.exists():
            logger.error(f"Geotransforms not found at {gt_path}")
            return

        with open(gt_path, 'r') as f:
            self.geotransforms = json.load(f)

        reports = []
        # Priority splits for better data quality
        for split in ["tier1", "hold", "tier3"]:
            labels_dir = self.xbd_root / split / "labels"
            if not labels_dir.exists(): continue
            
            for label_file in labels_dir.glob("*_post_disaster.json"):
                if len(reports) >= max_samples: break
                
                try:
                    with open(label_file, 'r') as f:
                        data = json.load(f)
                    
                    img_name = data.get("metadata", {}).get("img_name", "")
                    gt = self.geotransforms.get(img_name)
                    if not gt: continue
                    affine = gt[0] # [lon_origin, lon_res, 0, lat_origin, 0, lat_res]
                    
                    # Extract disaster name: guatemala-volcano_00000004_post_disaster.png -> guatemala-volcano
                    disaster = img_name.split('_')[0] if '_' in img_name else "unknown"
                    img_prefix = img_name.replace("_post_disaster.png", "")
                    
                    for feat in data.get("features", {}).get("xy", []):
                        props = feat.get("properties", {})
                        subtype = props.get("subtype", "no-damage")
                        
                        # Map xBD subtypes to internal Enum
                        dclass = DamageClass.INTACT
                        if subtype == "destroyed": dclass = DamageClass.DESTROYED
                        elif subtype == "major-damage": dclass = DamageClass.MAJOR
                        elif subtype == "minor-damage": dclass = DamageClass.MINOR
                        
                        # Only take damage samples for dashboard impact if we have too many
                        if dclass == DamageClass.INTACT and len(reports) > max_samples // 2:
                            continue

                        # Extract centroid from WKT or just use first point for speed
                        wkt = feat.get("wkt", "")
                        match = re.search(r"POLYGON\s*\(\(([\d\.]+)\s+([\d\.]+)", wkt)
                        if match:
                            px_x, px_y = float(match.group(1)), float(match.group(2))
                            # Apply affine transform: lon = a + b*x + c*y, lat = d + e*x + f*y
                            lon = affine[0] + px_x * affine[1] + px_y * affine[2]
                            lat = affine[3] + px_x * affine[4] + px_y * affine[5]
                            
                            rep = BuildingReport(props.get("uid", "unk"), dclass, lat, lon, img_prefix, split, disaster)
                            reports.append(rep)
                            self.stats[dclass.value] += 1
                except Exception as e:
                    continue

        self.all_reports = reports
        logger.info(f"DamageAssessor loaded {len(reports)} real building reports across {len(self.get_disasters())} disasters.")

    def get_disasters(self) -> List[str]:
        """Returns sorted list of unique disaster names loaded."""
        return sorted(list(set(r.disaster for r in self.all_reports)))

    def get_reports_for_disaster(self, disaster_name: str) -> List[BuildingReport]:
        """Filters reports by disaster region."""
        return [r for r in self.all_reports if r.disaster == disaster_name]

    def assess_synthetic(self, n_cells: int = 200, bbox: Dict[str, float] = None, seed: int = 42):
        """Fallback synthetic baseline."""
        rng = np.random.RandomState(seed)
        destroyed = int(n_cells * 0.05)
        major = int(n_cells * 0.15)
        minor = int(n_cells * 0.30)
        intact = n_cells - destroyed - major - minor
        self.stats = {"destroyed": destroyed, "major_damage": major, "minor_damage": minor, "intact": intact}
        logger.info(f"DamageAssessor generated synthetic baseline: {self.stats}")
        
    def adjust_from_vision_silo(self, destroyed_delta: int, major_delta: int, minor_delta: int):
        self.stats["destroyed"] += destroyed_delta
        self.stats["major_damage"] += major_delta
        self.stats["minor_damage"] += minor_delta
        loss = destroyed_delta + major_delta + minor_delta
        self.stats["intact"] = max(0, self.stats["intact"] - loss)
