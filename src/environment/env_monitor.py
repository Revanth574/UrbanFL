import json
import random
import logging

logger = logging.getLogger("urbanfl.environment")

class EnvironmentalMonitor:
    def __init__(self, bbox: dict):
        self.bbox = bbox
        self._readings = {"aqi": 42, "temp": 21.5, "humidity": 60}
        self._sensors = []
        self._tick_index = 0
        
    def load_real_data(self, json_path: str):
        """Loads real AQI sensor data from JSON."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            self._sensors = data.get("openaq_stations", [])
            logger.info(f"Loaded {len(self._sensors)} real environmental sensors.")
            self.tick() # Initial load
        except Exception as e:
            logger.error(f"Failed to load real AQI data: {e}")

    def tick(self):
        """Cycles through historical sensor readings to feel 'live'."""
        if not self._sensors: return
        
        sensor = self._sensors[self._tick_index % len(self._sensors)]
        latest = sensor.get("latest_measurements", {})
        
        # OpenAQ v3 uses 'pm25' or 'parameter' names
        # We'll pull any available value or synthesize relative to 50
        val = 50.0
        for p in latest.values():
            if isinstance(p, dict) and p.get("value"):
                val = p["value"]
                break
        
        self._readings = {
            "aqi": val,
            "temp": 20.0 + random.uniform(-2, 5),
            "humidity": 55 + random.uniform(-10, 10),
            "station": sensor.get("name", "Unknown")
        }
        self._tick_index += 1

    def compute_risk_overlay(self) -> dict:
        return {"max_risk": self._readings["aqi"] / 100.0, "high_risk_cells": int(self._readings["aqi"] > 70)}
        
    def status_summary(self):
        aqi = self._readings["aqi"]
        status = "Good"
        if aqi > 50: status = "Moderate"
        if aqi > 100: status = "Unhealthy"
        
        return {
            "aqi": round(aqi, 1),
            "status": status,
            "temp": round(self._readings["temp"], 1),
            "station": self._readings["station"]
        }
