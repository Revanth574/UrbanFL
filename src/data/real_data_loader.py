import json
import logging
import pickle
from pathlib import Path
import networkx as nx

logger = logging.getLogger("urbanfl.data.real")

class RealDataLoader:
    """
    Loads real-world Istanbul datasets required for the UrbanCityOS backend simulation.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        # data/real is at the root of the UrbanFL project
        # This file is in src/data, so go up two levels
        self.data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "real"
        
        self._hospitals = []
        self._earthquakes = []
        self._reports = []
        self._population = {}
        self._road_graph = None

    def load_all(self):
        """Load all core datasets from disk."""
        if self.verbose:
            logger.info(f"Loading real datasets from {self.data_dir}...")
            
        # load hospital data
        hosp_file = self.data_dir / "istanbul_hospitals.json"
        if hosp_file.exists():
            with open(hosp_file, 'r', encoding='utf-8') as f:
                self._hospitals = json.load(f)
                
        # load earthquake data
        eq_file = self.data_dir / "usgs_earthquakes.json"
        if eq_file.exists():
            with open(eq_file, 'r', encoding='utf-8') as f:
                self._earthquakes = json.load(f)
                
        # load crisis reports
        rep_file = self.data_dir / "crisis_reports.json"
        if rep_file.exists():
            with open(rep_file, 'r', encoding='utf-8') as f:
                self._reports = json.load(f)

        # load population data
        pop_file = self.data_dir / "istanbul_population.json"
        if pop_file.exists():
            with open(pop_file, 'r', encoding='utf-8') as f:
                self._population = json.load(f)

    def get_istanbul_bbox(self):
        """Returns the European Istanbul bounding box."""
        # Using a restricted bounding box covering European Istanbul
        # lon_min, lat_min, lon_max, lat_max
        return {"lon_min": 28.5, "lat_min": 40.9, "lon_max": 29.05, "lat_max": 41.25}

    def get_hospitals(self):
        """Returns the list of hospitals."""
        if isinstance(self._hospitals, dict):
            return self._hospitals.get("hospitals", [])
        return self._hospitals if self._hospitals else []

    def get_road_graph(self):
        """Returns the NetworkX road graph from a gpickle file."""
        if self._road_graph is None:
            rg_file = self.data_dir / "istanbul_roads.gpickle"
            if rg_file.exists():
                if self.verbose:
                    logger.info(f"Loading road graph from {rg_file}")
                with open(rg_file, 'rb') as f:
                    self._road_graph = pickle.load(f)
            else:
                logger.warning(f"Road graph not found at {rg_file}, returning empty graph.")
                self._road_graph = nx.MultiDiGraph()
        return self._road_graph

    def get_earthquakes(self):
        """Returns earthquake events data."""
        if isinstance(self._earthquakes, dict):
            return self._earthquakes.get("earthquakes", [])
        return self._earthquakes if self._earthquakes else []

    def get_crisis_reports(self):
        """Returns crowd-sourced citizen reports."""
        if isinstance(self._reports, dict):
            return self._reports.get("reports", [])
        return self._reports if self._reports else []

    def get_population_data(self):
        """Returns Istanbul district population data."""
        return self._population

    def get_shelters(self):
        """Returns Istanbul shelter data."""
        shelter_file = self.data_dir / "istanbul_shelters.json"
        if shelter_file.exists():
            with open(shelter_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
