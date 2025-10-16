"""
Simplified unified scenario format for ScenarioMax.

This module defines a clean, simple structure for representing autonomous driving scenarios
across different datasets (Waymo, nuPlan, nuScenes, Argoverse2).
"""

from typing import Any

from scenariomax.core import types


class UnifiedScenario(dict):
    """
    Simplified unified scenario representation.

    Structure:
    {
        "id": str,                    # Unique scenario identifier

        "dynamic_agents": {           # Dynamic objects (vehicles, pedestrians, etc.)
            "obj_id": {                 # Unique object identifier
                "type": str,                # Type of participant
                "states": {                 # State data for each timestep
                    "position": np.ndarray,     # (length, 3) - x, y
                    "heading": np.ndarray,      # (length,) - heading angle
                    "velocity": np.ndarray,     # (length, 2) - vx, vy
                    "length": np.ndarray,            # (length,) - Object length
                    "width": np.ndarray,             # (length,) - Object width
                    "height": np.ndarray,            # (length,) - Object height
                    "valid": np.ndarray,        # (length,) - boolean validity mask
                },
            },
        },

        "static_map_elements": {      # Static map features
            "element_id": {             # str: Unique map element identifier
                "type": str,                    # From types.LANE_TYPES, ROAD_LINE_TYPES, etc.
                "polyline": np.ndarray,         # (N, 3) - (x, y, z)

                # IF TYPE IS LANE:
                "speed_limit_mph": float,       # Speed limit in mph # lane only
                "speed_limit_kmh": float,       # Speed limit in km/h # lane only
                "entry_lanes": list[int],       # List of entry lane IDs # lane only
                "exit_lanes": list[int],        # List of exit lane IDs # lane only
                "left_boundaries": list[int],   # Extracted left boundaries
                "right_boundaries": list[int],  # Extracted right boundaries
                "left_neighbor": list[int],     # IDs of left neighbor lanes
                "right_neighbor": list[int],    # IDs of right neighbor lanes
            },
        },

        "dynamic_map_elements": {     # Dynamic traffic light states
            "element_id": {             # str: Unique traffic light identifier
                "type": str,               # From types.TRAFFIC_LIGHT_TYPES
                "position": np.ndarray,    # (3,) - x, y, z position
                "states": list,            # (length,) - state at each timestep
                "lane": int,               # Controlled lane ID (if applicable)
            },
        },

        "metadata": {                 # Additional scenario information
            "dataset_name": str,          # Source dataset name
            "dataset_version": str,       # Dataset version
            "scenario_id": str,           # Original scenario identifier
            "source_file": str,           # Original source file name
            "length": int,                # Number of timesteps
            "timesteps": np.ndarray,      # Timestamp array (length,)
            "ego_id": str,                # Self-driving car ID
            # Waymo-specific
            "current_frame_index": int,       # Current frame index
            "sdc_track_index": int,           # SDC track index
            "objects_of_interest": list[int], # List of object indices of interest
            "tracks_to_predict": list[int],   # List of track indices to predict
        },
    }
    """

    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedScenario":
        """Create a UnifiedScenario from a dictionary."""
        metadata = data.get("metadata", {}) or {}

        scenario = cls(
            scenario_id=data.get("id", ""),
            dataset_name=metadata.get("dataset_name", data.get("dataset_name", "")),
            dataset_version=metadata.get("dataset_version", data.get("dataset_version", "")),
        )
        for key, value in data.items():
            scenario[key] = value

        return scenario

    def __init__(self, scenario_id: str = "", dataset_name: str = "", dataset_version: str = ""):
        super().__init__()
        self["id"] = scenario_id
        self["dynamic_agents"] = {}
        self["static_map_elements"] = {}
        self["dynamic_map_elements"] = {}
        self["metadata"] = {
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "length": 0,
            "ego_id": "",
            "timesteps": [],
        }

    @property
    def export_file_name(self):
        """Return the file name of .pkl file of this scenario, if exported."""
        return f"{self['metadata']['dataset_name']}_{self['metadata']['dataset_version']}_{self['id']}"

    def get_dynamic_agents_by_type(self, agent_type: str) -> dict[str, dict]:
        """Get all dynamic agents of a specific type."""
        if not types.is_participant(agent_type):
            raise ValueError(f"Invalid dynamic agent type: {agent_type}")

        return {aid: agent for aid, agent in self["dynamic_agents"].items() if agent["type"] == agent_type}

    def get_static_map_elements_by_type(self, element_type: str) -> dict[str, dict]:
        """Get all static map elements of a specific type."""
        return {eid: element for eid, element in self["static_map_elements"].items() if element["type"] == element_type}

    def validate(self) -> bool:
        """Validate the scenario structure and data consistency."""
        # Check required top-level fields
        required_fields = ["id", "dynamic_agents", "static_map_elements", "dynamic_map_elements", "metadata"]
        for field in required_fields:
            if field not in self:
                raise ValueError(f"Missing required field: {field}")

        # Check required metadata fields
        metadata_fields = ["dataset_name", "dataset_version", "length", "timesteps"]
        for field in metadata_fields:
            if field not in self["metadata"]:
                raise ValueError(f"Missing required metadata field: {field}")

        # Check timesteps consistency
        length = self["metadata"]["length"]
        timesteps = self["metadata"]["timesteps"]
        if len(timesteps) != length:
            raise ValueError(f"Timesteps array length ({len(timesteps)}) doesn't match scenario length ({length})")

        # Check dynamic agents data consistency
        for aid, agent in self["dynamic_agents"].items():
            states = agent.get("states", {})

            if "position" in states and len(states["position"]) != length:
                raise ValueError(f"Dynamic agent {aid} position length mismatch: {len(states['position'])} != {length}")
            if "heading" in states and len(states["heading"]) != length:
                raise ValueError(f"Dynamic agent {aid} heading length mismatch: {len(states['heading'])} != {length}")
            if "valid" in states and len(states["valid"]) != length:
                raise ValueError(f"Dynamic agent {aid} valid mask length mismatch: {len(states['valid'])} != {length}")

        # Check dynamic map elements data consistency
        for dmid, dme in self["dynamic_map_elements"].items():
            if "states" in dme and len(dme["states"]) != length:
                raise ValueError(f"Dynamic map element {dmid} states length mismatch: {len(dme['states'])} != {length}")

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get basic statistics about the scenario."""
        stats = {
            "num_dynamic_agents": len(self["dynamic_agents"]),
            "num_static_map_elements": len(self["static_map_elements"]),
            "num_dynamic_map_elements": len(self["dynamic_map_elements"]),
            "duration_steps": self["length"],
            "dynamic_agent_types": {},
            "static_map_element_types": {},
        }

        # Count dynamic agent types
        for agent in self["dynamic_agents"].values():
            atype = agent["type"]
            stats["dynamic_agent_types"][atype] = stats["dynamic_agent_types"].get(atype, 0) + 1

        # Count static map element types
        for element in self["static_map_elements"].values():
            etype = element["type"]
            stats["static_map_element_types"][etype] = stats["static_map_element_types"].get(etype, 0) + 1

        return stats
