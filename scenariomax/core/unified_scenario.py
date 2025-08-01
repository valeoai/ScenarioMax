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
            "obj_id": {
                "type": str,          # From types.PARTICIPANT_TYPES
                "states": {      # State data for each timestep
                    "position": np.ndarray,    # (length, 3) - x, y
                    "heading": np.ndarray,     # (length,) - heading angle
                    "velocity": np.ndarray,    # (length, 2) - vx, vy
                    "valid": np.ndarray,       # (length,) - boolean validity mask
                },
                "length": float,           # Object length (optional)
                "width": float,            # Object width (optional)
                "height": float,           # Object height (optional)
            }
        },

        "static_map_elements": {      # Static map features
            "element_id": {
                "type": str,          # From types.LANE_TYPES, ROAD_LINE_TYPES, etc.
                "geometry": np.ndarray,    # (N, 3) - polyline points
                "polyline": np.ndarray,    # (N, 3) - polyline points for line/lane
                "speed_limit_mph": _lane.speed_limit_mph, # Speed limit in mph # lane only
                "speed_limit_kmh": converter_utils.mph_to_kmh(_lane.speed_limit_mph), # Speed limit in km/h # lane only
                "entry_lanes": list(_lane.entry_lanes),         # List of entry lane IDs # lane only
                "exit_lanes": list(_lane.exit_lanes),           # List of exit lane IDs # lane only
                "left_boundaries": _extract_boundaries(_lane.left_boundaries), # Extracted left boundaries
                "right_boundaries": _extract_boundaries(_lane.right_boundaries),
                "left_neighbor": _extract_neighbors(_lane.left_neighbors),
                "right_neighbor": _extract_neighbors(_lane.right_neighbors),
            }
        },

        "dynamic_map_elements": {     # Dynamic traffic light states
            "element_id": {
                "type": str,          # From types.TRAFFIC_LIGHT_TYPES
                "position": np.ndarray,    # (3,) - x, y, z position
                "states": list,            # (length,) - state at each timestep
                "controlled_lanes": list,  # List of lane IDs controlled by this light
            }
        },

        "metadata": {                 # Additional scenario information
            "dataset_name": str,          # Source dataset name
            "dataset_version": str,       # Dataset version
            "length": int,                # Number of timesteps
            "timesteps": np.ndarray,      # Timestamp array (length,)
            "ego_id": str,                # Self-driving car ID
        }
    }
    """

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
        # Check required fields
        required_fields = [
            "id",
            "version",
            "length",
            "timesteps",
            "dynamic_agents",
            "static_map_elements",
            "dynamic_map_elements",
            "metadata",
        ]
        for field in required_fields:
            if field not in self:
                raise ValueError(f"Missing required field: {field}")

        # Check timesteps consistency
        if len(self["timesteps"]) != self["length"]:
            raise ValueError("Timesteps array length doesn't match scenario length")

        # Check dynamic agents data consistency
        for aid, agent in self["dynamic_agents"].items():
            if len(agent["position"]) != self["length"]:
                raise ValueError(f"Dynamic agent {aid} position length mismatch")
            if len(agent["heading"]) != self["length"]:
                raise ValueError(f"Dynamic agent {aid} heading length mismatch")
            if len(agent["valid"]) != self["length"]:
                raise ValueError(f"Dynamic agent {aid} valid mask length mismatch")

        # Check dynamic map elements data consistency
        for dmid, dme in self["dynamic_map_elements"].items():
            if len(dme["states"]) != self["length"]:
                raise ValueError(f"Dynamic map element {dmid} states length mismatch")

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
