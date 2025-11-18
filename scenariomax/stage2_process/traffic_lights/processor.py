from typing import Any

from scenariomax.stage2_process.traffic_lights.waymonic_tlsgen import WaymonicTLSGenerator
from scenariomax.stage2_process.traffic_lights.waymonizer import Waymonizer


class ScenarioProcessor(Waymonizer, WaymonicTLSGenerator):
    def __init__(self, scenario) -> None:
        self.scenario = scenario
        Waymonizer.__init__(self, scenario)
        WaymonicTLSGenerator.__init__(self, scenario, self.lanecenters, self.signalized_intersections)


def add_traffic_lights_to_scenario(unified_scenario: dict[str, Any]) -> dict[str, Any]:
    # Validate required fields
    if "static_map_elements" not in unified_scenario:
        raise ValueError("Scenario missing 'static_map_elements' - cannot generate traffic lights")
    if "metadata" not in unified_scenario or "scenario_length" not in unified_scenario["metadata"]:
        raise ValueError("Scenario missing 'metadata.scenario_length' - cannot generate traffic lights")
    if "dynamic_agents" not in unified_scenario:
        raise ValueError("Scenario missing 'dynamic_agents' - cannot infer traffic light states")

    sp = ScenarioProcessor(unified_scenario)

    dynamic_map_states = sp.generate_waymonic_tls(
        return_data="dynamic_states",
        end_step=unified_scenario["metadata"]["length"],
    )

    # Replace dynamic_map_elements with generated traffic lights
    unified_scenario["dynamic_map_elements"] = {**dynamic_map_states}

    return unified_scenario
