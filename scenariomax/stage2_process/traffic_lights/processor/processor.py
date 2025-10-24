import copy
from typing import Any

from scenariomax.stage2_process.traffic_lights.processor.waymonic_tlsgen import WaymonicTLSGenerator
from scenariomax.stage2_process.traffic_lights.processor.waymonizer import Waymonizer


class ScenarioProcessor(Waymonizer, WaymonicTLSGenerator):
    def __init__(self, scenario) -> None:
        self.scenario = scenario
        Waymonizer.__init__(self, scenario)
        WaymonicTLSGenerator.__init__(self, scenario, self.lanecenters, self.signalized_intersections)


def add_traffic_lights_to_scenario(unified_scenario: dict[str, Any]) -> dict[str, Any]:
    sp = ScenarioProcessor(unified_scenario)

    dynamic_map_states = sp.generate_waymonic_tls(return_data="dynamic_states")

    # after that, you can replace the old dynamic_map_states object with the new one
    scenario_copied = copy.deepcopy(unified_scenario)

    scenario_copied["dynamic_map_elements"] = {**dynamic_map_states}

    return scenario_copied
