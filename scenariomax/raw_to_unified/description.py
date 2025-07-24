import math
import os
from collections import defaultdict

import numpy as np

from scenariomax.raw_to_unified.type import ScenarioType


class ScenarioDescription(dict):
    """
    Scenario Description. It stores keys of the data dict.
    """

    TRACKS = "tracks"
    VERSION = "version"
    ID = "id"
    DYNAMIC_MAP_STATES = "dynamic_map_states"
    MAP_FEATURES = "map_features"
    LENGTH = "length"
    METADATA = "metadata"
    FIRST_LEVEL_KEYS = {TRACKS, VERSION, ID, DYNAMIC_MAP_STATES, MAP_FEATURES, LENGTH, METADATA}

    # lane keys
    POLYLINE = "polyline"
    POLYGON = "polygon"
    LEFT_BOUNDARIES = "left_boundaries"
    RIGHT_BOUNDARIES = "right_boundaries"
    LEFT_NEIGHBORS = "left_neighbor"
    RIGHT_NEIGHBORS = "right_neighbor"
    ENTRY = "entry_lanes"
    EXIT = "exit_lanes"

    # object
    TYPE = "type"
    STATE = "state"
    OBJECT_ID = "object_id"
    STATE_DICT_KEYS = {TYPE, STATE, METADATA}
    ORIGINAL_ID_TO_OBJ_ID = "original_id_to_obj_id"
    OBJ_ID_TO_ORIGINAL_ID = "obj_id_to_original_id"
    TRAFFIC_LIGHT_POSITION = "stop_point"
    TRAFFIC_LIGHT_STATUS = "object_state"
    TRAFFIC_LIGHT_LANE = "lane"
    #  for object position/heading
    POSITION = "position"
    HEADING = "heading"

    TIMESTEP = "ts"
    SDC_ID = "sdc_id"  # Not necessary, but can be stored in metadata.
    METADATA_KEYS = {TIMESTEP}

    ALLOW_TYPES = (int, float, str, np.ndarray, dict, list, tuple, type(None), set)

    class SUMMARY:
        OBJECT_SUMMARY = "object_summary"
        NUMBER_SUMMARY = "number_summary"

        # for each object summary
        TYPE = "type"
        OBJECT_ID = "object_id"
        TRACK_LENGTH = "track_length"
        MOVING_DIST = "moving_distance"
        VALID_LENGTH = "valid_length"
        CONTINUOUS_VALID_LENGTH = "continuous_valid_length"

        # for number summary:
        OBJECT_TYPES = "object_types"
        NUM_OBJECTS = "num_objects"
        NUM_MOVING_OBJECTS = "num_moving_objects"
        NUM_OBJECTS_EACH_TYPE = "num_objects_each_type"
        NUM_MOVING_OBJECTS_EACH_TYPE = "num_moving_objects_each_type"

        NUM_TRAFFIC_LIGHTS = "num_traffic_lights"
        NUM_TRAFFIC_LIGHT_TYPES = "num_traffic_light_types"
        NUM_TRAFFIC_LIGHTS_EACH_STEP = "num_traffic_light_each_step"

        NUM_MAP_FEATURES = "num_map_features"
        MAP_HEIGHT_DIFF = "map_height_diff"

    class DATASET:
        SUMMARY_FILE = "dataset_summary.pkl"  # dataset summary file name
        MAPPING_FILE = "dataset_mapping.pkl"  # store the relative path of summary file and each scenario

    @classmethod
    def sanity_check(cls, scenario_dict, check_self_type=False, valid_check=False):
        """Check if the input scenario dict is self-consistent and has filled required fields.

        The required high-level fields include tracks, dynamic_map_states, metadata, map_features.
        For each object, the tracks[obj_id] should at least contain type, state, metadata.
        For each object, the tracks[obj_id]['state'] should at least contain position, heading.
        For each lane in map_features, map_feature[map_feat_id] should at least contain polyline.
        We have more checks to ensure the consistency of the data.

        Args:
            scenario_dict: the input dict.
            check_self_type: if True, assert the input dict is a native Python dict.
            valid_check: if True, we will assert the values for a given timestep are zeros if valid=False at that
                timestep.
        """
        if check_self_type:
            assert isinstance(scenario_dict, dict)
            assert not isinstance(scenario_dict, ScenarioDescription)

        # Whether input has all required keys
        assert cls.FIRST_LEVEL_KEYS.issubset(set(scenario_dict.keys())), (
            f"You lack these keys in first level: {cls.FIRST_LEVEL_KEYS.difference(set(scenario_dict.keys()))}"
        )

        # Check types, only native python objects
        # This is to avoid issue in pickle deserialization
        _recursive_check_type(scenario_dict, cls.ALLOW_TYPES)

        scenario_length = scenario_dict[cls.LENGTH]

        # Check tracks data
        assert isinstance(scenario_dict[cls.TRACKS], dict)
        for obj_id, obj_state in scenario_dict[cls.TRACKS].items():
            cls._check_object_state_dict(
                obj_state,
                scenario_length=scenario_length,
                object_id=obj_id,
                valid_check=valid_check,
            )
            # position heading check
            assert ScenarioDescription.HEADING in obj_state[ScenarioDescription.STATE], (
                "heading is required for an object"
            )
            assert ScenarioDescription.POSITION in obj_state[ScenarioDescription.STATE], (
                "position is required for an object"
            )

        # Check dynamic_map_state
        assert isinstance(scenario_dict[cls.DYNAMIC_MAP_STATES], dict)
        for obj_id, obj_state in scenario_dict[cls.DYNAMIC_MAP_STATES].items():
            cls._check_object_state_dict(obj_state, scenario_length=scenario_length, object_id=obj_id)

        # Check map features
        assert isinstance(scenario_dict[cls.MAP_FEATURES], dict)
        cls._check_map_features(scenario_dict[cls.MAP_FEATURES])

        # Check metadata
        assert isinstance(scenario_dict[cls.METADATA], dict)
        assert cls.METADATA_KEYS.issubset(set(scenario_dict[cls.METADATA].keys())), (
            f"You lack these keys in metadata: {cls.METADATA_KEYS.difference(set(scenario_dict[cls.METADATA].keys()))}"
        )
        assert np.asarray(scenario_dict[cls.METADATA][cls.TIMESTEP]).shape == (scenario_length,)

    @classmethod
    def _check_map_features(cls, map_feature):
        """Check if all lanes in the map contain the polyline (center line) feature and if they are in correct types."""
        for id, feature in map_feature.items():
            if ScenarioType.is_lane(feature[ScenarioDescription.TYPE]):
                assert ScenarioDescription.POLYLINE in feature, "No lane center line in map feature"
                assert isinstance(feature[ScenarioDescription.POLYLINE], np.ndarray | list | tuple), (
                    "lane center line is in invalid type"
                )

    @classmethod
    def _check_object_state_dict(cls, obj_state, scenario_length, object_id, valid_check=True):
        """Check the state dict of an object (the dynamic objects such as road users, vehicles or traffic lights).

        Args:
            obj_state: the state dict of the object.
            scenario_length: the length (# of timesteps) of the scenario.
            object_id: the ID of the object.
            valid_check: if True, we will examine the data at each timestep and see if it's non-zero when valid=False
                at that timestep.
        """
        # Check keys
        assert set(obj_state).issuperset(cls.STATE_DICT_KEYS)

        # Check type
        assert ScenarioType.has_type(obj_state[cls.TYPE]), f"scenarionet doesn't have this type: {obj_state[cls.TYPE]}"

        # Check set type
        assert obj_state[cls.TYPE] != ScenarioType.UNSET, "Types should be set for objects and traffic lights"

        # Check state arrays temporal consistency
        assert isinstance(obj_state[cls.STATE], dict)
        for state_key, state_array in obj_state[cls.STATE].items():
            assert isinstance(state_array, np.ndarray | list | tuple)
            assert len(state_array) == scenario_length

            if not isinstance(state_array, np.ndarray):
                continue

            assert state_array.ndim in [1, 2], f"Haven't implemented test array with dim {state_array.ndim} yet"
            if state_array.ndim == 2:
                assert state_array.shape[1] != 0, (
                    "Please convert all state with dim 1 to a 1D array instead of 2D array."
                )

            if state_key == "valid" and valid_check:
                assert np.sum(state_array) >= 1, "No frame valid for this object. Consider removing it"

            # check valid
            if "valid" in obj_state[cls.STATE] and valid_check:
                _array = state_array[..., :2] if state_key == "position" else state_array
                assert abs(np.sum(_array[np.where(obj_state[cls.STATE]["valid"], False, True)])) < 1e-2, (
                    f"Valid array mismatches with {state_key} array, some frames in {state_key} have non-zero values, "
                    "so it might be valid"
                )

        # Check metadata
        assert isinstance(obj_state[cls.METADATA], dict)
        for metadata_key in (cls.TYPE, cls.OBJECT_ID):
            assert metadata_key in obj_state[cls.METADATA]

        # Check metadata alignment
        if cls.OBJECT_ID in obj_state[cls.METADATA]:
            assert obj_state[cls.METADATA][cls.OBJECT_ID] == object_id

    def to_dict(self):
        """Convert the object to a native python dict.

        Returns:
            A python dict
        """
        return dict(self)

    def get_sdc_track(self):
        """Return the object info dict for the SDC.

        Returns:
            The info dict for the SDC.
        """
        assert self.SDC_ID in self[self.METADATA]
        sdc_id = str(self[self.METADATA][self.SDC_ID])
        return self[self.TRACKS][sdc_id]

    @staticmethod
    def get_object_summary(object_dict, object_id: str):
        """Summarize the information of one dynamic object.

        Args:
            object_dict: the info dict of a particular object, aka scenario['tracks'][obj_id] (not the ['state'] dict!)
            object_id: the ID of the object

        Returns:
            A dict summarizing the information of this object.
        """
        object_type = object_dict["type"]
        state_dict = object_dict["state"]
        track = state_dict["position"]
        valid_track = track[np.where(state_dict["valid"].astype(int))][..., :2]
        distance = float(
            sum(np.linalg.norm(valid_track[i] - valid_track[i + 1]) for i in range(valid_track.shape[0] - 1)),
        )
        valid_length = int(sum(state_dict["valid"]))

        continuous_valid_length = 0
        for v in state_dict["valid"]:
            if v:
                continuous_valid_length += 1
            if continuous_valid_length > 0 and not v:
                break

        return {
            ScenarioDescription.SUMMARY.TYPE: object_type,
            ScenarioDescription.SUMMARY.OBJECT_ID: str(object_id),
            ScenarioDescription.SUMMARY.TRACK_LENGTH: len(track),
            ScenarioDescription.SUMMARY.MOVING_DIST: float(distance),
            ScenarioDescription.SUMMARY.VALID_LENGTH: int(valid_length),
            ScenarioDescription.SUMMARY.CONTINUOUS_VALID_LENGTH: int(continuous_valid_length),
        }

    @staticmethod
    def get_export_file_name(dataset: str, dataset_version: str, scenario_name: str):
        """Return the file name of .pkl file of this scenario, if exported."""
        return f"sd_{dataset}_{dataset_version}_{scenario_name}.pkl"

    @staticmethod
    def is_scenario_file(file_name: str):
        """Verify if the scenario file is valid.

        Args:
            file_name: The path to the .pkl file.

        Returns:
            A Boolean.
        """
        file_name = os.path.basename(file_name)
        if not file_name.endswith(".pkl"):
            return False
        file_name = file_name.replace(".pkl", "")
        return os.path.basename(file_name)[:3] == "sd_" or all(char.isdigit() for char in file_name)

    @staticmethod
    def calculate_num_moving_objects(scenario):
        """Calculate the number of moving objects, whose moving distance > 1m in this scenario."""
        # moving object
        number_summary_dict = {
            ScenarioDescription.SUMMARY.NUM_MOVING_OBJECTS: 0,
            ScenarioDescription.SUMMARY.NUM_MOVING_OBJECTS_EACH_TYPE: defaultdict(int),
        }
        for v in scenario[ScenarioDescription.METADATA][ScenarioDescription.SUMMARY.OBJECT_SUMMARY].values():
            # Fix a tiny compatibility issue
            if ScenarioDescription.SUMMARY.MOVING_DIST not in v:
                v[ScenarioDescription.SUMMARY.MOVING_DIST] = v["distance"]

            if v[ScenarioDescription.SUMMARY.MOVING_DIST] > 1:
                number_summary_dict[ScenarioDescription.SUMMARY.NUM_MOVING_OBJECTS] += 1
                number_summary_dict[ScenarioDescription.SUMMARY.NUM_MOVING_OBJECTS_EACH_TYPE][v["type"]] += 1
        return number_summary_dict

    @staticmethod
    def update_summaries(scenario):
        """Update the object summary and number summary of one scenario in-place.

        Args:
            scenario: The input scenario

        Returns:
            The same scenario with the scenario['metadata']['object/number_summary'] be overwritten.
        """
        SD = ScenarioDescription

        # add agents summary
        summary_dict = {}
        for track_id, track in scenario[SD.TRACKS].items():
            summary_dict[track_id] = SD.get_object_summary(object_dict=track, object_id=track_id)
        scenario[SD.METADATA][SD.SUMMARY.OBJECT_SUMMARY] = summary_dict

        # count some objects occurrence
        scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY] = SD.get_number_summary(scenario)
        return scenario

    @staticmethod
    def get_number_summary(scenario):
        """Return the stats of all objects in a scenario.

        Examples:
            {'num_objects': 211,
             'object_types': {'CYCLIST', 'PEDESTRIAN', 'VEHICLE'},
             'num_objects_each_type': {'VEHICLE': 184, 'PEDESTRIAN': 25, 'CYCLIST': 2},
             'num_moving_objects': 69,
             'num_moving_objects_each_type': defaultdict(int, {'VEHICLE': 52, 'PEDESTRIAN': 15, 'CYCLIST': 2}),
             'num_traffic_lights': 8,
             'num_traffic_light_types': {'LANE_STATE_STOP', 'LANE_STATE_UNKNOWN'},
             'num_traffic_light_each_step': {'LANE_STATE_UNKNOWN': 164, 'LANE_STATE_STOP': 564},
             'num_map_features': 358,
             'map_height_diff': 2.4652252197265625}

        Args:
            scenario: The input scenario.

        Returns:
            A dict describing the number of different kinds of data.
        """
        number_summary_dict = {}

        # object
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_OBJECTS] = len(scenario[ScenarioDescription.TRACKS])
        number_summary_dict[ScenarioDescription.SUMMARY.OBJECT_TYPES] = {
            v["type"] for v in scenario[ScenarioDescription.TRACKS].values()
        }
        object_types_counter = defaultdict(int)
        for v in scenario[ScenarioDescription.TRACKS].values():
            object_types_counter[v["type"]] += 1
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_OBJECTS_EACH_TYPE] = dict(object_types_counter)

        # If object summary does not exist, fill them here
        object_summaries = {}
        for track_id, track in scenario[scenario.TRACKS].items():
            object_summaries[track_id] = scenario.get_object_summary(object_dict=track, object_id=track_id)
        scenario[scenario.METADATA][scenario.SUMMARY.OBJECT_SUMMARY] = object_summaries

        # moving object
        number_summary_dict.update(ScenarioDescription.calculate_num_moving_objects(scenario))

        # Number of different dynamic object states
        dynamic_object_states_types = set()
        dynamic_object_states_counter = defaultdict(int)
        for v in scenario[ScenarioDescription.DYNAMIC_MAP_STATES].values():
            for step_state in v["state"]["object_state"]:
                if step_state is None:
                    continue
                dynamic_object_states_types.add(step_state)
                dynamic_object_states_counter[step_state] += 1
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHTS] = len(
            scenario[ScenarioDescription.DYNAMIC_MAP_STATES],
        )
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHT_TYPES] = dynamic_object_states_types
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHTS_EACH_STEP] = dict(
            dynamic_object_states_counter,
        )

        # map
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_MAP_FEATURES] = len(
            scenario[ScenarioDescription.MAP_FEATURES],
        )
        number_summary_dict[ScenarioDescription.SUMMARY.MAP_HEIGHT_DIFF] = ScenarioDescription.map_height_diff(
            scenario[ScenarioDescription.MAP_FEATURES],
        )
        return number_summary_dict

    @staticmethod
    def sdc_moving_dist(scenario):
        """Get the moving distance of SDC in this scenario. This is useful to filter the scenario.

        Args:
            scenario: The scenario description.

        Returns:
            (float) The moving distance of SDC.
        """
        scenario = ScenarioDescription(scenario)

        SD = ScenarioDescription
        metadata = scenario[SD.METADATA]

        sdc_id = metadata[SD.SDC_ID]
        sdc_info = metadata[SD.SUMMARY.OBJECT_SUMMARY][sdc_id]

        if SD.SUMMARY.MOVING_DIST not in sdc_info:
            sdc_info = SD.get_object_summary(object_dict=scenario.get_sdc_track(), object_id=sdc_id)

        moving_dist = sdc_info[SD.SUMMARY.MOVING_DIST]
        return moving_dist

    @staticmethod
    def get_num_objects(scenario, object_type: str | None = None):
        """Return the number of objects (vehicles, pedestrians, cyclists, ...).

        Args:
            scenario: The input scenario.
            object_type: The string of the object type. If None, return the number of all objects.

        Returns:
            (int) The number of objects.
        """
        SD = ScenarioDescription
        metadata = scenario[SD.METADATA]
        if SD.SUMMARY.NUMBER_SUMMARY not in metadata:
            scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY] = SD.get_number_summary(scenario)
        if object_type is None:
            return metadata[SD.SUMMARY.NUMBER_SUMMARY][SD.SUMMARY.NUM_OBJECTS]
        else:
            return metadata[SD.SUMMARY.NUMBER_SUMMARY][SD.SUMMARY.NUM_OBJECTS_EACH_TYPE].get(object_type, 0)

    @staticmethod
    def num_object(scenario, object_type: str | None = None):
        """Return the number of objects (vehicles, pedestrians, cyclists, ...).

        Args:
            scenario: The input scenario.
            object_type: The string of the object type. If None, return the number of all objects.

        Returns:
            (int) The number of objects.
        """
        return ScenarioDescription.get_num_objects(scenario, object_type)

    @staticmethod
    def get_num_moving_objects(scenario, object_type=None):
        """Return the number of moving objects (vehicles, pedestrians, cyclists, ...).

        Args:
            scenario: The input scenario.
            object_type: The string of the object type. If None, return the number of all objects.

        Returns:
            (int) The number of moving objects.
        """
        SD = ScenarioDescription
        metadata = scenario[SD.METADATA]
        if SD.SUMMARY.NUM_MOVING_OBJECTS not in metadata[SD.SUMMARY.NUMBER_SUMMARY]:
            num_summary = SD.calculate_num_moving_objects(scenario)
        else:
            num_summary = metadata[SD.SUMMARY.NUMBER_SUMMARY]

        if object_type is None:
            return num_summary[SD.SUMMARY.NUM_MOVING_OBJECTS]
        else:
            return num_summary[SD.SUMMARY.NUM_MOVING_OBJECTS_EACH_TYPE].get(object_type, 0)

    @staticmethod
    def num_moving_object(scenario, object_type=None):
        """Return the number of moving objects (vehicles, pedestrians, cyclists, ...).

        Args:
            scenario: The input scenario.
            object_type: The string of the object type. If None, return the number of all objects.

        Returns:
            (int) The number of moving objects.
        """
        return ScenarioDescription.get_num_moving_objects(scenario, object_type=object_type)

    @staticmethod
    def map_height_diff(map_features, target=10):
        """Compute the height difference in a map.

        Args:
            map_features: The map feature dict of a scenario.
            target: The target height difference, default to 10. If we find height difference > 10, we will return 10
                immediately. This can be used to accelerate computing if we are filtering a batch of scenarios.

        Returns:
            (float) The height difference in the map feature, or the target height difference if the diff > target.
        """
        max = -math.inf
        min = math.inf
        for feature in map_features.values():
            if not ScenarioType.is_road_line(feature[ScenarioDescription.TYPE]):
                continue
            polyline = feature[ScenarioDescription.POLYLINE]
            if len(polyline[0]) == 3:
                z = np.asarray(polyline)[..., -1]
                z_max = np.max(z)
                if z_max > max:
                    max = z_max
                z_min = np.min(z)
                if z_min < min:
                    min = z_min
            if max - min > target:
                break
        return float(max - min)


def _recursive_check_type(obj, allow_types, depth=0):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), "Must use string to be dict keys"
            _recursive_check_type(v, allow_types, depth=depth + 1)

    if isinstance(obj, list):
        for v in obj:
            _recursive_check_type(v, allow_types, depth=depth + 1)

    assert isinstance(obj, allow_types), f"Object type {type(obj)} not allowed! ({allow_types})"

    if depth > 1000:
        raise ValueError()
