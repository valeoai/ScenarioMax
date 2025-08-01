from dataclasses import dataclass, field

import numpy as np

from scenariomax.unified_to_tfexample import constants


@dataclass
class PathSamples:
    """
    A dataclass for storing and initializing path samples with given dimensions.

    Attributes:
        xyz (np.ndarray): 3D coordinates of points along the paths.
        valid (np.ndarray): Validity mask for points along the paths.
        id (np.ndarray): Identifier for points along the paths.
        arc_length (np.ndarray): Arc length of the points along the paths.
        on_route (np.ndarray): Route status for each path.
    """

    xyz: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_PATHS, constants.NUM_POINTS_PER_PATH, 3),
            -1.0,
            dtype=np.float64,
        ),
    )
    valid: np.ndarray = field(
        default_factory=lambda: np.full((constants.NUM_PATHS, constants.NUM_POINTS_PER_PATH), 0, dtype=np.int64),
    )
    id: np.ndarray = field(
        default_factory=lambda: np.full((constants.NUM_PATHS, constants.NUM_POINTS_PER_PATH), -1, dtype=np.int64),
    )
    arc_length: np.ndarray = field(
        default_factory=lambda: np.full((constants.NUM_PATHS, constants.NUM_POINTS_PER_PATH), 0.0, dtype=np.float64),
    )
    on_route: np.ndarray = field(default_factory=lambda: np.full((constants.NUM_PATHS,), 0, dtype=np.int64))


@dataclass
class MultiAgentPathSamples:
    """
    A dataclass for storing and initializing path samples for multiple objects.

    Attributes:
        path (np.ndarray): Path samples for multiple objects.
        path_valid (np.ndarray): Validity mask for path samples.
    """

    xyz: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_PATHS, constants.NUM_POINTS_PER_PATH, 3),
            -1.0,
            dtype=np.float64,
        ),
    )
    valid: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_PATHS, constants.NUM_POINTS_PER_PATH),
            0,
            dtype=np.int64,
        ),
    )
    id: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_PATHS, constants.NUM_POINTS_PER_PATH),
            -1,
            dtype=np.int64,
        ),
    )
    arc_length: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_PATHS, constants.NUM_POINTS_PER_PATH),
            0.0,
            dtype=np.float64,
        ),
    )
    on_route: np.ndarray = field(
        default_factory=lambda: np.full(
            (
                constants.DEFAULT_NUM_OBJECTS,
                constants.NUM_PATHS,
            ),
            0,
            dtype=np.int64,
        ),
    )

    def add_data(self, idx: int, path_samples: PathSamples) -> None:
        """
        Adds path samples for a specific object index.

        Args:
            idx (int): The index of the object.
            path_samples (MultiAgentPathSamples): The path samples to add.
        """
        self.xyz[idx] = path_samples.xyz
        self.valid[idx] = path_samples.valid
        self.id[idx] = path_samples.id
        self.arc_length[idx] = path_samples.arc_length
        self.on_route[idx] = path_samples.on_route


@dataclass
class RoadGraphSamples:
    """
    A dataclass for storing and initializing road graph samples with given dimensions.

    Attributes:
        xyz (np.ndarray): 3D coordinates of points in the road graph.
        valid (np.ndarray): Validity mask for the road graph points.
        id (np.ndarray): Identifiers for road graph points.
        dir (np.ndarray): Directions for road graph points.
        type (np.ndarray): Types for road graph points.
    """

    xyz: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_ROADMAPS, 3), -1.0, dtype=np.float64),
    )
    valid: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_ROADMAPS,), 0, dtype=np.int64),
    )
    id: np.ndarray = field(default_factory=lambda: np.full((constants.DEFAULT_NUM_ROADMAPS,), -1, dtype=np.int64))
    dir: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_ROADMAPS, 3), -1.0, dtype=np.float64),
    )
    speed_limit: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_ROADMAPS,), -1.0, dtype=np.float64),
    )
    type: np.ndarray = field(default_factory=lambda: np.full((constants.DEFAULT_NUM_ROADMAPS,), -1, dtype=np.int64))


@dataclass
class State:
    """
    A dataclass for storing and initializing the state of multiple objects over time.

    Attributes:
        id (np.ndarray): Identifiers for objects.
        type (np.ndarray): Types of objects.
        is_sdc (np.ndarray): Flags indicating if an object is the self-driving car (SDC).
        tracks_to_predict (np.ndarray): Flags indicating if an object's track should be predicted.
        objects_of_interest (np.ndarray): Flags indicating objects of interest.

        current_x (np.ndarray): Current x-coordinates of objects.
        current_y (np.ndarray): Current y-coordinates of objects.
        current_z (np.ndarray): Current z-coordinates of objects.
        current_bbox_yaw (np.ndarray): Current yaw of bounding boxes of objects.
        current_length (np.ndarray): Current length of objects.
        current_width (np.ndarray): Current width of objects.
        current_height (np.ndarray): Current height of objects.
        current_speed (np.ndarray): Current speed of objects.
        current_timestamps (np.ndarray): Current timestamps for objects.
        current_vel_yaw (np.ndarray): Current velocity yaw of objects.
        current_velocity_x (np.ndarray): Current x-component of velocity of objects.
        current_velocity_y (np.ndarray): Current y-component of velocity of objects.
        current_valid (np.ndarray): Flags indicating if current data is valid.

        past_x (np.ndarray): Past x-coordinates of objects.
        past_y (np.ndarray): Past y-coordinates of objects.
        past_z (np.ndarray): Past z-coordinates of objects.
        past_bbox_yaw (np.ndarray): Past yaw of bounding boxes of objects.
        past_length (np.ndarray): Past length of objects.
        past_width (np.ndarray): Past width of objects.
        past_height (np.ndarray): Past height of objects.
        past_speed (np.ndarray): Past speed of objects.
        past_timestamps (np.ndarray): Past timestamps for objects.
        past_vel_yaw (np.ndarray): Past velocity yaw of objects.
        past_velocity_x (np.ndarray): Past x-component of velocity of objects.
        past_velocity_y (np.ndarray): Past y-component of velocity of objects.
        past_valid (np.ndarray): Flags indicating if past data is valid.

        future_x (np.ndarray): Future x-coordinates of objects.
        future_y (np.ndarray): Future y-coordinates of objects.
        future_z (np.ndarray): Future z-coordinates of objects.
        future_bbox_yaw (np.ndarray): Future yaw of bounding boxes of objects.
        future_length (np.ndarray): Future length of objects.
        future_width (np.ndarray): Future width of objects.
        future_height (np.ndarray): Future height of objects.
        future_speed (np.ndarray): Future speed of objects.
        future_timestamps (np.ndarray): Future timestamps for objects.
        future_vel_yaw (np.ndarray): Future velocity yaw of objects.
        future_velocity_x (np.ndarray): Future x-component of velocity of objects.
        future_velocity_y (np.ndarray): Future y-component of velocity of objects.
        future_valid (np.ndarray): Flags indicating if future data is valid.
    """

    id: np.ndarray = field(default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1, dtype=np.int64))
    type: np.ndarray = field(default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), 0, dtype=np.int64))
    is_sdc: np.ndarray = field(default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), 0, dtype=np.int64))
    tracks_to_predict: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), 0, dtype=np.int64),
    )
    objects_of_interest: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), 0, dtype=np.int64),
    )

    current_x: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1.0, dtype=np.float64),
    )
    current_y: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1.0, dtype=np.float64),
    )
    current_z: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1.0, dtype=np.float64),
    )

    current_bbox_yaw: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1.0, dtype=np.float64),
    )
    current_length: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1.0, dtype=np.float64),
    )
    current_width: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1.0, dtype=np.float64),
    )
    current_height: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1.0, dtype=np.float64),
    )
    current_speed: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1.0, dtype=np.float64),
    )
    current_timestamps: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1, dtype=np.int64),
    )
    current_vel_yaw: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1.0, dtype=np.float64),
    )
    current_velocity_x: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1.0, dtype=np.float64),
    )
    current_velocity_y: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), -1.0, dtype=np.float64),
    )
    current_valid: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS,), 0, dtype=np.int64),
    )

    past_x: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1.0, dtype=np.float64),
    )
    past_y: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1.0, dtype=np.float64),
    )
    past_z: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1.0, dtype=np.float64),
    )

    past_bbox_yaw: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1.0, dtype=np.float64),
    )
    past_length: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1.0, dtype=np.float64),
    )
    past_width: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1.0, dtype=np.float64),
    )
    past_height: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1.0, dtype=np.float64),
    )
    past_speed: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1.0, dtype=np.float64),
    )
    past_timestamps: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1, dtype=np.int64),
    )
    past_vel_yaw: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1.0, dtype=np.float64),
    )
    past_velocity_x: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1.0, dtype=np.float64),
    )
    past_velocity_y: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), -1.0, dtype=np.float64),
    )
    past_valid: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_PAST), 0, dtype=np.int64),
    )

    future_x: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_y: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_z: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE),
            -1.0,
            dtype=np.float64,
        ),
    )

    future_bbox_yaw: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_length: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_width: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_height: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_speed: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_timestamps: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE), -1, dtype=np.int64),
    )
    future_vel_yaw: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_velocity_x: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_velocity_y: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_valid: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_OBJECTS, constants.NUM_TS_FUTURE), 0, dtype=np.int64),
    )

    @property
    def num_objects(self) -> int:
        """Returns the number of objects."""
        return self.id.shape[0]


@dataclass
class TrafficLightState:
    """
    A dataclass for storing and initializing the state of traffic lights over time.

    Attributes:
        current_state (np.ndarray): Current states of traffic lights.
        current_x (np.ndarray): Current x-coordinates of traffic lights.
        current_y (np.ndarray): Current y-coordinates of traffic lights.
        current_z (np.ndarray): Current z-coordinates of traffic lights.
        current_id (np.ndarray): Identifiers for traffic lights.
        current_valid (np.ndarray): Validity mask for the current traffic light data.
        current_timestamps (np.ndarray): Timestamps for the current traffic light data.

        past_state (np.ndarray): Past states of traffic lights.
        past_x (np.ndarray): Past x-coordinates of traffic lights.
        past_y (np.ndarray): Past y-coordinates of traffic lights.
        past_z (np.ndarray): Past z-coordinates of traffic lights.
        past_id (np.ndarray): Identifiers for past traffic lights.
        past_valid (np.ndarray): Validity mask for the past traffic light data.
        past_timestamps (np.ndarray): Timestamps for the past traffic light data.

        future_state (np.ndarray): Future states of traffic lights.
        future_x (np.ndarray): Future x-coordinates of traffic lights.
        future_y (np.ndarray): Future y-coordinates of traffic lights.
        future_z (np.ndarray): Future z-coordinates of traffic lights.
        future_id (np.ndarray): Identifiers for future traffic lights.
        future_valid (np.ndarray): Validity mask for the future traffic light data.
        future_timestamps (np.ndarray): Timestamps for the future traffic light data.
    """

    current_state: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_LIGHT_POSITIONS,), -1, dtype=np.int64),
    )
    current_x: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_LIGHT_POSITIONS,), -1.0, dtype=np.float64),
    )
    current_y: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_LIGHT_POSITIONS,), -1.0, dtype=np.float64),
    )
    current_z: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_LIGHT_POSITIONS,), -1.0, dtype=np.float64),
    )
    current_id: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_LIGHT_POSITIONS,), -1, dtype=np.int64),
    )
    current_valid: np.ndarray = field(
        default_factory=lambda: np.full((constants.DEFAULT_NUM_LIGHT_POSITIONS,), 0, dtype=np.int64),
    )
    current_timestamps: np.ndarray = field(default_factory=lambda: np.full((1,), 0, dtype=np.int64))

    past_state: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_PAST, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            -1,
            dtype=np.int64,
        ),
    )
    past_x: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_PAST, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            -1.0,
            dtype=np.float64,
        ),
    )
    past_y: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_PAST, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            -1.0,
            dtype=np.float64,
        ),
    )
    past_z: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_PAST, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            -1.0,
            dtype=np.float64,
        ),
    )
    past_id: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_PAST, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            -1,
            dtype=np.int64,
        ),
    )
    past_valid: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_PAST, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            0,
            dtype=np.int64,
        ),
    )
    past_timestamps: np.ndarray = field(default_factory=lambda: np.full((constants.NUM_TS_PAST,), 0, dtype=np.int64))

    future_state: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_FUTURE, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            -1,
            dtype=np.int64,
        ),
    )
    future_x: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_FUTURE, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_y: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_FUTURE, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_z: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_FUTURE, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            -1.0,
            dtype=np.float64,
        ),
    )
    future_id: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_FUTURE, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            -1,
            dtype=np.int64,
        ),
    )
    future_valid: np.ndarray = field(
        default_factory=lambda: np.full(
            (constants.NUM_TS_FUTURE, constants.DEFAULT_NUM_LIGHT_POSITIONS),
            0,
            dtype=np.int64,
        ),
    )
    future_timestamps: np.ndarray = field(
        default_factory=lambda: np.full((constants.NUM_TS_FUTURE,), 0, dtype=np.int64),
    )
