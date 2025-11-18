# Simplified and organized types for ScenarioMax unified format

# ===== Participant Types =====
VEHICLE = "VEHICLE"
PEDESTRIAN = "PEDESTRIAN"
CYCLIST = "CYCLIST"
OTHER = "OTHER"

# ===== Lane Types =====
LANE_SURFACE_STREET = "LANE_SURFACE_STREET"
LANE_FREEWAY = "LANE_FREEWAY"
LANE_BIKE_LANE = "LANE_BIKE_LANE"
LANE_UNKNOWN = "LANE_UNKNOWN"

# ===== Road Line Types =====
ROAD_LINE_BROKEN_SINGLE_WHITE = "ROAD_LINE_BROKEN_SINGLE_WHITE"
ROAD_LINE_SOLID_SINGLE_WHITE = "ROAD_LINE_SOLID_SINGLE_WHITE"
ROAD_LINE_SOLID_DOUBLE_WHITE = "ROAD_LINE_SOLID_DOUBLE_WHITE"
ROAD_LINE_BROKEN_SINGLE_YELLOW = "ROAD_LINE_BROKEN_SINGLE_YELLOW"
ROAD_LINE_BROKEN_DOUBLE_YELLOW = "ROAD_LINE_BROKEN_DOUBLE_YELLOW"
ROAD_LINE_SOLID_SINGLE_YELLOW = "ROAD_LINE_SOLID_SINGLE_YELLOW"
ROAD_LINE_SOLID_DOUBLE_YELLOW = "ROAD_LINE_SOLID_DOUBLE_YELLOW"
ROAD_LINE_PASSING_DOUBLE_YELLOW = "ROAD_LINE_PASSING_DOUBLE_YELLOW"
ROAD_LINE_UNKNOWN = "ROAD_LINE_UNKNOWN"

# ===== Road Edge Types =====
ROAD_EDGE_BOUNDARY = "ROAD_EDGE_BOUNDARY"
ROAD_EDGE_MEDIAN = "ROAD_EDGE_MEDIAN"
ROAD_EDGE_SIDEWALK = "ROAD_EDGE_SIDEWALK"
ROAD_EDGE_UNKNOWN = "ROAD_EDGE_UNKNOWN"

# ===== Map Feature Types =====
TRAFFIC_LIGHT = "TRAFFIC_LIGHT"
CROSSWALK = "CROSSWALK"
STOP_SIGN = "STOP_SIGN"
YIELD_SIGN = "YIELD_SIGN"
SPEED_BUMP = "SPEED_BUMP"
DRIVEWAY = "DRIVEWAY"
PARKING_LOT = "PARKING_LOT"
GUARDRAIL = "GUARDRAIL"
TRAFFIC_CONE = "TRAFFIC_CONE"
TRAFFIC_BARRIER = "TRAFFIC_BARRIER"

# ===== Traffic Light States =====
TRAFFIC_LIGHT_RED = "TRAFFIC_LIGHT_RED"
TRAFFIC_LIGHT_YELLOW = "TRAFFIC_LIGHT_YELLOW"
TRAFFIC_LIGHT_GREEN = "TRAFFIC_LIGHT_GREEN"
TRAFFIC_LIGHT_UNKNOWN = "TRAFFIC_LIGHT_UNKNOWN"
TRAFFIC_LIGHT_ARROW_RED = "TRAFFIC_LIGHT_ARROW_RED"
TRAFFIC_LIGHT_ARROW_YELLOW = "TRAFFIC_LIGHT_ARROW_YELLOW"
TRAFFIC_LIGHT_ARROW_GREEN = "TRAFFIC_LIGHT_ARROW_GREEN"
TRAFFIC_LIGHT_FLASHING_RED = "TRAFFIC_LIGHT_FLASHING_RED"
TRAFFIC_LIGHT_FLASHING_YELLOW = "TRAFFIC_LIGHT_FLASHING_YELLOW"

# ===== Type Groups =====
PARTICIPANT_TYPES = {VEHICLE, PEDESTRIAN, CYCLIST, OTHER}

LANE_TYPES = {
    LANE_SURFACE_STREET,
    LANE_FREEWAY,
    LANE_BIKE_LANE,
    LANE_UNKNOWN,
}

ROAD_LINE_TYPES = {
    ROAD_LINE_BROKEN_SINGLE_WHITE,
    ROAD_LINE_SOLID_SINGLE_WHITE,
    ROAD_LINE_SOLID_DOUBLE_WHITE,
    ROAD_LINE_BROKEN_SINGLE_YELLOW,
    ROAD_LINE_BROKEN_DOUBLE_YELLOW,
    ROAD_LINE_SOLID_SINGLE_YELLOW,
    ROAD_LINE_SOLID_DOUBLE_YELLOW,
    ROAD_LINE_PASSING_DOUBLE_YELLOW,
    ROAD_LINE_UNKNOWN,
}

ROAD_EDGE_TYPES = {
    ROAD_EDGE_BOUNDARY,
    ROAD_EDGE_MEDIAN,
    ROAD_EDGE_SIDEWALK,
    ROAD_EDGE_UNKNOWN,
}

MAP_FEATURE_TYPES = {
    TRAFFIC_LIGHT,
    CROSSWALK,
    STOP_SIGN,
    SPEED_BUMP,
    DRIVEWAY,
    GUARDRAIL,
}

TRAFFIC_LIGHT_STATES = {
    TRAFFIC_LIGHT_RED,
    TRAFFIC_LIGHT_YELLOW,
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_UNKNOWN,
    TRAFFIC_LIGHT_ARROW_RED,
    TRAFFIC_LIGHT_ARROW_YELLOW,
    TRAFFIC_LIGHT_ARROW_GREEN,
    TRAFFIC_LIGHT_FLASHING_RED,
    TRAFFIC_LIGHT_FLASHING_YELLOW,
}

ALL_TYPES = (
    PARTICIPANT_TYPES | LANE_TYPES | ROAD_LINE_TYPES | ROAD_EDGE_TYPES | MAP_FEATURE_TYPES | TRAFFIC_LIGHT_STATES
)

# ===== Output Formats =====
FORMAT_WAYMAX = "waymax"  # TFRecord format (Waymax/V-Max simulators)
FORMAT_GPUDRIVE = "gpudrive"  # JSON format (GPUDrive simulator)
FORMAT_PUFFERDRIVE = "pufferdrive"  # Binary format (PufferDrive simulator)

SUPPORTED_FORMATS = {FORMAT_WAYMAX, FORMAT_GPUDRIVE, FORMAT_PUFFERDRIVE}


# ===== Type Checking Functions =====
def is_participant(obj_type: str) -> bool:
    return obj_type in PARTICIPANT_TYPES


def is_lane(obj_type: str) -> bool:
    return obj_type in LANE_TYPES


def is_road_line(obj_type: str) -> bool:
    return obj_type in ROAD_LINE_TYPES


def is_road_edge(obj_type: str) -> bool:
    return obj_type in ROAD_EDGE_TYPES


def is_road_map_element(obj_type: str) -> bool:
    return obj_type in LANE_TYPES or obj_type in ROAD_LINE_TYPES or obj_type in ROAD_EDGE_TYPES


def is_marking_element(obj_type: str) -> bool:
    return obj_type in [CROSSWALK, SPEED_BUMP, DRIVEWAY, PARKING_LOT]


def is_traffic_object(obj_type: str) -> bool:
    return obj_type in [STOP_SIGN, TRAFFIC_CONE, TRAFFIC_BARRIER, GUARDRAIL]


def is_traffic_sign(obj_type: str) -> bool:
    return obj_type in [STOP_SIGN, SPEED_BUMP, DRIVEWAY]


def is_crosswalk(obj_type: str) -> bool:
    return obj_type == CROSSWALK


def is_map_feature(obj_type: str) -> bool:
    return obj_type in MAP_FEATURE_TYPES


def is_traffic_light_state(state: str) -> bool:
    return state in TRAFFIC_LIGHT_STATES


def is_valid_type(obj_type: str) -> bool:
    return obj_type in ALL_TYPES


# ===== Utility Functions =====
def is_yellow_line(line_type: str) -> bool:
    return "YELLOW" in line_type.upper()


def is_broken_line(line_type: str) -> bool:
    return "BROKEN" in line_type.upper()


def is_solid_line(line_type: str) -> bool:
    return "SOLID" in line_type.upper()


def simplify_traffic_light(state: str) -> str:
    """Convert detailed traffic light states to basic red/yellow/green/unknown"""
    state_upper = state.upper()
    if "RED" in state_upper or state == TRAFFIC_LIGHT_FLASHING_RED:
        return TRAFFIC_LIGHT_RED
    elif "YELLOW" in state_upper:
        return TRAFFIC_LIGHT_YELLOW
    elif "GREEN" in state_upper:
        return TRAFFIC_LIGHT_GREEN
    else:
        return TRAFFIC_LIGHT_UNKNOWN
