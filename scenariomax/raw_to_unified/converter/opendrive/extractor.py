from math import pi, sin, cos
import bisect


from scenariomax.raw_to_unified.description import ScenarioDescription as SD
from scenariomax.raw_to_unified.type import ScenarioType
from scenariomax.raw_to_unified.converter.opendrive.opendriveparser.elements.openDrive import OpenDrive

def calculate_reference_points_of_one_geometry(geometry, elevations, length, step, s_offset):
    """
    Calculate the stepwise reference points with position(x, y), tangent and distance between the point and the start.
    :param geometry:
    :param length:
    :param step:
    :return:
    """
    s_positions = [e.sPos for e in elevations]
    
    def eval_z_at_s(s):
        # find rightmost elevation segment whose sPos ≤ s
        idx = bisect.bisect_right(s_positions, s) - 1
        if idx < 0:
            # before first segment: clamp to first
            idx = 0
        ev = elevations[idx]
        ds = s - ev.sPos

        # cubic polynomial: a + b*ds + c*ds^2 + d*ds^3
        return ev.a + ev.b*ds + ev.c*(ds**2) + ev.d*(ds**3)

    nums = int(length / step) + 1
    res = []
    for i in range(nums):
        s_local = step * i
        s_global = s_offset + s_local
        pos_, tangent_ = geometry.calcPosition(s_local)
        x, y = pos_
        z = eval_z_at_s(s_global)
        one_point = {
            "position": (x, y, z),     # The location of the reference point
            "tangent": tangent_,    # Orientation of the reference point
            "s_geometry": s_local,       # The distance between the start point of the geometry and current point along the reference line
            "s_road": s_global,
        }
        res.append(one_point)
    return res


def get_geometry_length(geometry):
    """
    Get the length of one geometry (or the length of the reference line of the geometry).
    :param geometry:
    :return:
    """
    if hasattr(geometry, "length"):
        length = geometry.length
    elif hasattr(geometry, "_length"):
        length = geometry._length           # Some geometry has the attribute "_length".
    else:
        raise AttributeError("No attribute length found!!!")
    return length


def get_all_reference_points_of_one_road(geometries, elevations, step):
    """
    Obtain the sampling point of the reference line of the road, including:
    the position of the point
    the direction of the reference line at the point
    the distance of the point along the reference line relative to the start of the road
    the distance of the point relative to the start of geometry along the reference line
    :param geometries: Geometries of one road.
    :param step: Calculate steps.
    :return:
    """
    reference_points = []
    s_start_road = 0
    for geometry_id, geometry in enumerate(geometries):
        geometry_length = get_geometry_length(geometry)

        # Calculate all the reference points of current geometry.
        pos_tangent_s_list = calculate_reference_points_of_one_geometry(geometry, elevations, geometry_length, step=step, s_offset=s_start_road)

        # As for every reference points, add the distance start by road and its geometry index.
        pos_tangent_s_s_list = [{**point,
                                 "s_road": point["s_geometry"]+s_start_road,
                                 "index_geometry": geometry_id}
                                for point in pos_tangent_s_list]
        reference_points.extend(pos_tangent_s_s_list)

        s_start_road += geometry_length
    return reference_points

def get_width(widths, s):
    assert isinstance(widths, list), TypeError(type(widths))
    widths.sort(key=lambda x: x.sOffset)
    current_width = None
    # EPS = 1e-5
    milestones = [width.sOffset for width in widths] + [float("inf")]

    control_mini_section = [(start, end) for (start, end) in zip(milestones[:-1], milestones[1:])]
    for width, start_end in zip(widths, control_mini_section):
        start, end = start_end
        if start <= s < end:
            ds = s - width.sOffset
            current_width = width.a + width.b * ds + width.c * ds ** 2 + width.d * ds ** 3
    return current_width


def get_lane_offset(lane_offsets, section_s, length=float("inf")):

    assert isinstance(lane_offsets, list), TypeError(type(lane_offsets))
    if not lane_offsets:
        return 0
    lane_offsets.sort(key=lambda x: x.sPos)
    current_offset = 0
    EPS = 1e-5
    milestones = [lane_offset.sPos for lane_offset in lane_offsets] + [length+EPS]

    control_mini_section = [(start, end) for (start, end) in zip(milestones[:-1], milestones[1:])]
    for offset_params, start_end in zip(lane_offsets, control_mini_section):
        start, end = start_end
        if start <= section_s < end:
            ds = section_s - offset_params.sPos
            current_offset = offset_params.a + offset_params.b * ds + offset_params.c * ds ** 2 + offset_params.d * ds ** 3
    return current_offset


class LaneOffsetCalculate:

    def __init__(self, lane_offsets):
        lane_offsets = list(sorted(lane_offsets, key=lambda x: x.sPos))
        lane_offsets_dict = dict()
        for lane_offset in lane_offsets:
            a = lane_offset.a
            b = lane_offset.b
            c = lane_offset.c
            d = lane_offset.d
            s_start = lane_offset.sPos
            lane_offsets_dict[s_start] = (a, b, c, d)
        self.lane_offsets_dict = lane_offsets_dict

    def calculate_offset(self, s):
        for s_start, (a, b, c, d) in reversed(self.lane_offsets_dict.items()): # e.g. 75, 25
            if s >= s_start:
                ds = s - s_start
                offset = a + b * ds + c * ds ** 2 + d * ds ** 3
                return offset
        return 0


def calculate_area_of_one_left_lane(left_lane, points, most_left_points):
    inner_points = most_left_points[:]

    widths = left_lane.widths
    update_points = []
    for reference_point, inner_point in zip(points, inner_points):

        tangent = reference_point["tangent"]
        s_lane_section = reference_point["s_lane_section"]
        lane_width = get_width(widths, s_lane_section)

        normal_left = tangent + pi / 2
        x_inner, y_inner, z_inner = inner_point

        lane_width_offset = lane_width

        x_outer = x_inner + cos(normal_left) * lane_width_offset
        y_outer = y_inner + sin(normal_left) * lane_width_offset

        update_points.append((x_outer, y_outer, z_inner))

    outer_points = update_points[:]
    most_left_points = outer_points[:]

    current_ara = {
        "inner": inner_points,
        "outer": outer_points,
    }
    return current_ara, most_left_points


def calculate_area_of_one_right_lane(right_lane, points, most_right_points):
    inner_points = most_right_points[:]

    widths = right_lane.widths
    update_points = []
    for reference_point, inner_point in zip(points, inner_points):

        tangent = reference_point["tangent"]
        s_lane_section = reference_point["s_lane_section"]
        lane_width = get_width(widths, s_lane_section)

        normal_eight = tangent - pi / 2
        x_inner, y_inner, z_inner = inner_point

        lane_width_offset = lane_width

        x_outer = x_inner + cos(normal_eight) * lane_width_offset
        y_outer = y_inner + sin(normal_eight) * lane_width_offset

        update_points.append((x_outer, y_outer, z_inner))

    outer_points = update_points[:]
    most_right_points = outer_points[:]

    current_ara = {
        "inner": inner_points,
        "outer": outer_points,
    }
    return current_ara, most_right_points


def calculate_lane_area_within_one_lane_section(lane_section, points):
    """
    Lane areas are represented by boundary lattice. Calculate boundary points of every lanes.
    :param lane_section:
    :param points:
    :return:
    """

    all_lanes = lane_section.allLanes

    # Process the lane indexes.
    left_lanes = [lane for lane in all_lanes if int(lane.id) > 0]
    right_lanes = [lane for lane in all_lanes if int(lane.id) < 0]
    left_lanes.sort(key=lambda x: x.id)
    right_lanes.sort(reverse=True, key=lambda x: x.id)

    # Get the lane area of left lanes and the most left lane line.
    left_lanes_area = dict()
    most_left_points = [point["position_center_lane"] for point in points][:]
    for left_lane in left_lanes:
        current_area, most_left_points = calculate_area_of_one_left_lane(left_lane, points, most_left_points)
        left_lanes_area[left_lane.id] = current_area

    # Get the lane area of right lanes and the most right lane line.
    right_lanes_area = dict()
    most_right_points = [point["position_center_lane"] for point in points][:]
    for right_lane in right_lanes:
        current_area, most_right_points = calculate_area_of_one_right_lane(right_lane, points, most_right_points)
        right_lanes_area[right_lane.id] = current_area

    return left_lanes_area, right_lanes_area, most_left_points, most_right_points


def calculate_points_of_reference_line_of_one_section(points):
    """
    Calculate center lane points accoding to the reference points and offsets.
    :param points: Points on reference line including position and tangent.
    :return: Updated points.
    """
    res = []
    for point in points:
        tangent = point["tangent"]
        x, y, z = point["position"]    # Points on reference line.
        normal = tangent + pi / 2
        lane_offset = point["lane_offset"]  # Offset of center lane.

        x += cos(normal) * lane_offset
        y += sin(normal) * lane_offset

        point = {
            **point,
            "position_center_lane": (x, y, z),
        }
        res.append(point)
    return res


def calculate_s_lane_section(reference_points, lane_sections):

    res = []
    for point in reference_points:

        for lane_section in reversed(lane_sections):
            if point["s_road"] >= lane_section.sPos:
                res.append(
                    {
                        **point,
                        "s_lane_section": point["s_road"] - lane_section.sPos,
                        "index_lane_section": lane_section.idx,
                    }
                )
                break
    return res


def uncompress_dict_list(dict_list: list):
    assert isinstance(dict_list, list), TypeError("Keys")
    if not dict_list:
        return dict()

    keys = set(dict_list[0].keys())
    for dct in dict_list:
        cur = set(dct.keys())
        assert keys == cur, "Inconsistency of dict keys! {} {}".format(keys, cur)

    res = dict()
    for sample in dict_list:
        for k, v in sample.items():
            if k not in res:
                res[k] = [v]
            else:
                res[k].append(v)

    keys = list(sorted(list(keys)))
    res = {k: res[k] for k in keys}
    return res

def get_lane_line(section_data: dict):
    left_lanes_area = section_data["left_lanes_area"]
    right_lanes_area = section_data["right_lanes_area"]

    lane_line_left = {}
    if left_lanes_area:
        indexes = list(left_lanes_area.keys())
        for index_inner, index_outer in zip(indexes, indexes[1:] + [None]):
            if index_outer is not None:
                lane_line_left[(index_inner, index_outer)] = left_lanes_area[index_inner]["outer"]

    lane_line_right = {}
    if right_lanes_area:
        indexes = list(right_lanes_area.keys())
        for index_inner, index_outer in zip(indexes, indexes[1:] + [None]):
            if index_outer is not None:
                lane_line_right[(index_inner, index_outer)] = right_lanes_area[index_inner]["outer"]

    return {"lane_line_left": lane_line_left, "lane_line_right": lane_line_right}



def get_lane_area_of_one_road(road, step):
    """
    Get all corresponding positions of every lane section in one road.
    :param road:
    :param step:
    :return: A dictionary of dictionary: {(road id, lane section id): section data}
    Section data is a dictionary of position information.
    section_data = {
        "left_lanes_area": left_lanes_area,
        "right_lanes_area": right_lanes_area,
        "most_left_points": most_left_points,
        "most_right_points": most_right_points,
        "types": types,
        "reference_points": uncompressed_lane_section_data,
    }
    """
    geometries = road.planView._geometries
    elevations = road.elevationProfile._elevations
    # Lane offset is the offset between center lane (width is 0) and the reference line.
    lane_offsets = road.lanes.laneOffsets
    lane_offset_calculate = LaneOffsetCalculate(lane_offsets=lane_offsets)
    lane_sections = road.lanes.laneSections
    lane_sections = list(sorted(lane_sections, key=lambda x: x.sPos))   # Sort the lane sections by start position.

    reference_points = get_all_reference_points_of_one_road(geometries, elevations, step=step)  # Extract the reference points.

    # Calculate the offsets of center lane.
    reference_points = [{**point, "lane_offset":  lane_offset_calculate.calculate_offset(point["s_road"])}
                        for point in reference_points]

    # Calculate the points of center lane based on reference points and offsets.
    reference_points = calculate_points_of_reference_line_of_one_section(reference_points)

    # Calculate the distance of each point starting from the current section along the direction of the reference line.
    reference_points = calculate_s_lane_section(reference_points, lane_sections)

    total_areas = dict()
    for lane_section in lane_sections:
        section_start = lane_section.sPos  # Start position of the section in current road.
        section_end = lane_section.sPos + lane_section.length  # End position of the section in current road.

        # Filter out the points belonging to current lane section.
        current_reference_points = list(filter(lambda x: section_start <= x["s_road"] < section_end, reference_points))

        # Calculate the boundary point of every lane in current lane section.
        area = calculate_lane_area_within_one_lane_section(lane_section, current_reference_points)
        left_lanes_area, right_lanes_area, most_left_points, most_right_points = area

        # Extract types and indexes.
        types = {lane.id: lane.type for lane in lane_section.allLanes if lane.id != 0}
        index = (road.id, lane_section.idx)

        # Convert dict list to list dict of the reference points information.
        uncompressed_lane_section_data = uncompress_dict_list(current_reference_points)

        # Integrate all the information of current lane section of current road.
        section_data = {
            "left_lanes_area": left_lanes_area,
            "right_lanes_area": right_lanes_area,
            "most_left_points": most_left_points,
            "most_right_points": most_right_points,
            "types": types,
            "reference_points": uncompressed_lane_section_data,  # 这些是lane section的信息
        }

        # Get all lane lines with their left and right lanes.
        lane_line = get_lane_line(section_data)
        section_data.update(lane_line)

        total_areas[index] = section_data

    return total_areas

# --- Helper: Convert roadMark attributes to ScenarioNet line type ---
def _map_roadmark_to_line_type(road_mark):
    if road_mark is None or road_mark.type is None:
        return ScenarioType.LINE_UNKNOWN

    mark_type = road_mark.type.lower()
    mark_color = road_mark.color.lower() if road_mark.color else None

    if mark_type == "solid":
        if mark_color == "white":
            return ScenarioType.LINE_SOLID_SINGLE_WHITE
        elif mark_color == "yellow":
            return ScenarioType.LINE_SOLID_SINGLE_YELLOW
    elif mark_type == "broken":
        if mark_color == "white":
            return ScenarioType.LINE_BROKEN_SINGLE_WHITE
        elif mark_color == "yellow":
            return ScenarioType.LINE_BROKEN_SINGLE_YELLOW
    elif mark_type == "curb":
        return ScenarioType.BOUNDARY_LINE  # You may want a dedicated "CURB" type if needed
    elif mark_type == "none":
        if mark_color == "white":
            return ScenarioType.LINE_SOLID_SINGLE_WHITE  # Or BROKEN? your call
        else:
            return ScenarioType.LINE_UNKNOWN

    return ScenarioType.LINE_UNKNOWN

# --- Helper: Convert (s, t) to (x, y) using reference line ---
def st_to_world_xy(s, t, ref):
    s_list = ref["s_road"]
    for i in range(len(s_list) - 1):
        if s_list[i] <= s <= s_list[i + 1]:
            heading = ref["tangent"][i]
            x, y, z = ref["position"][i]
            x_new = x + cos(heading + pi / 2) * t
            y_new = y + sin(heading + pi / 2) * t
            return x_new, y_new, z
    # fallback
    i = min(range(len(s_list)), key=lambda j: abs(s_list[j] - s))
    heading = ref["tangent"][i]
    x, y, z = ref["position"][i]
    x_new = x + cos(heading + pi / 2) * t
    y_new = y + sin(heading + pi / 2) * t
    return x_new, y_new, z

def convert_opendrive_to_scenarionet_map_features(total_areas: dict, roads: list) -> dict:
    map_features = {}
    lane_id_counter = 0
    line_id_counter = 0

    for (road_id, section_id), section in total_areas.items():
        types = section["types"]
        reference_points = section["reference_points"]
        road_obj = next((r for r in roads if r.id == road_id), None)
        if road_obj is None:
            continue

        # Process lanes and road edges
        for lane_dict in [section["left_lanes_area"], section["right_lanes_area"]]:
            for lane_id, lane_area in lane_dict.items():
                inner_pts = lane_area["inner"]
                outer_pts = lane_area["outer"]
                polyline = [[x, y] for x, y, z in inner_pts]  # Directly use inner boundary as the "line"
                polygon = [[p[0], p[1]] for p in inner_pts + outer_pts[::-1]]

                odr_type = types.get(lane_id, "unknown")

                # Map OpenDRIVE lane types to ScenarioNet types
                if odr_type == "driving":
                    scenarionet_type = ScenarioType.LINE_BROKEN_SINGLE_WHITE
                    prefix = "line"
                elif odr_type in {"shoulder"}:
                    scenarionet_type = ScenarioType.BOUNDARY_LINE
                    prefix = "road_edge"
                else:
                    scenarionet_type = ScenarioType.LANE_UNKNOWN
                    prefix = "lane"

                map_features[f"{prefix}_{lane_id_counter}"] = {
                    SD.TYPE: scenarionet_type,
                    SD.POLYLINE: polyline,
                    SD.POLYGON: polygon,
                    SD.ENTRY: [],
                    SD.EXIT: [],
                    SD.LEFT_NEIGHBORS: [],
                    SD.RIGHT_NEIGHBORS: [],
                }
                lane_id_counter += 1

        # Process lane lines
        for boundary_name, boundary_dict in [
            ("lane_line_left", section.get("lane_line_left", {})),
            ("lane_line_right", section.get("lane_line_right", {})),
        ]:
            for key, line_pts in boundary_dict.items():
                polyline = [[p[0], p[1]] for p in line_pts]

                # Get roadMark from either of the two neighboring lanes
                lane_type = ScenarioType.LINE_UNKNOWN
                lane_id_1, lane_id_2 = key
                for lid in [lane_id_1, lane_id_2]:
                    for sec in road_obj.lanes.laneSections:
                        full_lane = next((l for l in sec.allLanes if int(l.id) == int(lid)), None)
                        if full_lane and full_lane.road_mark:
                            lane_type = _map_roadmark_to_line_type(full_lane.road_mark)
                            break

                map_features[f"{boundary_name}_{line_id_counter}"] = {
                    SD.TYPE: lane_type,
                    SD.POLYLINE: polyline,
                }
                line_id_counter += 1


        # Process stop signs (OpenDRIVE <object name="StopLine">)
        for obj in getattr(road_obj, "objects", []):
            if obj.name.lower() == "stopline":
                x, y, z = st_to_world_xy(obj.s, obj.t, reference_points)
                map_features[f"stop_sign_{obj.id}"] = {
                    SD.TYPE: ScenarioType.STOP_SIGN,
                    SD.POSITION: [x, y, z],
                    "road_id": road_id,
                    "length": obj.length,
                    "width": obj.width,
                }

    return map_features


def extract_dynamic_map_states(total_areas: dict, roads: list) -> dict:
    """
    Extract static traffic light states from OpenDRIVE signals into ScenarioNet dynamic map states.
    """
    dynamic_map_states = {}
    length = 0  # Static map → traffic light state doesn’t change

    for (road_id, section_id), section in total_areas.items():
        reference_points = section["reference_points"]
        road_obj = next((r for r in roads if r.id == road_id), None)
        if road_obj is None:
            continue

        for sig in getattr(road_obj, "signals", []):
            if sig.type == "1000001":  # treat this as a 3-light signal
                x, y, z = st_to_world_xy(sig.s, sig.t, reference_points)
                signal_id = str(sig.id)

                dynamic_map_states[signal_id] = {
                    SD.TYPE: ScenarioType.TRAFFIC_LIGHT,
                    "state": {SD.TRAFFIC_LIGHT_STATUS: [ScenarioType.LIGHT_UNKNOWN] * length},
                    SD.TRAFFIC_LIGHT_POSITION: [x, y, z],
                    SD.TRAFFIC_LIGHT_LANE: None,  # could be inferred later if needed
                    SD.METADATA: {
                        "track_length": length,
                        "type": ScenarioType.TRAFFIC_LIGHT,
                        "object_id": signal_id,
                        "dataset": "opendrive",
                    },
                }

    return dynamic_map_states

def get_all_lanes(road_network: OpenDrive, step) -> dict:
    """
    Get all lanes of one road network.
    :param road_network: Parsed road network.
    :param step: Step of calculation.
    :return: Dictionary with the following format:
        keys: (road id, lane section id)
        values: dict(left_lanes_area, right_lanes_area, most_left_points, most_right_points, types, reference_points)
    """
    roads = road_network.roads
    total_areas_all_roads = dict()

    for road in roads:
        lanes_of_one_road = get_lane_area_of_one_road(road, step=step)
        total_areas_all_roads = {**total_areas_all_roads, **lanes_of_one_road}
    return total_areas_all_roads

def convert_to_scenario(total_areas, version, map_name, roads):
    scenario = SD()
    scenario[SD.ID] = map_name
    scenario[SD.VERSION] = f"opendrive_{version}"
    scenario[SD.LENGTH] = 0.0
    scenario[SD.TRACKS] = {}
    scenario[SD.METADATA] = {
        "dataset": "opendrive",
        "map": map_name,
    }
    scenario[SD.METADATA][SD.TIMESTEP] = []
    scenario[SD.MAP_FEATURES] = convert_opendrive_to_scenarionet_map_features(total_areas, roads)
    scenario[SD.DYNAMIC_MAP_STATES] = extract_dynamic_map_states(total_areas, roads)
    return scenario

def convert_opendrive_maps(opendrive_obj: OpenDrive, version: str) -> dict:
    total_areas = get_all_lanes(opendrive_obj, step=0.5)
    scenario = convert_to_scenario(total_areas, version, opendrive_obj.header._name, opendrive_obj.roads)
    return scenario
