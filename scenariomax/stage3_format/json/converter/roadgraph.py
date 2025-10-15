import numpy as np


# Road types that are filtered out
FILTERED_ROAD_TYPES = ["ROAD_EDGE_SIDEWALK", "DRIVEWAY"]

# Mapping from scenario types to Waymax types
TYPE_MAPPING = {
    "LANE_FREEWAY": 1,
    "LANE_CENTER_FREEWAY": 1,
    "LANE_SURFACE_STREET": 2,
    "LANE_SURFACE_UNSTRUCTURE": 2,
    "LANE_BIKE_LANE": 3,
    "ROAD_LINE_BROKEN_SINGLE_WHITE": 6,
    "ROAD_LINE_SOLID_SINGLE_WHITE": 7,
    "ROAD_LINE_SOLID_DOUBLE_WHITE": 8,
    "ROAD_LINE_BROKEN_SINGLE_YELLOW": 9,
    "ROAD_LINE_BROKEN_DOUBLE_YELLOW": 10,
    "ROAD_LINE_SOLID_SINGLE_YELLOW": 11,
    "ROAD_LINE_SOLID_DOUBLE_YELLOW": 12,
    "ROAD_LINE_PASSING_DOUBLE_YELLOW": 13,
    "ROAD_EDGE_BOUNDARY": 15,
    "ROAD_EDGE_MEDIAN": 16,
    "STOP_SIGN": 17,
    "CROSSWALK": 18,
    "SPEED_BUMP": 19,
}

TYPE_TO_MAP_FEATURE_NAME = {
    "ROAD_EDGE": "road_edge",
    "ROAD_LINE": "road_line",
    "LANE": "lane",
    "STOP_SIGN": "stop_sign",
    "CROSSWALK": "crosswalk",
    "SPEED_BUMP": "speed_bump",
    "DRIVEWAY": "driveway",
}


def _map_type_to_mapfeature(type):
    # For exact matches first
    if type in TYPE_TO_MAP_FEATURE_NAME:
        return TYPE_TO_MAP_FEATURE_NAME[type]
    # For partial matches (contains)
    for key, value in TYPE_TO_MAP_FEATURE_NAME.items():
        if key in type:
            return value
    return str.lower(type)


def convert_map_features(scenario_net_map_features):
    road_features = []
    edge_segments = []
    edge_points = []
    index = 0

    for map_feature_id in scenario_net_map_features:
        feature = scenario_net_map_features[map_feature_id]
        feature_type = feature["type"]

        if feature_type in FILTERED_ROAD_TYPES:
            continue

        new_feature = {
            "geometry": [],
            "type": _map_type_to_mapfeature(feature_type),
            "map_element_id": TYPE_MAPPING.get(feature_type, 0),
            "id": int(map_feature_id) if map_feature_id.isdigit() else index,
        }
        geometry_key = next((k for k in ["polyline", "polygon", "position"] if k in feature), None)

        if not geometry_key:
            continue

        original_geometry = feature[geometry_key]
        original_geometry = (
            np.expand_dims(original_geometry, 0) if len(original_geometry.shape) == 1 else original_geometry
        )

        if original_geometry.shape[-1] == 3:
            geometry = [{"x": point[0], "y": point[1], "z": point[2]} for point in original_geometry]
        elif original_geometry.shape[-1] == 2:
            geometry = [{"x": point[0], "y": point[1], "z": 0.0} for point in original_geometry]

        new_feature["geometry"] = geometry

        if "ROAD_EDGE" in feature_type:
            edge_vertices = [[r["x"], r["y"], r["z"]] for r in new_feature["geometry"]]
            edge_points.extend(edge_vertices)
            edge_segments += [[edge_vertices[i], edge_vertices[i + 1]] for i in range(len(edge_vertices) - 1)]

        road_features.append(new_feature)
        index += 1

    # Check for 3D structures like GPUDrive template does
    if len(edge_points) > 0:
        edge_points = np.array(edge_points)
        if len(edge_points) > 0:
            # Calculate pairwise distances in xy plane efficiently
            xy_points = edge_points[:, :2]
            tolerance = 0.2
            has_3d = False

            # Process in chunks to avoid memory issues
            chunk_size = 1000
            for i in range(0, len(xy_points), chunk_size):
                chunk = xy_points[i : i + chunk_size]
                # Calculate distances between current chunk and all points
                dists = np.linalg.norm(chunk[:, np.newaxis] - xy_points, axis=2)
                potential_pairs = np.where((dists < tolerance) & (dists > 0))

                # Check z-values for identified pairs
                for p1, p2 in zip(*potential_pairs):
                    p1_idx = i + p1  # Adjust index for chunking
                    if abs(edge_points[p1_idx, 2] - edge_points[p2, 2]) > tolerance:
                        has_3d = True
                        break

                if has_3d:
                    break

            # Skip this scenario if it has 3D structures
            if has_3d:
                return None, None

    return road_features, edge_segments
