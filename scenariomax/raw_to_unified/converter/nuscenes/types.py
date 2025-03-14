from scenariomax.raw_to_unified.converter.type import ScenarioType


ALL_TYPE = {
    "noise": "noise",
    "human.pedestrian.adult": "adult",
    "human.pedestrian.child": "child",
    "human.pedestrian.wheelchair": "wheelchair",
    "human.pedestrian.stroller": "stroller",
    "human.pedestrian.personal_mobility": "p.mobility",
    "human.pedestrian.police_officer": "police",
    "human.pedestrian.construction_worker": "worker",
    "animal": "animal",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus.bendy",
    "vehicle.bus.rigid": "bus.rigid",
    "vehicle.truck": "truck",
    "vehicle.construction": "constr. veh",
    "vehicle.emergency.ambulance": "ambulance",
    "vehicle.emergency.police": "police car",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "trafficcone",
    "movable_object.pushable_pullable": "push/pullable",
    "movable_object.debris": "debris",
    "static_object.bicycle_rack": "bicycle racks",
    "flat.driveable_surface": "driveable",
    "flat.sidewalk": "sidewalk",
    "flat.terrain": "terrain",
    "flat.other": "flat.other",
    "static.manmade": "manmade",
    "static.vegetation": "vegetation",
    "static.other": "static.other",
    "vehicle.ego": "ego",
}
NOISE_TYPE = {
    "noise": "noise",
    "animal": "animal",
    "static_object.bicycle_rack": "bicycle racks",
    "movable_object.pushable_pullable": "push/pullable",
    "movable_object.debris": "debris",
    "static.manmade": "manmade",
    "static.vegetation": "vegetation",
    "static.other": "static.other",
}
HUMAN_TYPE = {
    "human.pedestrian.adult": "adult",
    "human.pedestrian.child": "child",
    "human.pedestrian.wheelchair": "wheelchair",
    "human.pedestrian.stroller": "stroller",
    "human.pedestrian.personal_mobility": "p.mobility",
    "human.pedestrian.police_officer": "police",
    "human.pedestrian.construction_worker": "worker",
}
BICYCLE_TYPE = {
    "vehicle.bicycle": "bicycle",
    "vehicle.motorcycle": "motorcycle",
}
VEHICLE_TYPE = {
    "vehicle.car": "car",
    "vehicle.bus.bendy": "bus.bendy",
    "vehicle.bus.rigid": "bus.rigid",
    "vehicle.truck": "truck",
    "vehicle.construction": "constr. veh",
    "vehicle.emergency.ambulance": "ambulance",
    "vehicle.emergency.police": "police car",
    "vehicle.trailer": "trailer",
    "vehicle.ego": "ego",
}
OBSTACLE_TYPE = {
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "trafficcone",
}
TERRAIN_TYPE = {
    "flat.driveable_surface": "driveable",
    "flat.sidewalk": "sidewalk",
    "flat.terrain": "terrain",
    "flat.other": "flat.other",
}


def get_scenarionet_type(obj_type):
    meta_type = obj_type
    _type = None
    if ALL_TYPE[obj_type] == "barrier":
        _type = ScenarioType.TRAFFIC_BARRIER
    elif ALL_TYPE[obj_type] == "trafficcone":
        _type = ScenarioType.TRAFFIC_CONE
    elif obj_type in VEHICLE_TYPE:
        _type = ScenarioType.VEHICLE
    elif obj_type in HUMAN_TYPE:
        _type = ScenarioType.PEDESTRIAN
    elif obj_type in BICYCLE_TYPE:
        _type = ScenarioType.CYCLIST

    # assert meta_type != ScenarioType.UNSET and meta_type != "noise"
    return _type, meta_type
