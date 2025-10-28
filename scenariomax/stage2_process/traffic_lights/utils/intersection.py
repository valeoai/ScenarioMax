from .generic import TLS, Direction, Pt
from .geometry import classify_direction


class VehicleState:
    def __init__(self, object_id: int, lane_pos_idx: int, speed: float, acceleration: float = None):
        self.object_id: int = object_id
        self.lane_pos_idx: int = lane_pos_idx
        self.speed: float = speed  # m/s
        self.acceleration: float = acceleration  # m/s^2


class InJunctionLane:
    def __init__(
        self,
        shape: list[Pt],
        record_tls: list = [],
        record_vehs: list[dict[int, VehicleState]] = [{}],
        id=None,
        length: int = 0,
    ) -> None:
        self.id = id
        self.shape: list[Pt] = shape[:]
        self.direction: Direction = classify_direction([(pt.x, pt.y) for pt in shape])

        if len(record_vehs) == 0:
            record_vehs = [{} for _ in range(length)]

        if len(record_tls) == 0:
            record_tls = [-1 for _ in range(length)]

        self.record_vehs: list[dict[int, VehicleState]] = record_vehs[:]

        # tls-related
        self.record_tls_waymonic: list = record_tls[:]
        self.record_tls: list[TLS] = [tls.generalize() for tls in record_tls]
        self.new_tls: list[TLS] = [TLS.UNKNOWN for _ in range(length)]


class ApproachingLane:
    def __init__(
        self,
        shape: list[Pt],
        record_vehs: list[dict[int, VehicleState]] = [{}],
        injunction_lanes: list[InJunctionLane] = [],
        id=None,
        length: int = 0,
    ) -> None:
        self.id = id
        self.shape: list[Pt] = shape[:]

        if len(record_vehs) == 0:
            record_vehs = [{} for _ in range(length)]

        self.record_vehs: list[dict[int, VehicleState]] = record_vehs[:]
        self.injunction_lanes: list[InJunctionLane] = injunction_lanes[:]
