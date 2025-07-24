class Signal:
    def __init__(self):
        self.id = None
        self.name = None
        self.s = None
        self.t = None
        self.zOffset = 0.0
        self.hOffset = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.orientation = "+"
        self.dynamic = False
        self.country = ""
        self.type = ""
        self.subtype = ""
        self.value = -1.0
        self.text = ""
        self.height = 0.0
        self.width = 0.0
        self.validities = []

class Validity:
    def __init__(self):
        self.fromLane = None
        self.toLane = None


class Object:
    def __init__(self):
        self.id = None
        self.name = None
        self.s = None
        self.t = None
        self.zOffset = 0.0
        self.hdg = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.orientation = "+"
        self.type = ""
        self.length = 0.0
        self.width = 0.0
        self.outline = []

class CornerLocal:
    def __init__(self):
        self.u = 0.0
        self.v = 0.0
        self.z = 0.0