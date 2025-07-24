

class Type(object):

    allowedTypes = ["unknown", "rural", "motorway", "town", "lowSpeed", "pedestrian", "bicycle"]

    def __init__(self):
        self._sPos = None
        self._type = None
        self._speed = None

    @property
    def sPos(self):
        return self._sPos

    @sPos.setter
    def sPos(self, value):
        self._sPos = float(value)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        if value not in self.allowedTypes:
            raise AttributeError("Type not allowed.")

        self._type = value

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        if not isinstance(value, Speed):
            raise TypeError("Value must be instance of Speed.")

        self._speed = value


class Speed(object):

    def __init__(self):
        self._max = None
        self._unit = None

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self._max = str(value)

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        # TODO validate input
        self._unit = str(value)
