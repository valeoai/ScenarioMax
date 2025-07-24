

class ElevationProfile(object):

    def __init__(self):
        self._elevations = []

    @property
    def elevations(self):
        return self._elevations


class Elevation(object):

    def __init__(self):
        self._sPos = None
        self._a = None
        self._b = None
        self._c = None
        self._d = None

    @property
    def sPos(self):
        return self._sPos

    @sPos.setter
    def sPos(self, value):
        self._sPos = float(value)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = float(value)

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = float(value)

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = float(value)

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        self._d = float(value)
