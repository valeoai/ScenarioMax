

class LateralProfile(object):

    def __init__(self):
        self._superelevations = []
        self._crossfalls = []
        self._shapes = []

    @property
    def superelevations(self):
        return self._superelevations

    @superelevations.setter
    def superelevations(self, value):
        if not isinstance(value, list) or not all(isinstance(x, Superelevation) for x in value):
            raise TypeError("Value must be an instance of Superelevation.")

        self._superelevations = value


    @property
    def crossfalls(self):
        return self._crossfalls

    @crossfalls.setter
    def crossfalls(self, value):
        if not isinstance(value, list) or not all(isinstance(x, Crossfall) for x in value):
            raise TypeError("Value must be an instance of Crossfall.")

        self._crossfalls = value


    @property
    def shapes(self):
        return self._shapes

    @shapes.setter
    def shapes(self, value):
        if not isinstance(value, list) or not all(isinstance(x, Shape) for x in value):
            raise TypeError("Value must be a list of instances of Shape.")

        self._shapes = value



class Superelevation(object):

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


class Crossfall(object):

    def __init__(self):
        self._side = None
        self._sPos = None
        self._a = None
        self._b = None
        self._c = None
        self._d = None

    @property
    def side(self):
        return self._side

    @side.setter
    def side(self, value):
        if value not in ["left", "right", "both"]:
            raise TypeError("Value must be string with content 'left', 'right' or 'both'.")

        self._side = value

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


class Shape(object):

    def __init__(self):
        self._sPos = None
        self._t = None
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
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = float(value)

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
