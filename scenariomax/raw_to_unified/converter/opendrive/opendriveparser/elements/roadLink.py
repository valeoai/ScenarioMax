

class Link(object):

    def __init__(self):
        self._id = None
        self._predecessor = None
        self._successor = None
        self._neighbors = []

    def __str__(self):
        return " > link id " + str(self._id) + " | successor: " + str(self._successor)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = int(value)

    @property
    def predecessor(self):
        return self._predecessor

    @predecessor.setter
    def predecessor(self, value):
        if not isinstance(value, Predecessor):
            raise TypeError("Value must be Predecessor")

        self._predecessor = value

    @property
    def successor(self):
        return self._successor

    @successor.setter
    def successor(self, value):
        if not isinstance(value, Successor):
            raise TypeError("Value must be Successor")

        self._successor = value

    @property
    def neighbors(self):
        return self._neighbors

    @neighbors.setter
    def neighbors(self, value):
        if not isinstance(value, list) or not all(isinstance(x, Neighbor) for x in value):
            raise TypeError("Value must be list of instances of Neighbor.")

        self._neighbors = value

    def addNeighbor(self, value):
        if not isinstance(value, Neighbor):
            raise TypeError("Value must be Neighbor")

        self._neighbors.append(value)


class Predecessor(object):

    def __init__(self):
        self._elementType = None
        self._elementId = None
        self._contactPoint = None

    def __str__(self):
        return str(self._elementType) + " with id " + str(self._elementId) + " contact at " + str(self._contactPoint)

    @property
    def elementType(self):
        return self._elementType

    @elementType.setter
    def elementType(self, value):
        if value not in ["road", "junction"]:
            raise AttributeError("Value must be road or junction")

        self._elementType = value

    @property
    def elementId(self):
        return self._elementId

    @elementId.setter
    def elementId(self, value):
        self._elementId = int(value)

    @property
    def contactPoint(self):
        return self._contactPoint

    @contactPoint.setter
    def contactPoint(self, value):
        if value not in ["start", "end"] and value is not None:
            raise AttributeError("Value must be start or end")

        self._contactPoint = value

class Successor(Predecessor):
    pass

    # def __init__(self):
    #     self._elementType = None
    #     self._elementId = None
    #     self._contactPoint = None

    # @property
    # def elementType(self):
    #     return self._elementType

    # @elementType.setter
    # def elementType(self, value):
    #     if value not in ["road", "junction"]:
    #         raise AttributeError("Value must be road or junction")

    #     self._elementType = value

    # @property
    # def elementId(self):
    #     return self._elementId

    # @elementId.setter
    # def elementId(self, value):
    #     self._elementId = int(value)

    # @property
    # def contactPoint(self):
    #     return self._contactPoint

    # @contactPoint.setter
    # def contactPoint(self, value):
    #     if value not in ["start", "end"] and value is not None:
    #         raise AttributeError("Value must be start or end")

    #     self._contactPoint = value


class Neighbor(object):

    def __init__(self):
        self._side = None
        self._elementId = None
        self._direction = None

    @property
    def side(self):
        return self._side

    @side.setter
    def side(self, value):
        if value not in ["left", "right"]:
            raise AttributeError("Value must be left or right")

        self._side = value

    @property
    def elementId(self):
        return self._elementId

    @elementId.setter
    def elementId(self, value):
        self._elementId = int(value)

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        if value not in ["same", "opposite"]:
            raise AttributeError("Value must be same or opposite")

        self._direction = value
