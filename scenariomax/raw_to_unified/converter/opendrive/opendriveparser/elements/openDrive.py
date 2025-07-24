
class OpenDrive(object):

    def __init__(self):
        self._header = None
        self._roads = []
        self._controllers = []
        self._junctions = []
        self._junctionGroups = []
        self._stations = []

    @property
    def header(self):
        return self._header

    @property
    def roads(self):
        return self._roads

    def getRoad(self, id):
        for road in self._roads:
            if road.id == id:
                return road

        return None

    @property
    def controllers(self):
        return self._controllers

    @property
    def junctions(self):
        return self._junctions

    @property
    def junctionGroups(self):
        return self._junctionGroups

    @property
    def stations(self):
        return self._stations


class Header(object):

    def __init__(self):
        self._revMajor = None
        self._revMinor = None
        self._name = None
        self._version = None
        self._date = None
        self._north = None
        self._south = None
        self._east = None
        self._west = None
        self._vendor = None
