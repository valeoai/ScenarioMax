
from scenariomax.raw_to_unified.converter.opendrive.opendriveparser.elements.roadPlanView import PlanView
from scenariomax.raw_to_unified.converter.opendrive.opendriveparser.elements.roadLink import Link
from scenariomax.raw_to_unified.converter.opendrive.opendriveparser.elements.roadLanes import Lanes
from scenariomax.raw_to_unified.converter.opendrive.opendriveparser.elements.roadElevationProfile import ElevationProfile
from scenariomax.raw_to_unified.converter.opendrive.opendriveparser.elements.roadLateralProfile import LateralProfile

class Road(object):

    def __init__(self):
        self._id = None
        self._name = None
        self._junction = None
        self._length = None

        self._header = None # TODO
        self._link = Link()
        self._types = []
        self._planView = PlanView()
        self._elevationProfile = ElevationProfile()
        self._lateralProfile = LateralProfile()
        self._lanes = Lanes()
        self._signals = []
        self._objects = []

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = int(value)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def junction(self):
        return self._junction

    @junction.setter
    def junction(self, value):
        if not isinstance(value, int) and value is not None:
            raise TypeError("Property must be a int or NoneType")

        if value == -1:
            value = None

        self._junction = value

    @property
    def link(self):
        return self._link

    @property
    def types(self):
        return self._types

    @property
    def planView(self):
        return self._planView

    @property
    def elevationProfile(self):
        return self._elevationProfile

    @property
    def lateralProfile(self):
        return self._lateralProfile

    @property
    def lanes(self):
        return self._lanes

    @property
    def objects(self):
        return self._objects
    
    @property
    def signals(self):
        return self._signals