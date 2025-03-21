class OverpassException(Exception):
    """Exception raised when an overpass is detected in the input data."""

    def __init__(self, message="Overpass detected in the roadgraph. Skip scenario."):
        super().__init__(message)
        self.message = message


class NotEnoughValidObjectsException(Exception):
    """Exception raised when there are not enough valid objects in the input data."""

    def __init__(self, message="Not enough valid objects in the scenario for multi-agents. Skip scenario."):
        super().__init__(message)
        self.message = message
