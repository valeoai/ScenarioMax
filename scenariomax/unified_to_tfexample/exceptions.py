# Legacy compatibility imports - use scenariomax.core.exceptions instead
from scenariomax.core.exceptions import (
    NotEnoughValidObjectsException,
    OverpassException,
)


__all__ = ["OverpassException", "NotEnoughValidObjectsException"]
