"""What happens during a run that something else may want to hear or show."""
import enum


class Event(enum.Enum):
    """What happens during a run."""

    HOP = 'hop'
    #: A hop was asked for and a hedge was in the way.
    BLOCKED = 'blocked'
    JUMP = 'jump'
    LAND = 'land'
    SLIDE = 'slide'
    COIN = 'coin'
    JEWEL = 'jewel'
    CRASH = 'crash'
