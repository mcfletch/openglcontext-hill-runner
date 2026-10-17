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
    #: The runner hit an obstacle, and the coins of a continue begin to count down.
    HIT = 'hit'
    #: The runner is back on its feet after a hit and runs on.
    RISE = 'rise'
    #: The continue is paid for: the coins have counted down and the red clears.
    PAID = 'paid'
    #: The coins ran out before the continue was paid for: the run is over.
    CRASH = 'crash'
