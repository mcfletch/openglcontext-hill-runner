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
    #: The runner hit an obstacle and has damage left to take.
    HIT = 'hit'
    #: Metres run without a hit took one damage away.
    HEAL = 'heal'
    #: The hit that ended the run.
    CRASH = 'crash'
