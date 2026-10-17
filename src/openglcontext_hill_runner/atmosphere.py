"""The air of the hill: bright and hazy in the open, dark and close in a cave. No GL.

The window applies what :func:`air_at` says to the sun, the environment light and the fog each
frame. How dark it is depends on how far into a cave the runner is, so the light goes over a few
metres inside the mouth and comes back the same way at the far end.
"""
from dataclasses import dataclass

from openglcontext_hill_runner import course as c

#: Metres inside the end of a cave over which the light goes out, or comes back.
LIGHT_FADE = 28.0
#: What is left of the environment light and of the sun in the dark of a cave.
CAVE_ENVIRONMENT = 0.045
CAVE_SUN = 0.2
#: The fog of a cave: near black, and the distance at which it hides everything.
CAVE_FOG = (0.012, 0.009, 0.007)
CAVE_RANGE = 150.0


@dataclass(frozen=True)
class Air:
    """How the air is at one place: the light it lets in and the fog in it."""

    #: Multiplies the environment light, and the sun's intensity; 1 in the open.
    ibl: float
    sun: float
    fog_colour: tuple[float, float, float]
    fog_range: float


def darkness(course: c.Course, distance: float) -> float:
    """How dark it is at ``distance``, from 0 in the open to 1 deep in a cave."""
    cave = course.cave_at(distance)
    if cave is None:
        return 0.0
    into = min(distance - cave.start, cave.end - distance) / LIGHT_FADE
    return float(c.smooth(into))


def _mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def air_at(course: c.Course, distance: float, haze: tuple[float, float, float],
           range_: float) -> Air:
    """The air at ``distance``: the open air's ``haze`` and ``range_`` of fog, darkened by a cave."""
    dark = darkness(course, distance)
    return Air(_mix(1.0, CAVE_ENVIRONMENT, dark), _mix(1.0, CAVE_SUN, dark),
               (_mix(haze[0], CAVE_FOG[0], dark), _mix(haze[1], CAVE_FOG[1], dark),
                _mix(haze[2], CAVE_FOG[2], dark)),
               _mix(range_, CAVE_RANGE, dark))
