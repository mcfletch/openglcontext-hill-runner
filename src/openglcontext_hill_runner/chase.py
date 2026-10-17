"""The point the camera follows: the runner, a little ahead, with sideways moves softened. No GL."""
import math

import numpy as np

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner.runner import Runner

#: Metres ahead of the runner that the camera looks, and above the ground.
LOOK_AHEAD = 3.0
HEIGHT = 1.1
#: How quickly the view eases toward the runner after a hop, per second.
SWAY_RATE = 7.0
#: The part of a jump's height that lifts the view.
JUMP_LIFT = 0.5
#: How far ahead the trail's heading is measured, in metres.
HEADING_AHEAD = 6.0
#: The camera stands this far behind the point it looks at along the runner's path, and this
#: far above it, in metres: above head height, with the runner's feet at the bottom of the frame.
CAMERA_BACK = 5.1
CAMERA_UP = 2.5
#: Under a ceiling the camera stands this far below it at most, and no lower than MIN_UP above the
#: point it looks at; it sinks the moment the ceiling comes down and rises again at RISE_RATE of
#: its own height a second. The ceiling is looked for from the camera to AHEAD_OF_CAMERA metres
#: before it.
CEILING_CLEARANCE = 0.75
MIN_UP = 0.0
RISE_RATE = 3.0
AHEAD_OF_CAMERA = 2.5


class Chase:
    """Works out where the camera looks, from the runner, each frame."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self._x = runner.x
        self._distance = runner.distance
        self._up = CAMERA_UP

    def advance(self, dt: float) -> tuple[float, float, float]:
        """Move ``dt`` seconds on; returns the point to look at.

        The view is carried along with the path the runner is on, wherever it wanders or draws
        away to, and eases toward the runner for a hop.
        """
        runner = self.runner
        course = runner.course
        self._x += (course.lane_x(runner.lane, runner.distance)
                    - course.lane_x(runner.lane, self._distance))
        self._distance = runner.distance
        self._x += (runner.x - self._x) * (1.0 - math.exp(-SWAY_RATE * dt))
        ahead = runner.distance + LOOK_AHEAD
        along = course.lane_x(runner.lane, ahead) - course.lane_x(runner.lane, runner.distance)
        height = float(c.ground_height(ahead)) + HEIGHT + JUMP_LIFT * runner.height
        room = self._room(height)
        self._up = room if room < self._up else self._up + (room - self._up) * (
            1.0 - math.exp(-RISE_RATE * dt))
        return (float(self._x + along), height, -ahead)

    def _room(self, looking_at: float) -> float:
        """How far above ``looking_at`` the camera may stand: as high as it likes in the open,
        and under a ceiling clear of the lowest of it between the camera and a little past it."""
        runner = self.runner
        first = runner.distance + LOOK_AHEAD - CAMERA_BACK
        spots = first + np.linspace(0.0, AHEAD_OF_CAMERA, 9)
        roof = np.asarray(runner.course.ceiling(spots))
        if not np.isfinite(roof).any():
            return CAMERA_UP
        lowest = float((np.asarray(c.ground_height(spots)) + roof).min())
        return max(MIN_UP, min(CAMERA_UP, lowest - CEILING_CLEARANCE - looking_at))

    def heading(self) -> float:
        """Which way the runner's path runs, in radians turned to the right of straight down the hill."""
        runner = self.runner
        course, d = runner.course, runner.distance
        return math.atan((course.lane_x(runner.lane, d + HEADING_AHEAD)
                          - course.lane_x(runner.lane, d)) / HEADING_AHEAD)

    def camera_offset(self) -> tuple[float, float, float]:
        """Where the camera stands, from the point it looks at: behind it along the runner's
        path."""
        turn = self.heading()
        return (-CAMERA_BACK * math.sin(turn), self._up, CAMERA_BACK * math.cos(turn))
