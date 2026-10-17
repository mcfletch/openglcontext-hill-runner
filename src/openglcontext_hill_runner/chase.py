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
#: point it looks at. It glides to the height the ceiling ahead allows, a critically damped move
#: that takes SMOOTH_TIME seconds to cover most of the way, and it starts as soon as the lowest
#: ceiling within LEAD_TIME seconds of running ahead of the camera comes down, and no later than
#: the ceiling reaches it: it never stands higher than the ceiling over the next AHEAD_OF_CAMERA
#: metres allows.
CEILING_CLEARANCE = 0.75
MIN_UP = 0.0
SMOOTH_TIME = 0.45
LEAD_TIME = 0.9
AHEAD_OF_CAMERA = 2.5
#: Where the camera has sunk to head height behind the runner, the runner fills the view along the
#: floor. It then stands SIDE_STEP metres to the right of the runner's path and looks LOOK_ASIDE
#: metres to the left of where it would look, across the room and not along the runner's back, by
#: as much as it has sunk; it has sunk fully when it is SIDE_SINKING metres below where it stands
#: in the open.
SIDE_STEP = 1.1
LOOK_ASIDE = 1.2
SIDE_SINKING = 0.8


class Chase:
    """Works out where the camera looks, from the runner, each frame."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self._x = runner.x
        self._distance = runner.distance
        self._up = CAMERA_UP
        self._rising = 0.0

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
        self._glide(self._room(height, max(AHEAD_OF_CAMERA, runner.speed * LEAD_TIME)),
                    self._room(height, AHEAD_OF_CAMERA), dt)
        return (float(self._x + along) - LOOK_ASIDE * self.sunk(), height, -ahead)

    def _glide(self, wanted: float, limit: float, dt: float) -> None:
        """Move the camera's height toward ``wanted`` without ever standing above ``limit``."""
        omega = 2.0 / SMOOTH_TIME
        x = omega * dt
        damping = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        change = self._up - wanted
        carried = (self._rising + omega * change) * dt
        self._rising = (self._rising - omega * carried) * damping
        self._up = wanted + (change + carried) * damping
        if self._up > limit:
            self._up, self._rising = limit, min(self._rising, 0.0)

    def _room(self, looking_at: float, reach: float) -> float:
        """How far above ``looking_at`` the camera may stand: as high as it likes in the open,
        and under a ceiling clear of the lowest of it from the camera to ``reach`` metres past
        it."""
        runner = self.runner
        first = runner.distance + LOOK_AHEAD - CAMERA_BACK
        spots = first + np.linspace(0.0, reach, max(9, int(reach / 0.5) + 1))
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
        path, and to the right of the path by SIDE_STEP once sunk. The point looked at has moved
        LOOK_ASIDE to the left, which the offset also covers."""
        turn = self.heading()
        return (-CAMERA_BACK * math.sin(turn) + (SIDE_STEP + LOOK_ASIDE) * self.sunk(), self._up,
                CAMERA_BACK * math.cos(turn))

    def sunk(self) -> float:
        """How far the camera has sunk under a ceiling, from 0 in the open to 1 at SIDE_SINKING
        metres down."""
        return min(1.0, max(0.0, (CAMERA_UP - self._up) / SIDE_SINKING))
