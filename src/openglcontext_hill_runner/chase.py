"""The point the camera follows: the runner, a little ahead, with sideways moves softened. No GL."""
import math

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


class Chase:
    """Works out where the camera looks, from the runner, each frame."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self._x = runner.x
        self._distance = runner.distance

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
        return (float(self._x + along),
                float(c.ground_height(ahead)) + HEIGHT + JUMP_LIFT * runner.height,
                -ahead)

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
        return (-CAMERA_BACK * math.sin(turn), CAMERA_UP, CAMERA_BACK * math.cos(turn))
