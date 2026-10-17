"""One run, advanced a frame at a time: the rules, the pilot, the cues, where the runner
stands and where the camera is. No GL.

The window's frame is this plus drawing: it gives :meth:`Session.step` the time that passed
and copies what comes back onto the model, the mixer and the camera.
"""
import math
from dataclasses import dataclass, field

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import cues
from openglcontext_hill_runner.chase import Chase
from openglcontext_hill_runner.game import Game, State
from openglcontext_hill_runner.landscape import Landscape
from openglcontext_hill_runner.pilot import Pilot
from openglcontext_hill_runner.runner import HOP_TIME

#: Seconds the fall is shown before the end-of-run screen.
GAME_OVER_DELAY = 1.4
#: How hard a footfall sounds.
FOOTSTEP_GAIN = 0.6
#: The longest time a frame may run for, in seconds: a stall in the window does not throw the
#: runner down the hill.
MAX_FRAME = 0.05
#: How much of the hill's slope the runner tips into, and how far it leans into a hop, in radians.
SLOPE_LEAN = 0.6
HOP_LEAN = 0.35


@dataclass
class Frame:
    """What one step of a session asks the window to show and play."""

    #: The seconds the step ran for.
    dt: float
    pose: cues.Pose
    sounds: list[cues.Sound]
    #: Where the camera looks, and where it stands relative to that.
    target: tuple[float, float, float]
    camera_offset: tuple[float, float, float]
    #: Where the runner stands in the world, and how it is turned: to the right of straight down
    #: the hill by ``heading``, tipped by ``pitch`` to the slope, rolled by ``lean`` into a hop
    #: (all in radians).
    position: tuple[float, float, float]
    heading: float
    pitch: float
    lean: float
    events: list = field(default_factory=list)


class Session:
    """A run on a landscape, played by the keys or, with ``demo``, by the pilot."""

    def __init__(self, land: Landscape, demo: bool = False) -> None:
        self.land = land
        self.course = land.course
        self.game = Game(self.course)
        self.chase = Chase(self.game.runner)
        self.footsteps = cues.Footsteps()
        self.pilot = Pilot(self.game) if demo else None
        self._crashed_for = 0.0
        self._last_x = self.game.runner.x

    @property
    def demo(self) -> bool:
        """Whether the pilot is playing. Set it to hand the run to the pilot or take it back."""
        return self.pilot is not None

    @demo.setter
    def demo(self, playing: bool) -> None:
        if playing and self.pilot is None:
            self.pilot = Pilot(self.game)
        elif not playing:
            self.pilot = None

    @property
    def over(self) -> bool:
        """Whether the fall has been on screen long enough for the end-of-run screen."""
        return self._crashed_for >= GAME_OVER_DELAY

    def step(self, dt: float) -> Frame:
        """Run for ``dt`` seconds, at most :data:`MAX_FRAME`."""
        dt = min(dt, MAX_FRAME)
        game = self.game
        if self.pilot is not None and game.state is State.RUNNING:
            self.pilot.drive()
        events = game.advance(dt)
        # The runner's own commands (a jump asked for between frames) are heard
        # with the frame that carries them out.
        sounds = cues.sounds_for(events)
        if self.footsteps.advance(dt, game):
            sounds.append(cues.Sound(cues.STEP, FOOTSTEP_GAIN))
        if game.state is State.CRASHED:
            self._crashed_for += dt
        runner = game.runner
        d = runner.distance
        fall = (c.ground_height(d - 1.0) - c.ground_height(d + 1.0)) / 2.0
        sideways = (runner.x - self._last_x) / dt if dt > 0 else 0.0
        self._last_x = runner.x
        lean = -HOP_LEAN * max(-1.0, min(1.0, sideways / (c.LANE_WIDTH / HOP_TIME)))
        return Frame(
            dt, cues.pose_for(game), sounds, self.chase.advance(dt), self.chase.camera_offset(),
            (runner.x, float(self.land.height(runner.x, d)) + runner.height, -d),
            self.chase.heading(), -math.atan(fall) * SLOPE_LEAN, lean, events)
