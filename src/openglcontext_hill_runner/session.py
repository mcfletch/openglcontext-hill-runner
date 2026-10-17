"""One run, advanced a frame at a time: the rules, the pilot, the cues and the camera's target.

The window's frame is this plus drawing: it gives :meth:`Session.step` the time
that passed and applies what comes back to the model, the mixer and the camera.
No GL.
"""
from dataclasses import dataclass, field

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import cues
from openglcontext_hill_runner.chase import Chase
from openglcontext_hill_runner.game import Game, State
from openglcontext_hill_runner.pilot import Pilot

#: Seconds the fall is shown before the end-of-run screen.
GAME_OVER_DELAY = 1.4
#: How hard a footfall sounds.
FOOTSTEP_GAIN = 0.6


@dataclass
class Frame:
    """What one step of a session asks the window to show and play."""

    pose: cues.Pose
    sounds: list[cues.Sound]
    #: Where the camera looks.
    target: tuple[float, float, float]
    events: list = field(default_factory=list)


class Session:
    """A run of one course, played by the keys or, with ``demo``, by the pilot."""

    def __init__(self, course: c.Course, demo: bool = False) -> None:
        self.course = course
        self.game = Game(course)
        self.chase = Chase(self.game.runner)
        self.footsteps = cues.Footsteps()
        self.pilot = Pilot(self.game) if demo else None
        self._crashed_for = 0.0

    @property
    def over(self) -> bool:
        """Whether the fall has been on screen long enough for the end-of-run screen."""
        return self._crashed_for >= GAME_OVER_DELAY

    def step(self, dt: float) -> Frame:
        """Run for ``dt`` seconds."""
        game = self.game
        if self.pilot is not None:
            self.pilot.drive()
        events = game.advance(dt)
        # The runner's own commands (a jump asked for between frames) are heard
        # with the frame that carries them out.
        sounds = cues.sounds_for(events)
        if self.footsteps.advance(dt, game):
            sounds.append(cues.Sound(cues.STEP, FOOTSTEP_GAIN))
        if game.state is State.CRASHED:
            self._crashed_for += dt
        return Frame(cues.pose_for(game), sounds, self.chase.advance(dt), events)
