"""The HUD: coins in hand, score, hits, metres run and the red of a continue being paid for, in the
engine's HUD layer."""
from collections.abc import Iterable

from OpenGLContext.ui.hudwidgets import DamageIndicator, HUDGroup, HUDLayer, Readout

from openglcontext_hill_runner.events import Event
from openglcontext_hill_runner.game import Game

#: How solid the shading at the screen's edges is while a continue is being paid for, 0 to 1.
SHADING = 0.75
#: How far in from the edge it reaches, in pixels at the reference font size, and how long the
#: flash of a new hit lasts, in seconds.
REACH = 110.0
FLASH = 0.9
#: The strips the shading is drawn from: enough that none shows as a band at this width.
STEPS = 16
#: Times a second the coins flash red while they count down.
COUNTDOWN_FLASHES = 5.0


class RunHud:
    """Readouts in the top corners and shading round the edges, set from the game each frame."""

    def __init__(self) -> None:
        self.coins = Readout(label='COINS', value='0')
        self.score = Readout(label='SCORE', value='0')
        self.hits = Readout(label='HITS', value='0')
        self.meters = Readout(label='METERS', value='0 m', anchor='top-right')
        #: Shades all four edges red while a continue is being paid for, and flashes them at each
        #: hit.
        self.damage = DamageIndicator(strength=SHADING, thickness=REACH, duration=FLASH,
                                       steps=STEPS)
        self.group = HUDGroup(anchor='top-left', spacing=4,
                              children=[self.coins, self.score, self.hits])
        # The shading is first so that it is drawn under the readouts.
        self.layer = HUDLayer(children=[self.damage, self.group, self.meters])

    def update(self, game: Game, events: Iterable[Event] = (), now: float | None = None) -> None:
        """Set the readouts and the shading from the game; flash if ``events`` has a hit in it.
        The coins flash red while they count down and the shading is whole until they have;
        ``now`` is the time in seconds the flashing is counted from."""
        self.coins.value = str(game.coins)
        self.score.value = str(game.points)
        self.hits.value = str(game.hits)
        self.meters.value = '%d m' % game.meters
        self.damage.level = 1.0 if game.paying else 0.0
        self.coins.critical = game.paying and int((now or 0.0) * 2.0 * COUNTDOWN_FLASHES) % 2 == 0
        if Event.HIT in events:
            self.damage.hurt_all(now=now)

    def reset(self) -> None:
        """Clear the shading and any flash left from the run before."""
        self.damage.level = 0.0
        self.damage.clear()
        self.coins.critical = False
