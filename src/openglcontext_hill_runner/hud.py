"""The HUD: coins collected, score and metres run, in the engine's HUD layer."""
from OpenGLContext.ui.hudwidgets import HUDGroup, HUDLayer, Readout

from openglcontext_hill_runner.game import Game


class RunHud:
    """Readouts in the top corners, set from the game each frame."""

    def __init__(self) -> None:
        self.coins = Readout(label='COINS', value='0')
        self.score = Readout(label='SCORE', value='0')
        self.meters = Readout(label='METERS', value='0 m', anchor='top-right')
        self.group = HUDGroup(anchor='top-left', spacing=4,
                              children=[self.coins, self.score])
        self.layer = HUDLayer(children=[self.group, self.meters])

    def update(self, game: Game) -> None:
        """Set the readouts from the game."""
        self.coins.value = str(game.coins)
        self.score.value = str(game.points)
        self.meters.value = '%d m' % game.meters
