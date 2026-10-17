"""The HUD: coins collected and metres run."""
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner.hud import RunHud


@pytest.fixture
def game():
    return g.Game(c.Course(1))


def test_the_hud_starts_at_nothing_collected_and_nothing_run(game):
    hud = RunHud()
    hud.update(game)
    assert hud.coins.value == '0'
    assert hud.meters.value == '0 m'
    assert hud.score.value == '0'


def test_the_hud_reads_the_game(game):
    game.coins, game.points = 12, 31
    game.runner.distance = 845.7
    hud = RunHud()
    hud.update(game)
    assert (hud.coins.value, hud.score.value, hud.meters.value) == ('12', '31', '845 m')


def test_the_layers_corners_hold_coins_at_the_left_and_metres_at_the_right():
    hud = RunHud()
    assert hud.meters.anchor == 'top-right'
    assert hud.group.anchor == 'top-left'
    assert hud.coins in hud.group.children
