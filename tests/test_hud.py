"""The HUD: coins in hand, metres run and a continue being paid for."""
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner.events import Event
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


def test_the_edges_are_red_while_a_continue_is_paid_for_and_clear_after(game):
    hud = RunHud()
    hud.update(game)
    assert hud.damage.level == 0.0
    game.owed = g.CONTINUE_COST
    hud.update(game)
    assert hud.damage.level == 1.0
    game.owed = 0
    hud.update(game)
    assert hud.damage.level == 0.0


def test_the_coins_flash_red_while_they_count_down_and_only_then(game):
    hud = RunHud()
    game.owed = 40
    states = []
    for now in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
        hud.update(game, now=now)
        states.append(hud.coins.critical)
    assert True in states and False in states
    game.owed = 0
    hud.update(game, now=0.0)
    assert hud.coins.critical is False


def test_the_hits_are_read_out_beside_the_score(game):
    hud = RunHud()
    game.hits = 2
    hud.update(game)
    assert hud.hits.value == '2'
    assert hud.hits in hud.group.children


def test_the_shading_is_drawn_beneath_the_readouts():
    hud = RunHud()
    assert hud.layer.children[0] is hud.damage


def test_a_hit_flashes_every_edge_over_the_steady_shading(game):
    hud = RunHud()
    game.owed = g.CONTINUE_COST
    hud.update(game, [Event.HIT], now=10.0)
    hud.damage.tick(10.0)
    assert len(hud.damage.marks) == 4
    assert set(hud.damage.shares()) == {'top', 'right', 'bottom', 'left'}
    hud.damage.tick(20.0)
    assert hud.damage.marks == []
    assert hud.damage.shares()['top'] == pytest.approx(1.0)


def test_a_frame_with_no_hit_does_not_flash(game):
    hud = RunHud()
    hud.update(game, [Event.COIN, Event.JUMP], now=10.0)
    assert hud.damage.marks == []


def test_a_new_run_starts_unshaded(game):
    hud = RunHud()
    game.owed = g.CONTINUE_COST
    hud.update(game, [Event.HIT], now=1.0)
    game.restart()
    hud.update(game)
    hud.reset()
    assert hud.damage.level == 0.0 and hud.damage.marks == []
