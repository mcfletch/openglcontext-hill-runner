"""The treasure on the hill: which coins to show, and the nodes that show them."""
import numpy as np
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner import treasure as tr

FRAME = 1 / 60


@pytest.fixture
def game():
    return g.Game(c.Course(3))


def test_the_coins_shown_lie_between_a_little_behind_and_the_view_ahead(game):
    game.runner.distance = 300.0
    shown = tr.coins_in_view(game)
    assert shown
    for coin in shown:
        assert 300.0 - tr.BEHIND <= coin.distance < 300.0 + tr.AHEAD


def test_a_coin_that_has_been_taken_is_no_longer_shown(game):
    game.runner.distance = 100.0
    first = tr.coins_in_view(game)[0]
    game._taken.add(first.ident)
    assert first not in tr.coins_in_view(game)


@pytest.fixture
def field():
    return tr.TreasureField()


def test_the_field_holds_a_node_for_each_coin_shown(field, game):
    game.runner.distance = 300.0
    field.update(game, 0.0)
    assert len(field.children) == len(tr.coins_in_view(game))


def test_a_coin_keeps_its_node_while_it_stays_in_view(field, game):
    game.runner.distance = 300.0
    field.update(game, 0.0)
    before = {id(node): node for node in field.children}
    game.runner.distance = 302.0
    field.update(game, 0.1)
    kept = [node for node in field.children if id(node) in before]
    assert len(kept) > len(field.children) // 2


def test_coins_leave_the_field_as_the_runner_passes_and_new_ones_arrive(field, game):
    game.runner.distance = 300.0
    field.update(game, 0.0)
    game.runner.distance = 700.0
    field.update(game, 0.0)
    positions = [-node.translation[2] for node in field.children]
    assert all(700.0 - tr.BEHIND - 1 <= d < 700.0 + tr.AHEAD + 1 for d in positions)


def test_a_coin_sits_in_its_lane_at_its_height_above_the_ground(field, game):
    game.runner.distance = 300.0
    field.update(game, 0.0)
    coin = tr.coins_in_view(game)[0]
    node = next(n for n in field.children if abs(-n.translation[2] - coin.distance) < 1e-3)
    assert node.translation[0] == pytest.approx(
        game.course.lane_x(coin.lane, coin.distance))
    ground = c.ground_height(coin.distance)
    assert node.translation[1] == pytest.approx(ground + coin.height, abs=tr.BOB + 1e-6)


def test_coins_turn_as_time_passes(field, game):
    game.runner.distance = 300.0
    field.update(game, 0.0)
    node = field.children[0]
    first = node.rotation[3]
    field.update(game, 0.25)
    assert node.rotation[3] != first


def test_a_jewel_is_drawn_differently_from_a_coin(field, game):
    jewels = [k for i in range(60) for k in game.course.chunk(i).coins if k.value > 1]
    coins = [k for i in range(60) for k in game.course.chunk(i).coins if k.value == 1]
    assert field.shape_for(jewels[0]) is not field.shape_for(coins[0])
    assert field.shape_for(coins[0]) is field.shape_for(coins[1])
    assert field.shape_for(jewels[0]) is field.shape_for(jewels[1])


def test_a_coin_is_plate_sized_and_a_jewel_fits_in_a_hand():
    coin = tr.coin_geometry().positions
    jewel = tr.jewel_geometry().positions
    assert 0.2 <= np.ptp(coin[:, 1]) <= 0.3
    assert 0.12 <= np.ptp(jewel[:, 1]) <= 0.4 and np.ptp(jewel[:, 0]) <= 0.4


def test_a_jewel_is_cut_in_flat_facets_and_a_coin_is_textured():
    jewel = tr.jewel_geometry()
    facets = jewel.indices.reshape(-1, 3)
    assert len(facets) >= 16
    for tri in facets[:: max(1, len(facets) // 8)]:
        assert np.allclose(jewel.normals[tri[0]], jewel.normals[tri[1]], atol=1e-4)
    face = tr.coin_geometry()
    assert np.ptp(face.texcoords[:, 0]) > 0.5 and np.ptp(face.texcoords[:, 1]) > 0.5


def test_the_coin_texture_has_a_raised_rim_and_a_design():
    from openglcontext_hill_runner.look import coin_maps
    maps = coin_maps(128)
    centre, rim, plain = maps.height[64, 64], maps.height[64, 4], maps.height[64, 40]
    assert rim > plain + 0.1
    assert np.ptp(maps.height[40:90, 40:90]) > 0.15 and centre != plain
    assert (maps.metallic >= 0.9).all()
