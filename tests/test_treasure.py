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


def test_a_new_run_shows_its_own_coins_not_those_the_last_one_left_at_the_same_places():
    field = tr.TreasureField()
    old = g.Game(c.Course(3))
    old.runner.distance = 300.0
    field.update(old, 0.0)
    new = g.Game(c.Course(4))
    new.runner.distance = 300.0
    field.update(new, 0.0)
    wanted = sorted(tr.coins_in_view(new), key=lambda coin: coin.distance)
    shown = sorted(field.children, key=lambda node: -node.translation[2])
    assert len(shown) == len(wanted)
    for node, coin in zip(shown, wanted, strict=True):
        assert -node.translation[2] == pytest.approx(coin.distance, abs=1e-3)
        assert node.translation[0] == pytest.approx(
            new.course.lane_x(coin.lane, coin.distance), abs=1e-3)


def _nodes_by_distance(field):
    return sorted(field.children, key=lambda node: -node.translation[2])


def _spin(field, game, steps, start=0.0):
    """The rotation of every node after each of ``steps`` updates, a frame apart."""
    turns = []
    for step in range(steps):
        field.update(game, start + step * FRAME)
        turns.append({node: node.rotation[3] for node in field.children})
    return turns


def test_a_coin_is_placed_once_however_many_updates_follow(field, game, monkeypatch):
    game.runner.distance = 300.0
    field.update(game, 0.0)
    asked = []
    original = game.course.lane_x
    monkeypatch.setattr(game.course, 'lane_x', lambda *a: asked.append(a) or original(*a))
    for step in range(1, 6):
        field.update(game, step * FRAME)
    assert asked == []


def test_a_coin_near_the_runner_turns_on_every_update(field, game):
    game.runner.distance = 300.0
    turns = _spin(field, game, 6)
    near = [node for node in field.children
            if -node.translation[2] < game.runner.distance + tr.ANIMATE_NEAR]
    assert near
    for before, after in zip(turns, turns[1:], strict=False):
        assert all(before[node] != after[node] for node in near)


def test_a_coin_far_ahead_turns_on_some_updates_and_every_few(field, game):
    game.runner.distance = 300.0
    turns = _spin(field, game, 3 * tr.FAR_EVERY + 1)
    far = [node for node in field.children
           if -node.translation[2] > game.runner.distance + tr.ANIMATE_NEAR]
    assert far
    for node in far:
        changes = sum(before[node] != after[node] for before, after in zip(turns, turns[1:], strict=False))
        assert 0 < changes <= 3 * tr.FAR_EVERY // tr.FAR_EVERY + 1


def test_every_coin_is_turned_within_a_few_updates(field, game):
    game.runner.distance = 300.0
    turns = _spin(field, game, tr.FAR_EVERY + 1)
    for node in field.children:
        assert len({turn[node] for turn in turns}) > 1


def test_a_coin_coming_into_view_is_turned_and_placed_at_once(field, game):
    game.runner.distance = 300.0
    field.update(game, 0.0)
    known = set(field.children)
    game.runner.distance = 330.0
    field.update(game, 1.0)
    arrived = [node for node in field.children if node not in known]
    assert arrived
    for node in arrived:
        assert node.rotation[3] != 0.0 and node.translation[0] != 0.0
