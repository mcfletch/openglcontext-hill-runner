"""The caves: where they are, how high the ceiling is, and what the generator leaves out of them."""
import numpy as np
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner.events import Event

FRAME = 1 / 60


@pytest.fixture(scope='module')
def world():
    return c.Course(1)


def first_cave(world):
    return world.caves_between(0.0, 20000.0)[0]


# -- where they are ---------------------------------------------------------------------

def test_a_cave_is_where_the_seed_puts_it_and_is_the_same_every_time(world):
    assert c.Course(1).caves_between(0.0, 30000.0) == world.caves_between(0.0, 30000.0)
    assert c.Course(2).caves_between(0.0, 30000.0) != world.caves_between(0.0, 30000.0)


def test_no_cave_is_near_the_start_and_they_are_a_few_kilometres_apart(world):
    caves = world.caves_between(0.0, 60000.0)
    assert len(caves) >= 10
    assert caves[0].start >= c.CAVE_FIRST
    for before, after in zip(caves, caves[1:], strict=False):
        assert after.start - before.end >= 1000.0


def test_a_cave_is_a_few_hundred_metres_long(world):
    for cave in world.caves_between(0.0, 60000.0):
        assert c.CAVE_LENGTH[0] <= cave.end - cave.start <= c.CAVE_LENGTH[1]


def test_a_cave_comes_out_into_rock_or_into_forest(world):
    exits = {cave.exit for cave in world.caves_between(0.0, 200000.0)}
    assert exits == {'rocks', 'forest'}


def test_the_cave_at_a_distance_is_the_one_it_is_in(world):
    cave = first_cave(world)
    assert world.cave_at(cave.start - 1.0) is None
    assert world.cave_at(cave.start) == cave
    assert world.cave_at((cave.start + cave.end) / 2) == cave
    assert world.cave_at(cave.end + 1.0) is None


def test_no_fork_runs_in_or_near_a_cave(world):
    for cave in world.caves_between(0.0, 60000.0):
        near = world.forks_between(cave.start - c.CAVE_APPROACH, cave.end + c.CAVE_AFTER)
        assert near == [], (cave, near)


def test_forks_still_come_between_the_caves(world):
    assert len(world.forks_between(0.0, 30000.0)) >= 15


# -- the ceiling ------------------------------------------------------------------------

def test_there_is_no_ceiling_in_the_open(world):
    cave = first_cave(world)
    assert world.ceiling(cave.start - 5.0) == np.inf
    assert world.ceiling(cave.end + 5.0) == np.inf
    assert not world.in_cave(cave.start - 5.0) and world.in_cave(cave.start + 5.0)


def test_the_ceiling_is_out_of_reach_at_the_mouth_and_at_the_far_end(world):
    for cave in world.caves_between(0.0, 60000.0):
        for d in (cave.start + 1.0, cave.start + 10.0, cave.end - 10.0, cave.end - 1.0):
            assert world.ceiling(d) >= c.JUMP_NEEDS, (cave, d)


def test_the_ceiling_is_high_in_places_and_low_in_others(world):
    for cave in world.caves_between(0.0, 60000.0)[:12]:
        d = np.linspace(cave.start, cave.end, 800, endpoint=False)
        up = world.ceiling(d)
        assert (up >= c.JUMP_NEEDS).mean() > 0.35 and (up < c.JUMP_NEEDS).mean() > 0.15
        assert up.min() >= c.CEILING_LOW[0] - 1e-9 and up.max() <= c.CEILING_HIGH[1] + 1e-9


def test_a_low_ceiling_still_clears_a_runner_standing_and_sliding(world):
    low = min(world.ceiling(np.linspace(cave.start, cave.end, 800, endpoint=False)).min()
              for cave in world.caves_between(0.0, 60000.0))
    assert low > c.STAND_HEIGHT + 0.4


def test_the_ceiling_changes_smoothly(world):
    for cave in world.caves_between(0.0, 30000.0)[:8]:
        d = np.arange(cave.start, cave.end, 0.25)
        assert np.abs(np.diff(world.ceiling(d))).max() < 0.15


def test_low_ceiling_is_where_a_jump_would_hit(world):
    cave = first_cave(world)
    d = np.linspace(cave.start, cave.end, 400, endpoint=False)
    assert np.array_equal(world.low_ceiling(d), world.ceiling(d) < c.JUMP_NEEDS)
    assert c.JUMP_NEEDS > c.JUMP_PEAK + c.STAND_HEIGHT


def test_the_ground_near_a_jump_is_clear_where_no_ceiling_nearby_is_low(world):
    cave = first_cave(world)
    d = np.linspace(cave.start, cave.end, 600, endpoint=False)
    clear = np.array([world.jump_clear(float(v)) for v in d])
    assert clear.any() and not clear.all()
    for v, ok in zip(d, clear, strict=True):
        if ok:
            around = world.ceiling(np.linspace(v - c.JUMP_MARGIN, v + c.JUMP_MARGIN, 40))
            assert (around >= c.JUMP_NEEDS).all()


# -- what the generator leaves out ------------------------------------------------------

def cave_obstacles(world, count=8):
    found = []
    for cave in world.caves_between(0.0, 120000.0)[:count]:
        found += [(cave, o) for o in world.obstacles_between(cave.start, cave.end)]
    return found


def test_a_cave_holds_crevices(world):
    kinds = {o.kind for _, o in cave_obstacles(world)}
    assert c.Kind.CREVICE in kinds


def test_a_crevice_is_never_where_a_jump_would_hit_the_ceiling(world):
    crevices = [o for _, o in cave_obstacles(world, 24) if o.kind is c.Kind.CREVICE]
    assert len(crevices) > 10
    for o in crevices:
        assert world.jump_clear(o.distance), o


def test_nothing_that_needs_a_jump_is_under_a_low_ceiling(world):
    for _, o in cave_obstacles(world, 24):
        if o.kind in (c.Kind.LOW, c.Kind.CREVICE):
            assert world.jump_clear(o.distance), o


def test_a_crevice_runs_right_across_the_floor_so_it_has_to_be_jumped(world):
    for _, o in cave_obstacles(world, 24):
        if o.kind is c.Kind.CREVICE:
            assert (o.lane, o.span) == (0, c.LANE_COUNT)


def test_a_jewel_that_wants_a_jump_is_not_under_a_ceiling_that_hits_it(world):
    for cave in world.caves_between(0.0, 120000.0)[:24]:
        for coin in world.coins_between(cave.start, cave.end):
            assert coin.height <= world.ceiling(coin.distance) - 0.3, coin


def test_outside_the_caves_there_are_no_crevices(world):
    for index in range(0, 400):
        chunk = world.chunk(index)
        for o in chunk.obstacles:
            if o.kind is c.Kind.CREVICE:
                assert world.in_cave(o.distance)


def test_the_hill_outside_the_caves_is_made_as_it_always_was():
    assert c.Course(5).chunk(3) == c.Course(5).chunk(3)
    plain = c.Course(5)
    cave = plain.caves_between(0.0, 20000.0)[0]
    index = c.chunk_index(cave.start - 400.0)
    assert plain.chunk(index).obstacles == c.Course(5).chunk(index).obstacles


# -- the rules --------------------------------------------------------------------------

class Under(c.Course):
    """A course with one cave of a fixed ceiling and the obstacles a test puts in it."""

    def __init__(self, ceiling, obstacles=(), cave=(100.0, 400.0)):
        super().__init__(0)
        self.height = ceiling
        self.placed = list(obstacles)
        self._made = c.Cave(0, cave[0], cave[1], 'rocks')

    def cave_at(self, distance):
        return self._made if self._made.start <= distance < self._made.end else None

    def ceiling(self, distance):
        inside = self._made.start <= np.asarray(distance) < self._made.end
        return np.where(inside, self.height, np.inf) if np.ndim(distance) else (
            self.height if inside else np.inf)

    def _make(self, index):
        start = index * c.CHUNK_LENGTH
        found = tuple(o for o in self.placed if start <= o.distance < start + c.CHUNK_LENGTH)
        return c.Chunk(index, start, found, (), ())


def play(game, until, command=None):
    heard = []
    for frame in range(60 * 120):
        if command is not None:
            command(game, frame)
        heard += game.advance(FRAME)
        if until(game):
            return heard
    pytest.fail('the condition never came')


def test_a_jump_under_a_low_ceiling_is_a_hit():
    game = g.Game(Under(2.6))
    game.runner.distance = 150.0
    heard = play(game, lambda game: game.damage == 1, lambda game, frame: game.runner.jump())
    assert Event.HIT in heard
    assert game.crashed_into is None and game.state is g.State.DOWN


def test_running_under_a_low_ceiling_is_safe_and_so_is_a_slide():
    game = g.Game(Under(2.6))
    game.runner.distance = 150.0
    play(game, lambda game: game.runner.distance > 300.0)
    assert game.damage == 0
    game = g.Game(Under(2.6))
    game.runner.distance = 150.0
    play(game, lambda game: game.runner.distance > 300.0,
         lambda game, frame: game.runner.slide() if frame % 30 == 0 else None)
    assert game.damage == 0


def test_a_jump_under_a_high_ceiling_is_safe():
    game = g.Game(Under(5.0))
    game.runner.distance = 150.0
    play(game, lambda game: game.runner.distance > 300.0,
         lambda game, frame: game.runner.jump())
    assert game.damage == 0


def test_a_jump_just_before_a_low_ceiling_is_over_before_it_gets_there():
    game = g.Game(Under(2.6, cave=(160.0, 400.0)))
    game.runner.distance = 100.0
    game.runner.jump()
    play(game, lambda game: game.runner.distance > 200.0)
    assert game.damage == 0


def test_a_crevice_is_crossed_by_a_runner_who_jumps_and_is_a_hit_for_one_who_does_not():
    crevice = c.Obstacle(c.Kind.CREVICE, 0, 200.0, c.LANE_COUNT)
    walker = g.Game(Under(6.0, [crevice]))
    walker.runner.distance = 150.0
    heard = play(walker, lambda game: game.damage == 1)
    assert Event.HIT in heard and walker.crashed_into == crevice

    def jump_before(game, frame):
        if crevice.distance - game.runner.distance <= game.runner.speed * 0.25 and game.runner.grounded:
            game.runner.jump()
    jumper = g.Game(Under(6.0, [crevice]))
    jumper.runner.distance = 150.0
    play(jumper, lambda game: game.runner.distance > 260.0, jump_before)
    assert jumper.damage == 0


def test_a_crevice_is_not_cleared_by_a_slide_or_a_hop():
    crevice = c.Obstacle(c.Kind.CREVICE, 0, 200.0, c.LANE_COUNT)
    game = g.Game(Under(6.0, [crevice]))
    game.runner.distance = 150.0
    play(game, lambda game: game.damage == 1,
         lambda game, frame: game.runner.hop(+1) if frame == 5 else None)
    assert game.crashed_into == crevice
