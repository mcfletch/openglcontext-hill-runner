"""The course: lanes, the hill, and what is on it, as pure data."""
import numpy as np
import pytest

from openglcontext_hill_runner import course as c


def chunks(seed=1, count=40):
    world = c.Course(seed)
    return [world.chunk(i) for i in range(count)]


def test_the_lanes_are_centred_on_the_middle_path():
    xs = [c.lane_offset(lane) for lane in range(c.LANE_COUNT)]
    assert xs == [-c.LANE_WIDTH, 0.0, c.LANE_WIDTH]


def test_the_ground_runs_downhill_all_the_way():
    heights = [c.ground_height(d) for d in range(0, 5000, 5)]
    assert all(b < a for a, b in zip(heights, heights[1:], strict=False))


def test_the_ground_never_steepens_past_the_slope_limit():
    step = 0.5
    slopes = [(c.ground_height(d) - c.ground_height(d + step)) / step
              for d in [i * step for i in range(4000)]]
    assert min(slopes) > 0.05
    assert max(slopes) < c.MAX_SLOPE


def test_a_chunk_is_the_same_every_time_it_is_asked_for():
    world = c.Course(7)
    assert world.chunk(5) == world.chunk(5)
    assert c.Course(7).chunk(5) == world.chunk(5)


def test_another_seed_makes_another_hill():
    assert c.Course(1).chunk(6) != c.Course(2).chunk(6)


def test_a_chunk_holds_only_what_lies_in_it():
    for chunk in chunks():
        start, end = chunk.start, chunk.start + c.CHUNK_LENGTH
        for item in [*chunk.obstacles, *chunk.coins]:
            assert start <= item.distance < end


def test_the_run_starts_clear_of_obstacles():
    world = c.Course(3)
    early = [*world.chunk(0).obstacles, *world.chunk(1).obstacles]
    assert early and all(o.distance >= c.SAFE_START for o in early)
    assert world.chunk(0).obstacles == ()


def test_no_row_is_walled_off_across_every_path():
    world = c.Course(1)
    for chunk in (world.chunk(i) for i in range(160)):
        rows = {}
        for obstacle in chunk.obstacles:
            if obstacle.kind is c.Kind.WALL:
                rows.setdefault(obstacle.distance, set()).update(obstacle.lanes)
        for distance, lanes in rows.items():
            assert len(lanes) < len(world.lanes_at(distance))


def test_an_obstacle_may_stretch_across_two_paths_and_is_seen_across_both():
    found = {(o.kind, o.span) for chunk in chunks(count=120) for o in chunk.obstacles}
    for kind in c.Kind:
        if kind is not c.Kind.CREVICE:          # which is across the whole floor: see test_caves
            assert (kind, 1) in found and (kind, 2) in found, kind
    wide = next(o for chunk in chunks(count=120) for o in chunk.obstacles if o.span == 2)
    assert list(wide.lanes) == [wide.lane, wide.lane + 1]



def test_dividers_stand_only_where_no_wall_needs_a_hop():
    world = c.Course(11)
    found = False
    for i in range(120):
        chunk = world.chunk(i)
        for divider in chunk.dividers:
            found = True
            walls = [o for o in chunk.obstacles if o.kind is c.Kind.WALL
                     and divider.start <= o.distance <= divider.end]
            assert walls == []
    assert found, 'no chunk of 120 had a divider'


def test_a_divider_blocks_a_hop_between_its_two_lanes_only_inside_its_span():
    world = c.Course(11)
    divider = next(d for i in range(120) for d in world.chunk(i).dividers)
    low, high = divider.lane, divider.lane + 1
    inside = (divider.start + divider.end) / 2
    assert world.blocks_hop(low, high, inside, inside + 2)
    assert world.blocks_hop(high, low, inside, inside + 2)
    assert not world.blocks_hop(low, high, divider.end + 1, divider.end + 3)
    assert not world.blocks_hop(low, high, divider.start - 5, divider.start - 3)


def test_a_divider_blocks_only_the_pair_of_paths_it_stands_between():
    world = c.Course(11)
    for i in range(120):
        chunk = world.chunk(i)
        if len(chunk.dividers) == 1:
            divider = chunk.dividers[0]
            middle = (divider.start + divider.end) / 2
            other = 1 - divider.lane
            assert not world.blocks_hop(other, other + 1, middle, middle + 2)
            return
    pytest.fail('no chunk of 120 had exactly one divider')


def test_coins_float_above_the_ground_at_reachable_heights():
    for chunk in chunks():
        for coin in chunk.coins:
            assert 0.4 <= coin.height <= c.MAX_COIN_HEIGHT


def test_no_coin_hides_an_obstacle_from_the_runner_behind_it():
    for chunk in chunks(count=120):
        for coin in chunk.coins:
            for o in chunk.obstacles:
                if coin.lane in o.lanes:
                    assert not (o.distance - o.depth / 2 - c.COIN_CLEAR_BEFORE
                                <= coin.distance <= o.distance + o.depth / 2 + c.COIN_CLEAR_AFTER)


def test_a_jewel_hangs_where_only_a_jump_reaches_it():
    jewels = [k for chunk in chunks(count=120) for k in chunk.coins if k.value > 1]
    assert len(jewels) > 10
    assert all(k.height == c.JEWEL_HEIGHT for k in jewels)
    assert c.JEWEL_HEIGHT - c.COIN_HALF_HEIGHT > c.STAND_HEIGHT
    assert c.JEWEL_HEIGHT + c.COIN_HALF_HEIGHT < c.JUMP_PEAK + c.STAND_HEIGHT


def test_a_coin_has_an_identity_the_game_can_remember():
    ids = [coin.ident for chunk in chunks() for coin in chunk.coins]
    assert len(ids) == len(set(ids)) > 100


def test_jewels_are_worth_more_than_coins():
    coins = [coin for chunk in chunks(count=80) for coin in chunk.coins]
    assert {coin.value for coin in coins} >= {1, 5}


def test_obstacles_of_every_kind_turn_up():
    kinds = {o.kind for chunk in chunks(count=80) for o in chunk.obstacles}
    assert kinds == set(c.Kind)


def test_difficulty_rises_with_distance():
    world = c.Course(5)
    early = sum(len(world.chunk(i).obstacles) for i in range(2, 12))
    late = sum(len(world.chunk(i).obstacles) for i in range(60, 70))
    assert late > early


def test_a_chunk_index_is_found_from_a_distance():
    assert c.chunk_index(0.0) == 0
    assert c.chunk_index(c.CHUNK_LENGTH - 0.01) == 0
    assert c.chunk_index(c.CHUNK_LENGTH) == 1
    assert c.chunk_index(-5.0) == 0


@pytest.mark.parametrize('kind, band', [
    (c.Kind.LOW, (0.0, c.LOW_TOP)),
    (c.Kind.HIGH, (c.HIGH_BOTTOM, c.HIGH_TOP)),
    (c.Kind.WALL, (0.0, c.WALL_TOP)),
])
def test_each_kind_fills_its_own_band_of_the_air(kind, band):
    assert kind.band == band


def test_low_things_are_jumped_and_high_things_slid_under():
    assert c.LOW_TOP < c.JUMP_PEAK < c.HIGH_BOTTOM + c.JUMP_PEAK
    assert c.SLIDE_HEIGHT < c.HIGH_BOTTOM < c.STAND_HEIGHT


# -- the trails wander, and sometimes fork -------------------------------------------

def test_the_trail_starts_straight_and_wanders_gently_after():
    world = c.Course(5)
    d = np.arange(0.0, 6000.0, 2.0)
    centre = world.centre(d)
    assert abs(world.centre(0.0)) < 1e-9 and abs(centre[:20]).max() < 0.05
    assert abs(centre).max() > 3.0
    assert abs(centre).max() <= c.WANDER + 1e-6
    heading = np.abs(np.diff(centre) / 2.0)
    assert heading.max() < 0.12


def test_the_wander_belongs_to_the_seed():
    d = np.arange(300.0, 900.0, 7.0)
    assert np.array_equal(c.Course(5).centre(d), c.Course(5).centre(d))
    assert not np.allclose(c.Course(5).centre(d), c.Course(6).centre(d))


def test_the_paths_lie_a_lane_apart_either_side_of_the_trail():
    world = c.Course(1)
    for d in (0.0, 300.0, 3000.0):
        xs = [world.lane_x(lane, d) - world.centre(d) for lane in (0, 1, 2)]
        assert xs == pytest.approx([-c.LANE_WIDTH, 0.0, c.LANE_WIDTH])


def forks_of(world, count=12):
    return [f for f in world.forks_between(0.0, 12000.0)][:count]


def test_forks_leave_a_good_way_apart_and_run_a_few_hundred_metres():
    forks = forks_of(c.Course(3))
    assert len(forks) >= 6                      # a cave takes the place of a fork it would run into
    assert forks[0].start >= c.FORK_FIRST
    for fork in forks:
        assert c.FORK_LENGTH[0] <= fork.end - fork.start <= c.FORK_LENGTH[1]
        assert fork.side in (-1, 1)
    for a, b in zip(forks, forks[1:], strict=False):
        assert b.start - a.end >= 300.0


def test_forks_are_the_seeds_and_both_sides_turn_up():
    assert forks_of(c.Course(3)) == forks_of(c.Course(3))
    assert forks_of(c.Course(3)) != forks_of(c.Course(4))
    assert {f.side for f in forks_of(c.Course(3))} == {-1, 1}


def test_a_fork_adds_two_paths_beside_the_ones_there_were():
    world = c.Course(3)
    right = next(f for f in forks_of(world) if f.side == 1)
    left = next(f for f in forks_of(world) if f.side == -1)
    def middle(f):
        return (f.start + f.end) / 2

    assert world.lanes_at(right.start - 1.0) == (0, 1, 2)
    assert world.lanes_at(middle(right)) == (0, 1, 2, 3, 4)
    assert world.lanes_at(middle(left)) == (-2, -1, 0, 1, 2)
    assert world.lanes_at(right.end + 1.0) == (0, 1, 2)


def test_the_new_paths_start_beside_the_old_ones_and_draw_away_and_return():
    world = c.Course(3)
    fork = next(f for f in forks_of(world) if f.side == 1)
    def gap(d):
        return world.lane_x(3, d) - world.lane_x(2, d)

    assert gap(fork.start) == pytest.approx(c.LANE_WIDTH)
    assert gap((fork.start + fork.end) / 2) == pytest.approx(c.LANE_WIDTH + c.FORK_SEPARATION)
    assert gap(fork.end) == pytest.approx(c.LANE_WIDTH)
    left = next(f for f in forks_of(world) if f.side == -1)
    mid = (left.start + left.end) / 2
    assert world.lane_x(0, mid) - world.lane_x(-1, mid) == pytest.approx(
        c.LANE_WIDTH + c.FORK_SEPARATION)


def test_the_positions_of_every_path_at_many_distances_come_in_one_array():
    world = c.Course(3)
    fork = forks_of(world)[0]
    d = np.array([fork.start - 50.0, (fork.start + fork.end) / 2, fork.end + 50.0])
    positions = world.lane_positions(d)
    assert positions.shape == (len(c.ALL_LANES), 3)
    present = np.isfinite(positions)
    assert present.sum(axis=0).tolist() == [3, 5, 3]
    for column, distance in enumerate(d):
        for row, lane in enumerate(c.ALL_LANES):
            if present[row, column]:
                assert positions[row, column] == pytest.approx(world.lane_x(lane, distance))


def test_a_hop_toward_a_path_drawn_away_is_refused_and_one_at_the_start_is_not():
    world = c.Course(3)
    fork = next(f for f in forks_of(world) if f.side == 1)
    assert not world.blocks_hop(2, 3, fork.start + 2.0, fork.start + 6.0)
    middle = (fork.start + fork.end) / 2
    assert world.blocks_hop(2, 3, middle, middle + 4.0)


def test_nothing_stands_where_the_paths_part_or_join():
    world = c.Course(3)
    for fork in forks_of(world):
        for chunk in range(c.chunk_index(fork.start) - 1, c.chunk_index(fork.end) + 2):
            for o in world.chunk(chunk).obstacles:
                assert not (fork.start <= o.distance < fork.start + c.FORK_QUIET_AFTER_START), o
                assert not (fork.end - c.FORK_QUIET_BEFORE_END <= o.distance
                            < fork.end + c.FORK_QUIET_AFTER_END), o


def test_obstacles_and_coins_stand_only_on_paths_that_exist():
    world = c.Course(3)
    for chunk in (world.chunk(i) for i in range(120)):
        for o in chunk.obstacles:
            assert set(o.lanes) <= set(world.lanes_at(o.distance)), o
        for coin in chunk.coins:
            assert coin.lane in world.lanes_at(coin.distance), coin


def test_the_new_paths_have_things_to_run_through_too():
    world = c.Course(3)
    forks = forks_of(world, 8)
    on_fork_lanes = [o for fork in forks
                     for i in range(c.chunk_index(fork.start), c.chunk_index(fork.end) + 1)
                     for o in world.chunk(i).obstacles if o.lane > 2 or o.lane < 0]
    assert len(on_fork_lanes) >= 4


def test_only_the_chunks_near_the_runner_are_kept_and_the_rest_come_back_the_same():
    world = c.Course(2)
    first = world.chunk(5)
    for index in range(c.CHUNKS_KEPT + 40):
        world.chunk(index)
    assert world.chunks_held <= c.CHUNKS_KEPT
    assert world.chunk(5) == first


def test_each_kind_of_randomness_has_a_stream_of_its_own():
    draws = {name: c.stream(7, name, 3).random() for name in ('chunk', 'forks', 'wander', 'zones')}
    assert len(set(draws.values())) == 4
    assert c.stream(7, 'chunk', 3).random() == draws['chunk']
    assert c.stream(7, 'chunk', 90210).random() != c.stream(7, 'forks').random()
    assert c.stream(8, 'chunk', 3).random() != draws['chunk']


def test_a_forks_paths_grow_in_from_nothing_and_out_again_by_fork_growth():
    course = c.Course(3)
    fork = course.fork(0)
    d = np.array([fork.start, fork.start + c.FORK_GROW / 2, (fork.start + fork.end) / 2,
                  fork.end - c.FORK_GROW / 2, fork.end])
    growth = course.fork_growth(d)
    assert growth[0] == pytest.approx(0.0) and growth[-1] == pytest.approx(0.0)
    assert 0.0 < growth[1] < 1.0 and 0.0 < growth[3] < 1.0
    assert growth[2] == pytest.approx(1.0)


def _smooth_by_clip(t):
    """What smooth has always answered, written with np.clip."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@pytest.mark.parametrize('values', [
    np.linspace(-2.0, 3.0, 101), np.array([0.0, 1.0, -0.0, 0.5, 1e-12, 1.0 - 1e-12]),
    np.array([[0.2, 1.5], [-1.0, 0.7]]), np.array([np.nan, 0.5, np.inf, -np.inf])])
def test_smooth_is_the_clipped_polynomial_over_arrays(values):
    with np.errstate(invalid='ignore'):
        assert np.array_equal(c.smooth(values), _smooth_by_clip(values), equal_nan=True)


@pytest.mark.parametrize('value', [-3.0, 0.0, 0.25, 1.0, 7.5])
def test_smooth_is_the_clipped_polynomial_for_a_number(value):
    assert c.smooth(value) == _smooth_by_clip(value)
    assert np.ndim(c.smooth(value)) == 0


def _lane_x_by_arrays(world, lane, distance):
    """Where a path lies, worked out from the array methods that answer for a whole run at once."""
    d = np.array([distance], dtype=float)
    state = world.fork_state(d)
    extra = float(state.separation[0]) if state.running[0] and lane in c.Fork.lanes_of(int(state.side[0])) else 0.0
    return float(world.centre(d)[0]) + c.lane_offset(lane) + extra


@pytest.mark.parametrize('seed', [1, 2, 5])
def test_where_a_path_lies_is_what_the_array_methods_say(seed):
    world = c.Course(seed)
    rng = np.random.default_rng(seed)
    places = np.concatenate([rng.uniform(-50.0, 9000.0, 300), np.array([0.0, c.FORK_FIRST, c.FORK_FIRST - 1e-9]),
                             [float(world._fork_start[1]), float(world._fork_end[1]), float(world._fork_start[1]) + 1.0],
                             np.linspace(c.FORK_FIRST, c.FORK_FIRST + 4000.0, 400)])
    for distance in places:
        for lane in c.ALL_LANES:
            assert world.lane_x(lane, float(distance)) == pytest.approx(
                _lane_x_by_arrays(world, lane, float(distance)), abs=1e-9)


def test_where_a_path_lies_is_a_number():
    assert isinstance(c.Course(1).lane_x(1, 100.0), float)
