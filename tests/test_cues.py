"""What the runner looks and sounds like, as a function of what it is doing."""
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import cues
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner.events import Event


@pytest.fixture
def game():
    return g.Game(c.Course(1))


def test_a_runner_just_off_the_line_jogs(game):
    assert cues.pose_for(game).clip == 'Jog_Fwd_Loop'


def test_a_runner_at_speed_sprints(game):
    game.runner.speed = 20.0
    assert cues.pose_for(game).clip == 'Sprint_Loop'


def test_the_stride_quickens_with_speed_through_the_change_of_gait(game):
    cadence = []
    for speed in (10.0, 12.9, 13.0, 18.0, 24.0):
        game.runner.speed = speed
        pose = cues.pose_for(game)
        cadence.append(2 * pose.speed / cues.CYCLE[pose.clip])
    assert cadence == sorted(cadence)


def test_rising_in_a_jump_plays_the_takeoff_and_then_the_fall(game):
    game.runner.jump()
    game.runner.advance(0.05)
    assert cues.pose_for(game).clip == 'Jump_Start'
    game.runner.advance(0.3)
    assert game.runner.height > 0 and not game.runner.rising
    assert cues.pose_for(game).clip == 'Jump_Loop'


def test_the_takeoff_is_played_through_once_at_a_pace_that_fits_the_jump(game):
    game.runner.jump()
    pose = cues.pose_for(game)
    assert not pose.loop
    assert 1.33 / pose.speed <= 0.65


def test_a_slide_plays_the_slide_once(game):
    game.runner.slide()
    pose = cues.pose_for(game)
    assert (pose.clip, pose.loop) == ('Slide', False)


def test_a_crash_plays_the_fall(game):
    game.state = g.State.CRASHED
    pose = cues.pose_for(game)
    assert (pose.clip, pose.loop) == ('Death01', False)


def test_every_pose_names_a_clip_the_figure_carries():
    import importlib.resources as resources

    from OpenGLContext.character import CharacterModel
    path = resources.files('openglcontext_hill_runner') / 'assets' / 'runner.glb'
    clips = set(CharacterModel.load(str(path)).clips)
    assert set(cues.CLIPS) <= clips


@pytest.mark.parametrize('event, name', [
    (Event.JUMP, 'jump_grunt'), (Event.SLIDE, 'slide'), (Event.COIN, 'coin'),
    (Event.JEWEL, 'chime'), (Event.CRASH, 'crash'), (Event.HIT, 'crash'),
    (Event.HEAL, 'chime'),
])
def test_what_happens_is_heard(event, name):
    assert [sound.name for sound in cues.sounds_for([event])] == [name]


def test_a_jewel_has_a_sound_of_its_own():
    assert cues.sounds_for([Event.JEWEL])[0].name != cues.sounds_for([Event.COIN])[0].name


def test_a_hit_is_softer_than_the_crash_that_ends_the_run_and_a_heal_is_softer_than_a_jewel():
    hit, crash = (cues.sounds_for([e])[0] for e in (Event.HIT, Event.CRASH))
    heal, jewel = (cues.sounds_for([e])[0] for e in (Event.HEAL, Event.JEWEL))
    assert hit.gain < crash.gain
    assert heal.gain < jewel.gain and heal.rate < jewel.rate


def test_a_hop_is_a_footfall_and_a_landing_is_a_heavier_one():
    hop, land = (cues.sounds_for([e])[0] for e in (Event.HOP, Event.LAND))
    assert hop.name == land.name == 'footstep'
    assert land.gain > hop.gain


def test_every_sound_named_is_a_file_that_ships():
    import importlib.resources as resources
    audio = resources.files('openglcontext_hill_runner') / 'assets' / 'audio'
    names = {s.name for event in Event for s in cues.sounds_for([event])}
    names |= {cues.MUSIC, cues.STEP}
    for name in names:
        assert (audio / (name + '.ogg')).is_file(), name


def test_a_runner_on_the_ground_takes_steps_at_the_pace_of_its_stride(game):
    steps = cues.Footsteps()
    game.runner.speed = 20.0
    taken = sum(steps.advance(1 / 60, game) for _ in range(60))
    pose = cues.pose_for(game)
    assert taken == pytest.approx(2 * pose.speed / cues.CYCLE[pose.clip], abs=1.5)


def test_no_steps_are_taken_in_the_air_or_sliding_or_after_a_crash(game):
    steps = cues.Footsteps()
    game.runner.jump()
    game.runner.advance(0.1)
    assert not any(steps.advance(1 / 60, game) for _ in range(60))
    game.restart()
    game.runner.slide()
    assert not any(steps.advance(1 / 60, game) for _ in range(60))
    game.restart()
    game.state = g.State.CRASHED
    assert not any(steps.advance(1 / 60, game) for _ in range(60))


def test_a_runner_who_is_down_rolls_to_its_feet_over_as_long_as_it_is_down(game):
    game.state = g.State.DOWN
    pose = cues.pose_for(game)
    assert (pose.clip, pose.loop) == ('Roll', False)
    from OpenGLContext.character import CharacterModel
    import importlib.resources as resources
    path = resources.files('openglcontext_hill_runner') / 'assets' / 'runner.glb'
    length = CharacterModel.load(str(path)).clips['Roll'].duration
    assert length / pose.speed == pytest.approx(g.STUMBLE_TIME, abs=0.05)


def test_no_steps_are_taken_while_down(game):
    game.state = g.State.DOWN
    assert not cues.Footsteps().advance(1.0, game)
