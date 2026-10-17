"""Sounds: the game's sounds, through whatever audio engine the window has."""
import pytest

from openglcontext_hill_runner import cues
from openglcontext_hill_runner.sound import MUSIC_PRIORITY, PAUSE_MUFFLE, Sounds


class Handle:
    def __init__(self, name, gain, loop, priority=0.0):
        self.name, self.gain, self.loop, self.stopped = name, gain, loop, False
        self.priority = priority

    def set_gain(self, left, right):
        self.gain = (left, right)

    def stop(self):
        self.stopped = True


class Engine:
    """An audio engine that keeps what it was asked to play."""

    def __init__(self):
        self.played = []
        self.muffle = 0.0

    def play(self, source, emitter=None, position=None, gain=1.0, priority=0.0,
             loop=False, rate=1.0):
        handle = Handle(source, gain, loop, priority)
        handle.rate = rate
        self.played.append(handle)
        return handle


@pytest.fixture
def engine():
    return Engine()


@pytest.fixture
def mixer(engine):
    return Sounds(engine)


def test_a_sound_plays_from_its_file_at_its_gain_and_rate(mixer, engine):
    mixer.play([cues.Sound('coin', 0.7, 1.5)])
    (heard,) = engine.played
    assert heard.name.endswith('coin.ogg')
    assert (heard.gain, heard.rate, heard.loop) == (pytest.approx(0.7), 1.5, False)


def test_the_files_the_mixer_names_ship_with_the_game(mixer):
    import os
    for name in ('coin', 'crash', 'footstep', 'jump_grunt', 'slide', 'music'):
        assert os.path.isfile(mixer.path(name))


def test_the_music_loops_and_plays_once(mixer, engine):
    mixer.start_music()
    mixer.start_music()
    (music,) = engine.played
    assert music.loop and music.name.endswith('music.ogg')


def test_the_music_outranks_every_other_voice(mixer, engine):
    mixer.start_music()
    mixer.play([cues.Sound('coin')])
    music, coin = engine.played
    assert music.priority == MUSIC_PRIORITY > coin.priority


def test_the_engine_is_muffled_while_the_game_is_paused_and_clear_after(mixer, engine):
    mixer.pause(True)
    assert engine.muffle == PAUSE_MUFFLE
    mixer.pause(False)
    assert engine.muffle == 0.0


def test_stopping_stops_the_music(mixer, engine):
    mixer.start_music()
    mixer.stop()
    assert engine.played[0].stopped


def test_without_an_engine_everything_is_silent_and_nothing_fails():
    quiet = Sounds(None)
    results = [quiet.start_music(), quiet.play([cues.Sound('coin')]), quiet.pause(True),
               quiet.stop()]
    assert results == [None] * 4


def test_every_shipped_sound_plays_on_the_real_engine():
    from omi_audio.device import NullDevice
    from omi_audio.engine import AudioEngine
    engine = AudioEngine(device=NullDevice(), voices=16)
    sounds = Sounds(engine)
    sounds.start_music()
    sounds.play([cues.Sound(name) for name in ('coin', 'crash', 'footstep', 'jump_grunt', 'slide')])
    assert engine.mixer.active_voices == 6
    sounds.pause(True)
    assert engine.muffle == PAUSE_MUFFLE
