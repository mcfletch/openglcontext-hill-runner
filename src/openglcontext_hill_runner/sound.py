"""The game's sounds, played through the window's audio engine.

The engine's mixer does the mixing: it holds the voices, drops the least
important when they run out, and muffles its whole output on request.
:class:`Sounds` says which files play, how loud, and which matter most.

``engine`` is what :func:`OpenGLContext.audio.scene.engine_for` answers for a
context: an object whose ``play`` returns a handle, or ``None`` where sound is
switched off or the machine has no device, in which case the game is silent.
"""
import importlib.resources as resources
from collections.abc import Iterable
from typing import Any

from openglcontext_hill_runner.cues import MUSIC, Sound

MUSIC_GAIN = 0.35
#: The music is never the voice the mixer gives up when it runs out.
MUSIC_PRIORITY = 1.0
#: How muffled the engine's output is while the game is paused: 0 is clear, 1 is underwater.
PAUSE_MUFFLE = 0.8


class Sounds:
    """Plays one-shot sounds and the looping soundtrack."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._music: Any = None

    @staticmethod
    def path(name: str) -> str:
        """The shipped file for a sound's name."""
        return str(resources.files('openglcontext_hill_runner') / 'assets' / 'audio'
                   / (name + '.ogg'))

    def play(self, sounds: Iterable[Sound]) -> None:
        """Play each of ``sounds`` once."""
        if self.engine is None:
            return
        for sound in sounds:
            self.engine.play(self.path(sound.name), gain=sound.gain, rate=sound.rate)

    def start_music(self) -> None:
        """Start the soundtrack, once; it loops until :meth:`stop`."""
        if self.engine is not None and self._music is None:
            self._music = self.engine.play(self.path(MUSIC), gain=MUSIC_GAIN, loop=True,
                                          priority=MUSIC_PRIORITY)

    def pause(self, paused: bool) -> None:
        """Muffle everything for a pause screen, or clear it."""
        if self.engine is not None:
            self.engine.muffle = PAUSE_MUFFLE if paused else 0.0

    def stop(self) -> None:
        """Stop the music."""
        if self._music is not None:
            self._music.stop()
            self._music = None
