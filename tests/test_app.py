"""The window: what it is declared to be, and that it draws a frame of the game."""
import os
import subprocess
import sys

import numpy as np
import pytest
from OpenGLContext.context import Context

from openglcontext_hill_runner.app import RunnerContext

pytest.importorskip("glfw")
Image = pytest.importorskip("PIL.Image")


def test_it_is_a_glfw_window_with_no_free_camera():
    assert issubclass(RunnerContext, Context)
    assert RunnerContext.windowSystemName == 'glfw'
    assert not RunnerContext.resolveDefinition().navigation


def test_the_arrow_keys_and_escape_are_the_whole_control_scheme():
    from openglcontext_hill_runner.app import CONTROLS
    assert set(CONTROLS) == {'<left>', '<right>', '<up>', '<down>', '<escape>'}
    for method in CONTROLS.values():
        assert callable(getattr(RunnerContext, method))


def _has_gl():
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))


def capture(tmp_path, frames, *arguments):
    env = dict(os.environ)
    env.update(
        OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_DISABLE_FPS_DISPLAY="1",
        OPENGLCONTEXT_HIDDEN="1", OPENGLCONTEXT_AUTO_EXIT_FRAMES=str(frames),
        OPENGLCONTEXT_AUTO_EXIT_CAPTURE_DIR=str(tmp_path),
        OPENGLCONTEXT_AUTO_EXIT_CAPTURE_NAME="shot",
    )
    result = subprocess.run([sys.executable, "-m", "openglcontext_hill_runner",
                             "--no-sound", *arguments],
                            env=env, capture_output=True, text=True, timeout=240)
    shot = tmp_path / "shot.png"
    assert shot.exists(), f"no frame captured; stderr:\n{result.stderr[-2000:]}"
    assert 'Traceback' not in result.stderr, result.stderr[-2000:]
    return np.asarray(Image.open(shot).convert("RGB"), dtype=float)


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_the_window_draws_the_hill_with_sky_ground_and_a_runner(tmp_path):
    pixels = capture(tmp_path, 120, "--demo")
    height = pixels.shape[0]
    ground = pixels[height * 3 // 4:]
    assert pixels.std() > 25.0
    assert (pixels[..., 2] > pixels[..., 0] + 30).mean() > 0.02    # some sky
    assert ground[..., 1].mean() > 40                              # lit ground under the runner
    middle = pixels[height // 3:, pixels.shape[1] // 3: 2 * pixels.shape[1] // 3]
    red = (middle[..., 0] > 140) & (middle[..., 1] < 70) & (middle[..., 2] < 70)
    assert red.sum() > 20                                          # the red bands of her suit


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_the_window_draws_without_the_forest_demos_art_or_any_download(tmp_path):
    pixels = capture(tmp_path, 120, "--demo", "--no-forest", "--no-photos")
    assert pixels.std() > 25.0
    assert (pixels[..., 2] > pixels[..., 0] + 30).mean() > 0.02


RESTARTING = '''
import sys
from openglcontext_hill_runner.app import RunnerContext


class Restarting(RunnerContext):
    frames = 0
    seed = 1
    demo = True
    sound = False
    photos = False
    forest = True

    def OnIdle(self, *arguments):
        type(self).frames += 1
        if self.frames == 20:
            self.first = (self.cover, self.woods, self.terrain)
            self._restart()
        if self.frames == 60:
            now = (self.cover, self.woods, self.terrain)
            kept = [a is b for a, b in zip(self.first, now)]
            print('KEPT', kept, 'SEED', self.seed, flush=True)
        return super().OnIdle(*arguments)


Restarting.ContextMainLoop(size=(640, 360))
'''


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_a_restart_grows_the_same_cover_and_woods_on_the_new_hill_and_draws(tmp_path):
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_DISABLE_FPS_DISPLAY="1",
               OPENGLCONTEXT_HIDDEN="1", OPENGLCONTEXT_AUTO_EXIT_FRAMES="90",
               OPENGLCONTEXT_AUTO_EXIT_CAPTURE_DIR=str(tmp_path),
               OPENGLCONTEXT_AUTO_EXIT_CAPTURE_NAME="shot")
    script = tmp_path / "restart.py"
    script.write_text(RESTARTING)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert 'Traceback' not in result.stderr, result.stderr[-2000:]
    assert 'KEPT [True, True, False] SEED 2' in result.stdout, result.stdout[-1000:]


TREELESS = '''
import numpy as np
from openglcontext_hill_runner import woods
from openglcontext_hill_runner.app import RunnerContext

woods.chunk_trees = lambda land, index: woods.Trees(
    np.zeros((0, 3), 'f'), np.zeros(0, 'f'), np.zeros(0, 'f'), np.zeros(0, int))


class Treeless(RunnerContext):
    seed = 1
    demo = True
    sound = False
    photos = False
    forest = True


Treeless.ContextMainLoop(size=(640, 360))
'''


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_woods_with_no_trees_in_them_draw_without_error(tmp_path):
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_DISABLE_FPS_DISPLAY="1",
               OPENGLCONTEXT_HIDDEN="1", OPENGLCONTEXT_AUTO_EXIT_FRAMES="40",
               OPENGLCONTEXT_AUTO_EXIT_CAPTURE_DIR=str(tmp_path),
               OPENGLCONTEXT_AUTO_EXIT_CAPTURE_NAME="shot")
    script = tmp_path / "treeless.py"
    script.write_text(TREELESS)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert (tmp_path / "shot.png").exists(), result.stderr[-2000:]
    assert 'Traceback' not in result.stderr and 'Error' not in result.stderr, result.stderr[-2000:]
