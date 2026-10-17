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
    import inspect
    source = inspect.getsource(RunnerContext.OnInit)
    for key in ('<left>', '<right>', '<up>', '<down>', '<escape>'):
        assert key in source
    assert source.count("'<") == 5


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


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_the_window_draws_without_the_forest_demos_art_or_any_download(tmp_path):
    pixels = capture(tmp_path, 120, "--demo", "--no-forest", "--no-photos")
    assert pixels.std() > 25.0
    assert (pixels[..., 2] > pixels[..., 0] + 30).mean() > 0.02
