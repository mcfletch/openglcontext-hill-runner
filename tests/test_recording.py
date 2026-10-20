"""Recording the autopilot's run to a video file, in each codec the GPU encodes."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("glfw")
pytest.importorskip("pyopengl_video")


def _has_gl():
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))


pytestmark = [pytest.mark.needs_gl,
              pytest.mark.skipif(not _has_gl(), reason="no GL display available")]

#: Asks the hardware which codecs it encodes, from a process with a context current.
ENCODED = '''
import glfw, pyopengl_video
from OpenGLContext.testing.glcontext import hidden_window
with hidden_window("probe", size=(64, 64)):
    found = sorted({c for b in pyopengl_video.encoders() for c in b.codecs})
print("ENCODES", *found, flush=True)
'''


def run_game(*arguments, timeout=420):
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_HIDDEN="1")
    return subprocess.run(
        [sys.executable, "-m", "openglcontext_hill_runner", "--no-forest", "--no-photos",
         *arguments], env=env, capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def encoded():
    result = subprocess.run([sys.executable, "-c", ENCODED], capture_output=True, text=True,
                            timeout=120)
    for line in result.stdout.splitlines():
        if line.startswith("ENCODES"):
            return set(line.split()[1:])
    pytest.skip("no hardware video encoder here: " + result.stderr[-500:])


@pytest.mark.parametrize("codec", ["h264", "hevc", "vp9", "av1"])
def test_a_run_is_recorded_in_the_codec_asked_for(tmp_path, codec, encoded):
    if codec not in encoded:
        pytest.skip("this GPU does not encode %s" % codec)
    from pyopengl_video import codecs
    from pyopengl_video.mp4 import MP4Reader

    path = tmp_path / ("%s.mp4" % codec)
    result = run_game("--record", str(path), "--record-codec", codec, "--record-seconds", "1",
                      "--record-fps", "30", "--record-delay", "0.5", "--size", "320", "180")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "recorded 30 frames" in result.stdout, result.stdout[-1000:]
    with MP4Reader(path) as movie:
        assert codecs.canonical(movie.video.codec) == codec
        # AV1 is coded in blocks of eight, so its picture is trimmed to a multiple of 8
        assert movie.video.size == ((320, 176) if codec == "av1" else (320, 180))
        assert movie.duration == pytest.approx(1.0, abs=0.1)


def test_a_codec_the_gpu_lacks_ends_the_program_saying_so(tmp_path, encoded):
    missing = next((c for c in ("vp9", "av1", "hevc") if c not in encoded), None)
    if missing is None:
        pytest.skip("this GPU encodes every codec there is")
    path = tmp_path / "none.mp4"
    result = run_game("--record", str(path), "--record-codec", missing, "--record-seconds", "1",
                      "--size", "320", "180")
    assert result.returncode != 0
    assert "Traceback" not in result.stderr, result.stderr[-2000:]
    assert missing in result.stderr.lower() or missing.upper() in result.stderr


HELD = '''
import sys
from openglcontext_hill_runner import run
from openglcontext_hill_runner.app import RunnerContext

original = RunnerContext.presentFrame


def present(self):
    if self.recorder is not None:
        print("SEEN", int(self.recorder.started), self.session.game.runner.distance, flush=True)
    return original(self)


RunnerContext.presentFrame = present
run.main(sys.argv[1:])
'''


def test_the_run_waits_for_the_recording_and_so_opens_the_video_at_its_start(tmp_path, encoded):
    if "h264" not in encoded:
        pytest.skip("this GPU does not encode h264")
    script = tmp_path / "held.py"
    script.write_text(HELD)
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_HIDDEN="1")
    result = subprocess.run(
        [sys.executable, str(script), "--no-forest", "--no-photos", "--record",
         str(tmp_path / "held.mp4"), "--record-seconds", "0.5", "--record-fps", "30",
         "--record-delay", "1.5", "--size", "320", "180"],
        env=env, capture_output=True, text=True, timeout=420)
    assert result.returncode == 0, result.stderr[-2000:]
    seen = [line.split()[1:] for line in result.stdout.splitlines() if line.startswith("SEEN")]
    waiting = [float(distance) for started, distance in seen if started == "0"]
    kept = [float(distance) for started, distance in seen if started == "1"]
    assert len(waiting) > 5                     # frames were drawn while it waited
    assert max(waiting) < 1.0                   # and the runner stood still for them
    assert kept and kept[0] < 1.0               # so the video opens on the start of the run
