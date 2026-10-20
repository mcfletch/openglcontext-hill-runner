"""The command line: which options there are, and what recording implies."""
import pytest

from openglcontext_hill_runner import run


def parsed(*arguments):
    return run.build_parser().parse_args(list(arguments))


def test_no_recording_is_asked_for_by_default():
    assert run.recording_request(parsed()) is None


def test_record_asks_for_thirty_seconds_at_sixty_frames_a_second_in_h264():
    request = run.recording_request(parsed('--record', 'run.mp4'))
    assert (request.path, request.codec, request.seconds, request.fps) == (
        'run.mp4', 'h264', 30.0, 60)


@pytest.mark.parametrize('codec', ['h264', 'hevc', 'vp9', 'av1'])
def test_each_codec_can_be_recorded_in(codec):
    request = run.recording_request(parsed('--record', 'run.mp4', '--record-codec', codec))
    assert request.codec == codec


def test_the_options_say_how_long_how_fast_and_how_much_to_let_the_world_arrive():
    request = run.recording_request(parsed(
        '--record', 'run.mp4', '--record-seconds', '8', '--record-fps', '30',
        '--record-delay', '2.5', '--record-bitrate', '4000000'))
    assert request.options() == dict(codec='h264', fps=30, seconds=8.0, start_after=2.5,
                                     bitrate=4_000_000)


def test_a_recording_is_of_the_autopilot_with_no_main_screen_and_no_sound():
    arguments = parsed('--record', 'run.mp4')
    assert run.plays_itself(arguments) is True
    assert run.plays_silently(arguments) is True


def test_playing_is_unchanged_without_a_recording():
    arguments = parsed()
    assert run.plays_itself(arguments) is False
    assert run.plays_silently(arguments) is False
    assert run.plays_itself(parsed('--demo')) is True
    assert run.plays_silently(parsed('--no-sound')) is True
