"""The main, pause and game-over screens."""
import pytest
from OpenGLContext.ui.widgets import Button

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner import menu


def buttons(panel):
    found = {}

    def walk(widget):
        if isinstance(widget, Button):
            found[widget.name] = widget
        for child in getattr(widget, 'children', []) or []:
            walk(child)
    walk(panel)
    return found


@pytest.fixture
def actions():
    calls = []
    return calls, {name: (lambda name=name: calls.append(name))
                   for name in ('resume', 'restart', 'settings', 'credits', 'quit', 'demo')}


def main_screen(calls):
    return menu.main_panel(
        on_play=lambda: calls.append('play'), on_demo=lambda: calls.append('demo'),
        on_settings=lambda: calls.append('settings'), on_credits=lambda: calls.append('credits'),
        on_quit=lambda: calls.append('quit'))


def test_the_main_screen_offers_play_demo_settings_credits_and_quit():
    calls = []
    offered = buttons(main_screen(calls))
    assert list(offered) == ['play', 'demo', 'settings', 'credits', 'quit']
    for button in offered.values():
        button.on_activate(button)
    assert calls == ['play', 'demo', 'settings', 'credits', 'quit']


def test_the_main_screen_is_titled_for_the_game_and_cannot_be_dismissed_without_choosing():
    panel = main_screen([])
    assert panel.title == 'Hill Runner'
    assert panel.modal and not panel.closeOnEscape


def test_settings_and_credits_open_over_the_main_screen_and_the_others_close_it():
    for name, closes in (('settings', False), ('credits', False), ('play', True),
                         ('demo', True), ('quit', True)):
        calls = []
        panel = main_screen(calls)
        button = buttons(panel)[name]
        button.on_activate(button)
        assert (calls, panel.closed) == ([name], closes), name


def test_the_main_screens_demo_button_says_demo_whatever_the_state_of_the_run():
    assert buttons(main_screen([]))['demo'].text == 'Demo'


def test_the_pause_screen_offers_resume_restart_demo_settings_credits_and_quit(actions):
    calls, handlers = actions
    panel = menu.pause_panel(**{'on_' + k: v for k, v in handlers.items()})
    offered = buttons(panel)
    assert set(offered) == {'resume', 'restart', 'demo', 'settings', 'credits', 'quit'}
    for button in offered.values():
        button.on_activate(button)
    assert sorted(calls) == ['credits', 'demo', 'quit', 'restart', 'resume', 'settings']


def test_without_a_demo_to_start_the_pause_screen_has_no_demo_button(actions):
    _, handlers = actions
    del handlers['demo']
    assert 'demo' not in buttons(menu.pause_panel(**{'on_' + k: v for k, v in handlers.items()}))


def test_the_demo_button_says_demo_to_hand_the_run_over_and_play_to_take_it_back(actions):
    _, handlers = actions
    handlers = {'on_' + k: v for k, v in handlers.items()}
    assert buttons(menu.pause_panel(**handlers))['demo'].text == 'Demo'
    assert buttons(menu.pause_panel(demo=True, **handlers))['demo'].text == 'Play'


def test_choosing_the_demo_does_not_also_resume_by_the_panels_own_close(actions):
    calls, handlers = actions
    panel = menu.pause_panel(**{'on_' + k: v for k, v in handlers.items()})
    button = buttons(panel)['demo']
    button.on_activate(button)
    panel.on_close(panel)
    assert calls == ['demo']


def test_escape_on_the_pause_screen_resumes(actions):
    calls, handlers = actions
    panel = menu.pause_panel(**{'on_' + k: v for k, v in handlers.items()})
    panel.on_close(panel)
    assert calls == ['resume']


def test_the_pause_screen_is_modal_over_a_dimmed_game(actions):
    _, handlers = actions
    panel = menu.pause_panel(**{'on_' + k: v for k, v in handlers.items()})
    assert panel.modal and panel.scrim


def test_the_game_over_screen_reports_the_run_and_offers_a_retry_or_quit():
    game = g.Game(c.Course(1))
    game.coins, game.points = 9, 21
    game.runner.distance = 412.0
    calls = []
    panel = menu.game_over_panel(game, on_retry=lambda: calls.append('retry'),
                                 on_quit=lambda: calls.append('quit'))
    texts = []

    def collect(widget):
        if hasattr(widget, 'text') and isinstance(widget.text, str):
            texts.append(widget.text)
        for child in getattr(widget, 'children', []) or []:
            collect(child)
    collect(panel)
    joined = ' '.join(texts)
    assert '412' in joined and '9' in joined and '21' in joined
    offered = buttons(panel)
    assert set(offered) == {'retry', 'quit'}
    offered['retry'].on_activate(offered['retry'])
    assert calls == ['retry']


def test_the_game_over_screen_cannot_be_dismissed_without_choosing():
    panel = menu.game_over_panel(g.Game(c.Course(1)), on_retry=lambda: None, on_quit=lambda: None)
    assert not panel.closeOnEscape
