"""The screens over the game: the main screen, pause, and the end of a run, from the engine's
overlay UI."""
from collections.abc import Callable
from gettext import gettext as _

from OpenGLContext.ui.layout import Column
from OpenGLContext.ui.panel import Panel
from OpenGLContext.ui.widgets import PRIMARY, Button, Label

from openglcontext_hill_runner.game import Game

Action = Callable[[], None]


def _button(text: str, name: str, action: Action, panel: Panel, chosen: list[str],
            role: str = '', closes: bool = True) -> Button:
    """A button that closes the panel and then does ``action``.

    ``chosen`` records that a button did it, so the panel's own close handler,
    which stands for dismissing it some other way, does not also run. With ``closes`` false the
    panel stays up and ``action`` opens something over it.
    """
    button = Button(text=text, name=name)
    if role:
        button.role = role

    def activate(widget: object) -> None:  # noqa: ARG001 the engine calls a button with itself
        if closes:
            chosen.append(name)
            panel.close()
        action()
    button.on_activate = activate
    return button


def main_panel(on_play: Action, on_demo: Action, on_settings: Action, on_credits: Action,
               on_quit: Action) -> Panel:
    """The screen the game opens on. It stays until a button is chosen.

    ``on_demo`` hands the run to the computer, which plays on without a crash for as long as it
    is left. Settings and credits open over the screen, which is there again when they close.
    """
    panel = Panel(title=_('Hill Runner'), modal=True, scrim=True, closeOnEscape=False,
                  preferredColumns=28)
    chosen: list[str] = []
    panel.children = [Column(spacing=6, children=[
        _button(_('Play'), 'play', on_play, panel, chosen, PRIMARY),
        _button(_('Demo'), 'demo', on_demo, panel, chosen),
        _button(_('Settings'), 'settings', on_settings, panel, chosen, closes=False),
        _button(_('Credits'), 'credits', on_credits, panel, chosen, closes=False),
        _button(_('Quit'), 'quit', on_quit, panel, chosen),
    ])]
    return panel


def pause_panel(on_resume: Action, on_restart: Action, on_settings: Action,
                on_credits: Action, on_quit: Action, on_demo: Action | None = None,
                demo: bool = False) -> Panel:
    """Escape's screen. Closing it any other way than a button resumes.

    ``on_demo`` adds a button that hands the run to the computer, or back to the player where
    ``demo`` says the computer is playing.
    """
    panel = Panel(title=_('Paused'), modal=True, scrim=True, preferredColumns=28)
    chosen: list[str] = []
    panel.children = [Column(spacing=6, children=[
        _button(_('Resume'), 'resume', on_resume, panel, chosen, PRIMARY),
        _button(_('Restart'), 'restart', on_restart, panel, chosen),
        *([_button(_('Play') if demo else _('Demo'), 'demo', on_demo, panel, chosen)]
          if on_demo is not None else []),
        _button(_('Settings'), 'settings', on_settings, panel, chosen),
        _button(_('Credits'), 'credits', on_credits, panel, chosen),
        _button(_('Quit'), 'quit', on_quit, panel, chosen),
    ])]
    panel.on_close = lambda _closing: None if chosen else on_resume()
    return panel


def game_over_panel(game: Game, on_retry: Action, on_quit: Action) -> Panel:
    """The end of a run, which only a hit that the coins could not pay for ends: how far, and
    what was collected."""
    panel = Panel(title=_('No coins for continue, game over'), modal=True, scrim=True, closeOnEscape=False,
                  preferredColumns=30)
    chosen: list[str] = []
    panel.children = [Column(spacing=6, children=[
        Label(text=_('You ran %d m') % game.meters, name='distance'),
        Label(text=_('%d coins, %d points') % (game.coins, game.points), name='treasure'),
        _button(_('Run again'), 'retry', on_retry, panel, chosen, PRIMARY),
        _button(_('Quit'), 'quit', on_quit, panel, chosen),
    ])]
    return panel
