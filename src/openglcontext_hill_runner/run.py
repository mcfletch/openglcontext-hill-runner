"""The ``oglc-hill-runner`` command."""
import argparse
import os
from typing import Any

# The PBR renderer with image-based lighting gives the shadows and the reflective
# coins; set before OpenGLContext is imported so it takes effect.
os.environ.setdefault("OPENGLCONTEXT_RENDERER", "pbr")
os.environ.setdefault("OPENGLCONTEXT_IBL", "full")

from OpenGLContext.contentpacks.application import ask_on_console, console_progress, ensure_needs
from OpenGLContext.contentpacks.store import CONTENT_OVERRIDE

from openglcontext_hill_runner import content
from openglcontext_hill_runner.app import RunnerContext

CONTROLS = """\
Hill runner: left and right hop between the paths, up jumps, down slides.
Escape opens the menu. F2 saves a screenshot. Alt+F shows the developer overlay.
A hit costs 100 coins to continue; with fewer, the run is over.
"""


def build_parser() -> argparse.ArgumentParser:
    """The command line's options."""
    parser = argparse.ArgumentParser(description="An endless downhill runner for OpenGLContext")
    parser.add_argument("--seed", type=int, default=1, help="which hill to run")
    parser.add_argument("--demo", action="store_true", help="let the autopilot run")
    parser.add_argument("--yes", action="store_true",
                        help="fetch the game's art, the forest demo's art and the texture pack without "
                             "asking, where they are not here")
    parser.add_argument("--no-sound", action="store_true", help="play silently")
    parser.add_argument("--no-forest", action="store_true",
                        help="plant simple trees and no ground cover, with no use of (or download "
                             "of) the forest demo's art")
    parser.add_argument("--no-photos", action="store_true",
                        help="draw the ground with procedural textures, with no download")
    parser.add_argument("--size", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"),
                        help="the window's size in pixels")
    return parser


def fetch_assets(assume_yes: bool = False, forest: bool = True, photos: bool = True) -> None:
    """Make sure the packs the game runs on are here before anything opens them.

    The game's own art, the forest demo's trees and ground cover (``forest``) and the engine's
    photographic textures (``photos``). A checkout has the art and an installed pack is found, and
    either way nothing is asked. Otherwise this lists the packs' titles, sizes and terms on the
    console once and fetches them in the foreground on a yes (``assume_yes`` goes ahead without
    asking, for a script). Declining, or a download that fails, stops the program with a message
    saying what happened and how to run from a copy of the art.
    """
    needs = content.required(forest=forest, photos=photos)
    if not needs:
        return
    offline = ("To run from a copy of the art you already have, point %s at a directory holding "
               "%s/assets." % (CONTENT_OVERRIDE, content.NAMESPACE))

    def announce(pack: Any) -> None:
        print("\n  %s" % pack.title)

    try:
        here = ensure_needs(needs, consent=ask_on_console(assume_yes=assume_yes),
                            progress=console_progress(), started=announce)
    except OSError as error:
        raise SystemExit("\nThe game's art did not download: %s\n%s" % (error, offline)) from error
    if not here:
        raise SystemExit("Not downloading; the game has nothing to show without its art. " + offline)
    print("\r  done")


def main(argv: Any = None) -> None:
    """Play the game, with the options of ``argv`` (the command line by default)."""
    args = build_parser().parse_args(argv)
    fetch_assets(assume_yes=args.yes, forest=not args.no_forest, photos=not args.no_photos)
    RunnerContext.seed = args.seed
    RunnerContext.demo = args.demo
    RunnerContext.main_menu = not args.demo
    RunnerContext.sound = not args.no_sound
    RunnerContext.photos = not args.no_photos
    RunnerContext.forest = not args.no_forest
    print(CONTROLS)
    if args.size:
        RunnerContext.ContextMainLoop(size=tuple(args.size))
    else:
        RunnerContext.ContextMainLoop(size=(1280, 720))


if __name__ == "__main__":
    main()
