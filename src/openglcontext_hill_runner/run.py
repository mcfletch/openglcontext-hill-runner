"""The ``oglc-hill-runner`` command."""
import argparse
import os
from typing import Any

# The PBR renderer with image-based lighting gives the shadows and the reflective
# coins; set before OpenGLContext is imported so it takes effect.
os.environ.setdefault("OPENGLCONTEXT_RENDERER", "pbr")
os.environ.setdefault("OPENGLCONTEXT_IBL", "full")

from openglcontext_hill_runner.app import RunnerContext

CONTROLS = """\
Hill runner: left and right hop between the paths, up jumps, down slides.
Escape opens the menu.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="An endless downhill runner for OpenGLContext")
    parser.add_argument("--seed", type=int, default=1, help="which hill to run")
    parser.add_argument("--demo", action="store_true", help="let the autopilot run")
    parser.add_argument("--no-sound", action="store_true", help="play silently")
    parser.add_argument("--no-forest", action="store_true",
                        help="plant simple trees and no ground cover, with no use of the forest "
                             "demo's art")
    parser.add_argument("--no-photos", action="store_true",
                        help="draw the ground with procedural textures, with no download")
    parser.add_argument("--size", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"),
                        help="the window's size in pixels")
    return parser


def main(argv: Any = None) -> None:
    args = build_parser().parse_args(argv)
    RunnerContext.seed = args.seed
    RunnerContext.demo = args.demo
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
