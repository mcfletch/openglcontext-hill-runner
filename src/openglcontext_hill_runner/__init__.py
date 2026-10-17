"""An endless downhill runner for OpenGLContext.

The player runs a hill that never ends, on parallel paths, hopping between them
unless something blocks the way, jumping low obstacles and sliding under high
ones, and collecting treasure. The hill is generated tile by tile as the runner
nears it and streamed through the engine's 3D Tiles runtime.

The rules (the course, the runner's body, collisions, scoring) are plain objects
with no GL, tested without a window; ``app`` is the window that draws them.

Run it with ``oglc-hill-runner``, or ``python -m openglcontext_hill_runner``.
"""

__version__ = "0.1.0a1"
