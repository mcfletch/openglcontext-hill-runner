# openglcontext-hill-runner

An endless runner down a hill that never ends, built on OpenGLContext.

```bash
oglc-hill-runner                # play
oglc-hill-runner --demo         # the autopilot plays
oglc-hill-runner --seed 7       # a different hill
oglc-hill-runner --no-sound --size 1920 1080
oglc-hill-runner --no-forest --no-photos   # no use of the forest demo's art, no download
```

## Playing

| Key | Does |
| --- | --- |
| Left, Right | Hop to the next path, unless a hedge stands between the two or the paths have drawn apart |
| Up | Jump over a log or a boulder, or up to a jewel |
| Down | Slide under a gate (in the air: drop, then slide) |
| Escape | Menu: resume, restart, demo, settings, credits, quit. Demo hands the run to the computer, which runs without a crash for as long as it is left; the menu's Play takes it back |
| Alt+F | Show or hide the developer overlay: frame rate, timings and draw counts |
| F2 | Save a screenshot to the pictures folder, as `Hill-Runner-0001.png` and so on |

The three trails are 0.5 m wide with 0.2 m between them, so a hop is one big step. The runner speeds
up from 10 to 24 m/s and cannot slow down. A log, a boulder, a gate board or a wall ends the run; they
stand across one or two trails and are a metre high or more. Coins hang at knee height and are worth 1.
Jewels hang above a runner's reach and are worth 5: they need a jump. The HUD shows coins, score and
metres run.

### The country

The hill runs through forest, rocks, sand, meadow and dense forest, a few hundred metres of each, in an
order the seed picks. The land, the ground, the trees, the ground cover and the rocks change with it.
The trail wanders from side to side.

### Forks

Every kilometre or so two more paths appear beside the trail, on the left or the right. They draw away
and run through another country, then come back and join. Hop onto them in the first few seconds, while
they are within a hop of the trail; after that they are on a trail of their own. A path that joins the
trail carries its runner onto the nearest path of it.

## What it uses from the engine

| Feature | From |
| --- | --- |
| The hill, one tile for each 72 m, made as the runner nears it | `GeneratedTiles`, `OpenGLContext.loaders.tiles3d.generated` |
| The runner, with `Sprint_Loop`, `Jog_Fwd_Loop`, `Jump_Start`, `Jump_Loop`, `Slide` and `Death01` | `OpenGLContext.character.CharacterModel` |
| Sun with cascaded shadow maps, PBR materials, fog | `DirectionalLight`, the PBR pass, `Fog` |
| Ground, trail, rock and bark photographs (CC0) | `OpenGLContext.loaders.cc0.physical_material`, with procedural `OpenGLContext.scenegraph.surfaces` as the fallback |
| Rocks, logs, fence rails and dry-stone walls | `OpenGLContext.scenegraph.natural` (`stone`, `timber`, `rubble`) |
| Rock country's ground: flat faces scraped into the land | `OpenGLContext.scenegraph.terrain.RockRelief` |
| Procedural textures for bark, sawn ends and the ground | `OpenGLContext.scenegraph.surfaces` |
| Trees: the forest demo's models, real branches near and cards far, swapped a stretch at a time | `VegetationField.replant`, from the forest demo's art |
| Grass, ferns, nettles, flowers on the forest floor | `GroundCover`, from the forest demo's art: the grass and the other plants as two nodes (`groundcover`) |
| Chase camera | `OpenGLContext.move.followcam.FollowCamera` |
| HUD, pause and end-of-run screens | `OpenGLContext.ui.hudwidgets`, `OpenGLContext.ui` panels |
| Music, footsteps, grunt, slide, coins | the engine's audio mixer, through `audioscene.engine_for` |

## How it is put together

The rules have no GL and are tested without a window.

| Module | Holds |
| --- | --- |
| `course` | Paths, the slope, the wandering trail, forks, obstacles, coins, jewels and hedges; a chunk is a function of the seed and its index |
| `landscape` | The countries, and the land, ground layers, tree, cover and rock densities of each |
| `runner` | The body: speed, hopping, jumping, sliding |
| `game` | Collisions, treasure, the end of a run |
| `pilot` | The autopilot; the test that every generated course can be run |
| `cues` | Which clip plays, which sounds fire, when a foot lands |
| `chase` | The point the camera follows |
| `session` | One run advanced a frame at a time: the above, together |
| `scenery`, `look`, `tiles` | The hill as meshes and materials, and the tile generator; where one country gives way to another the second ground is blended over the first |
| `groundcover` | The grass, grown tall, and the other plants of the forest demo's art as two nodes, with where each grows |
| `woods` | The forest demo's trees, planted a stretch of hill at a time, clear of the trails |
| `treasure`, `hud`, `menu`, `credits`, `sound` | Coins and jewels in view, the HUD, the screens, the credits shown at start and in the menu, the engine's audio mixer |
| `app`, `run` | The window and the command |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The render smoke test skips where there is no display.

## Assets

Audio and the character are CC0. The forest demo's tree models are CC-BY 4.0 and credited on launch. `ASSET-LICENSES.md` lists each file. The character's `Slide` clip is made by
`grass-clumps/slide.py` in the OpenGL-dev workspace.
