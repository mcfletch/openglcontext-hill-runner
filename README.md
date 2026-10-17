# openglcontext-hill-runner

An endless runner down a hill that never ends, built on OpenGLContext.

```bash
oglc-hill-runner                # the main screen: play, demo, settings, credits, quit
oglc-hill-runner --demo         # the autopilot plays, with no main screen
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
| Escape | Menu: resume, restart, demo, settings, credits, quit. Demo hands the run to the computer, which runs without a hit for as long as it is left; the menu's Play takes it back. The main screen the game opens on offers Play and Demo too |
| Alt+F | Show or hide the developer overlay: frame rate, timings and draw counts |
| F2 | Save a screenshot to the pictures folder, as `Hill-Runner-0001.png` and so on |

The three trails are 0.5 m wide with 0.2 m between them, so a hop is one big step. The runner speeds
up from 10 to 24 m/s and cannot slow down on its own. A log, a boulder, a gate board or a wall is a
hit; they stand across one or two trails and are a metre high or more. Coins hang at knee height and are
worth 1. Jewels hang above a runner's reach and are worth 5: they need a jump. The HUD shows coins,
score, hits and metres run.

### Damage

A hit knocks the runner down just short of the obstacle. It stays there for 1.3 s, rolling to its feet, with the keys doing nothing, and then runs on from 10 m/s: the speed gained is lost. Three hits end the run. The edges of the
screen are shaded red in proportion to the hits taken (a third, two thirds, all of the shading) and
flash at each new hit. Nothing hits the runner again for the next 15 m, so a row of obstacles is one hit.
500 m run without a hit takes one hit away, and another 500 m after that takes the next.

### The country

The hill runs through forest, rocks, sand, meadow and dense forest, a few hundred metres of each, in an
order the seed picks. The land, the ground, the trees, the ground cover and the rocks change with it.
The trail wanders from side to side.

### Caves

Every two or three kilometres, after a stretch of rock country, the trail runs into a cave: the
hill closes in over the trail, a face of rock stands over the mouth, and the way on is a tunnel
250 to 320 m long. The tunnel leads out into more rock country or into forest, the seed's choice.

* The ceiling is high in places (4.4 to 7.5 m over the floor) and low in others (2.4 to 3.0 m),
  changing over 16 m. Running or sliding under a low ceiling is safe; **a jump under one is a hit**,
  and so is a jump at the edge of one, since the head arrives at the top of the jump. The camera
  ducks under a low ceiling and rises again once it is clear.
* A crevice runs right across the floor, 1.8 m wide and 2.6 m deep. It has to be jumped, and a runner
  who is on the ground over it is a hit. Crevices stand only where no low ceiling is within 14 m, so
  the jump a crevice asks for is always safe; so is every jump the course asks for in a cave, and the
  jewels that hang out of reach of a standing runner.
* The logs, boughs and walls of the open hill are in caves too. A fallen log under a low ceiling is a
  bough to slide under instead.
* A torch on the wall every 18 m, on alternate sides, is all the light there is. The sun, the sky's light
  and the haze go out over the first 28 m inside the mouth. The torches' lights are the nearest of the
  scene's lights, which is what the engine binds when there are more than it has room for.
* No fork runs near a cave, and no trees or ground cover grow over one.

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
| HUD, with the edge shading of damage; main, pause and end-of-run screens | `OpenGLContext.ui.hudwidgets` (`DamageIndicator`: `level`, `hurt_all`), `OpenGLContext.ui` panels |
| Music, footsteps, grunt, slide, coins | the engine's audio mixer, through `audioscene.engine_for` |

## How it is put together

The rules have no GL and are tested without a window.

| Module | Holds |
| --- | --- |
| `course` | Paths, the slope, the wandering trail, forks, caves and their ceilings, obstacles (a crevice among them), coins, jewels and hedges; a chunk is a function of the seed and its index |
| `landscape` | The countries, and the land, ground layers, tree, cover and rock densities of each |
| `runner` | The body: speed, hopping, jumping, sliding |
| `game` | Collisions, damage and healing, the fall and getting up, head against a ceiling, treasure, the end of a run |
| `atmosphere` | How dark and how hazy the air is at a distance: bright in the open, dark in a cave |
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

Audio and the character are CC0. The music is Medieval: Market Day by RandomMind. The forest demo's tree models are CC-BY 4.0 and credited on launch. `ASSET-LICENSES.md` lists each file. The character's `Slide` clip is made by
`grass-clumps/slide.py` in the OpenGL-dev workspace.
