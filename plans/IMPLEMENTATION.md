# Hill runner

## Done

- Engine: `GeneratedTiles` (see openglcontext `plans/GENERATED-TILES.md`).
- Rules without GL: course, runner, game, pilot, cues, chase, session. 159 tests, including a pilot that runs 24
  generated courses for 3500 m each.
- Slide clip: `grass-clumps/slide.py` adds `Slide` to the UAL figure; `assets/runner.glb` is its output.
- Scale: trails 0.5 m wide, 0.2 m apart; the camera sits above head height with the feet at the bottom edge of the frame.
- Ground: metre and hand-sized relief in the land, patches of lush and dry ground by vertex colour, rocks, CC0 photographs for the ground and trails.
- Forest: the forest demo's trees through `VegetationField` and its grass, ferns and flowers through `GroundCover`, kept clear of the trails by each tree's crown radius.
- Window: hill tiles, runner, coins, HUD, pause and end-of-run screens, music and sound effects, shadows.

- Barriers a metre high across one or two paths, as logs, boulders, gates and walls, and coins kept out of their way; coins textured, jewels cut in facets that hang where a jump reaches.
- Credits at start and in the menu.
- Countries (forest, rocks, sand, meadow, dense forest) along the hill, and a wandering trail; the land is carved around every trail.
- Forks: two extra paths on a side that draw away through another country and rejoin; the pilot takes or declines them and survives every seed it is tried on.

## Open

- Branching paths and rivers to swim: a course chunk needs a way to say a path forks, and the runner needs a
  swim state with `Swim_Fwd_Loop`.
- The engine's splat ground (`SplatTerrain`, `GroundShading`) reads one control map over a bounded world; an endless hill needs a control map that repeats.
- The slide reads small from behind the runner; a lower camera during a slide would show the pose.
