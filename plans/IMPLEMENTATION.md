# Hill runner

## Done

- Engine: `GeneratedTiles` with a per-tile `origin`, `VegetationField.replant` and empty forests,
  `GroundCover` on any `Ground` with `relocate` and `dispose`, and `surfaces.merge([])` (see openglcontext
  `plans/GENERATED-TILES.md`).
- The woods' cards take `TreeSpecies.fitted_card_width()`: the forest demo's widths left each card 25-40% narrower than its mesh, which showed as every tree thinning across the 38-52 m hand-over.
- Tiles are made in two worker processes (`tiles.TileMaker`, which returns `TileData` arrays; `HillTiles.assemble` builds the nodes in the game), since threads of the game's process got a few per cent of the interpreter lock from the render thread: a tile took 4-13 s in the game and 0.3 s alone.
- Tiles are built around the start of their chunk and drawn translated to it, so vertex coordinates stay
  small and precise at any distance; textures are laid from the same origin.
- Rules without GL: course, runner, game, pilot, cues, chase, session. A pilot runs generated courses to
  3500 m in every policy for the forks (always, never, by turns) and survives them all.
- Slide clip: `grass-clumps/slide.py` adds `Slide` to the UAL figure; `assets/runner.glb` is its output.
- Scale: trails 0.5 m wide and 0.2 m apart; the camera sits above head height with the feet at the bottom
  of the frame.
- Barriers a metre high across one or two paths, as logs, boulders, gates and walls, with coins kept out of
  their way. Coins are textured; jewels are cut in facets and hang where a jump reaches.
- Countries (forest, rocks, sand, meadow, dense forest) along the hill and a wandering trail. The land is
  cut down to every trail and rises beside it. Where two countries meet the second ground is blended over
  the first.
- Forks: two extra paths on a side draw away through another country and rejoin. The pilot takes or
  declines them.
- Ground: metre- and hand-sized relief, patches of lush and dry ground by vertex colour, rocks, CC0
  photographs for the ground and the trails.
- Forest: the forest demo's trees through `VegetationField`, planted a stretch at a time on a worker, and its
  grass, ferns and flowers through `GroundCover`. Trees keep clear of every path by their crown radius.
- Window: tiles, runner, coins and jewels, HUD, pause, credits and end-of-run screens, music and effects,
  shadows. Where the runner stands, how it is turned and where the camera is come from `Session`.
- Credits printed at start and shown from the menu.
- Damage: three hits end a run, a hit knocks the runner down for 1.3 s (the Roll clip) and resets the speed, 500 m without one heals one; edge shading from the
  engine's `DamageIndicator` (`level`, `hurt_all`). Main screen with Play and Demo.
- Caves: `Course.caves` (a cave takes the place of a fork), `ceiling`, `jump_clear`, `Kind.CREVICE`; the valley over a cave is filled
  to the hills beside it and the tunnel is bored through the fill (`tunnel`, `portal`), rock country is forced in and out of a cave, crevices are cut into the floor
  (`Landscape.footing` is where feet stand); the tunnel's rock, the ground about its openings and the torches are tile meshes with a point light
  at each flame; `atmosphere` darkens the sun, the environment light and the fog; the chase camera ducks.
  The engine binds the nearest lights when a scene has more than the shader's slots.
- Music: Medieval: Market Day (CC0), a faster folk loop than Virtual Rush.

## Open

- Rivers to swim: a swim state with `Swim_Fwd_Loop`, and a country with water.
- A branch always rejoins the trail. A branch that becomes the trail, and a trail that ends, need a course
  that depends on the runner's choices.
- The fork and country schedules are fixed arrays: 3000 forks (about 3000 km) and 6000 countries. Past
  them the hill carries on with the last country and no more forks. A schedule made block by block would
  have no end.
- The game-over screen is latched by the window (`_game_over_shown`); the rest of the window's logic is in
  `Session`.
- The engine's splat ground (`SplatTerrain`, `GroundShading`) reads one control map over a bounded world;
  an endless hill needs a control map that repeats.
- Sign posts at a fork's start; each country's fog and sky colour.
- The slide reads small from behind the runner; a lower camera during a slide would show the pose.
