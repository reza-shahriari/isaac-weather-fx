# Changelog

## [Unreleased]

### Fixed
- **Grey, faded path-traced clouds.** Three settings starved the volume of light:
  - the droplet albedo was 0.96. Water at 0.55 um is 0.99999, and a thick cloud scatters each
    photon hundreds of times, so 4 % per bounce darkened its whole body. Now 0.999;
  - the delta-tracking collision limit was *lowered* from the renderer's 1024 to 128, and the
    shadow-ray limit was 64, cutting paths off inside thick cloud. The limits are now floors
    (1024 and 256) that never lower the user's own values;
  - 8 volume bounces. Now `clouds.volume_bounces`, default 32.
  `phase_bias` defaults to 0.8 (from 0.85), giving the shadowed side a little of the side and back
  scattering that a single Henyey-Greenstein lobe lacks.

### Changed
- **The clouds' drift is a pure function of weather time.** It was a running sum of velocity
  times frame time in the manager's update loop, which a headless render could not reproduce for
  a given moment, so the thermal repo held it at zero while its own wind kept blowing. Now
  `core.clouds.cloud_drift_at(elapsed_s, speed, direction, factor, up_axis, frame)` and
  `cloud_drift_from_state(state, elapsed_s, up_axis, frame)` give it from the time and the wind,
  and the manager evaluates a `DriftTrack` (one linear segment per steady wind; a wind change
  continues from where the clouds are) at its `time` every step. For a wind set before the clock
  starts the manager's drift equals `cloud_drift_from_state(state, t)` exactly. The drift now
  runs whether or not the clouds are drawn. New API: `WeatherController.elapsed_s`,
  `set_elapsed_s(t)`, `cloud_drift_at(t, frame)`; `WeatherManager.set_time(t)`, `reset_time()`.
- **The march's jitter is a per-ray hash** of the ray direction (SplitMix64 over the float bits)
  instead of a smooth function of it. The smooth jitter printed concentric rings through every
  cloud (thermal repo ADR 0146); the hash turns the same residue into per-ray noise. Still
  deterministic, still invariant to moving the observer by whole tiles; `march(jitter_seed=)`
  changes the pattern. The dome bake gets it too.
- **Sharper cloud edges.** With edge detail on, the field interpolates its signed distance from
  the threshold and softens after, instead of interpolating an already-softened grid, so edges are
  a softness-wide band rather than a 60 m ramp. Backlit rims at a low sun are brighter for it; two
  exposure tests that include the 25 degrees around the sun (which the meter excludes) now allow
  sunset highlights up to 1.3x noon's, against the 2.4x defect they guard.

### Added
- **Per-genus cloud microphysics.** `CloudProfile` gains `phase` ("liquid", "ice", "mixed"),
  `effective_radius_um`, `ice_effective_diameter_um` and `ice_fraction`, with estimated values per
  genus (cirrus is ice, congestus mixed with a quarter of its extinction in ice).
  `CloudField.microphysics` gives the liquid and ice water content at unit density, **derived from
  the visible extinction** (geometric optics for water, `beta = 1.5 LWC / r_e`; Fu 1996 for ice)
  so it cannot disagree with the visible render; `liquid_water_content(x, y, z)` and
  `ice_water_content(x, y, z)` sample it. Conversion helpers `liquid_extinction_per_m`,
  `liquid_water_content_g_m3`, `ice_extinction_per_m`, `ice_water_content_g_m3` are exported, so a
  band model can compute its own coefficients from published parameterisations.
- **Fine cloud detail** (`clouds.detail_strength`, default 0.08): a 64-cubed periodic noise at a
  quarter of the field's cell (15 m by default), whose period divides the tile, is added to the
  signed field near its edges at sample time. The core is untouched and every consumer of
  `density()` -- the dome, the volumes, an infrared march -- sees the same detail.
  `clouds.volume_detail` (default 2) voxelises the path-traced volumes that much finer
  (30 m by default) so the detail shows; `CloudField.volume_grid(up_axis, refine)`.
- **Real-time auto exposure** (`backends/viewport/exposure.py`, `general.realtime_auto_exposure`,
  on by default). On Kit 110 RTX Real-Time (`RealTimePathTracing`) renders every daylight frame
  black, with or without the weather: its fixed exposure is orders of magnitude too dark
  (`tools/probe_realtime_renderer.py` measured it: black with the default settings, a normal frame
  with `/rtx/post/histogram/enabled`). The extension now turns the histogram auto exposure on while
  the viewport is in a real-time mode, and restores it for the path tracer and on detach.
- `WeatherController.diagnose()`: renderer detected, cloud path, every prim under `/WeatherFX`
  with visibility and intensity, the dome texture and whether it exists, hidden stage lights and
  the live RTX fog settings, in one printout.
- **Clouds as path-traced volumes** (`backends/viewport/clouds_volume.py`). Under the path tracer
  the cloud field is written to OpenVDB (the binding inside `omni.volume`) and placed as volume
  boxes under `/WeatherFX/Clouds`, with the recipe the thermal-camera project measured to render
  (mesh box at the grid's world bounds, no transform, `primvars:isVolume`, `OmniVolumeDensity`,
  and the non-uniform-volume render settings, restored on detach). Clouds now have parallax, an
  inside and a shadow. Under real-time they stay in the dome. `clouds.render_path` ("auto" by
  default) picks by the viewport's renderer and switches when it changes.
- Cloud controls: `lit_color`, `shadow_color` (dome), `density_scale` and `phase_bias`. On
  volumes these are material inputs and change instantly.
- `core/jobs.LatestJob`: slow work (the cloud field, the dome bake, the VDB files) runs in a
  worker thread and only the newest request is kept, so the UI no longer freezes. With
  `general.time_source = "manual"` the work stays synchronous, for deterministic frames.
- `core/clouds.cloud_field_from_state` (cached per cloud shape), `CloudField.volume_grid` (the
  density in stage axes, voxel for voxel the field) and `tile_placements`.
- `examples/check_sky_and_clouds.py`: a Script Editor walkthrough of the new behaviour.
- **Clouds drift with the wind** (`clouds.wind_factor`, 1.5 x the surface wind by default). The
  manager integrates the drift in `step` (`context.cloud_drift_m`), so manual time is
  deterministic; `WeatherController.cloud_drift_m()` hands it to sensor models that march the
  field. Volumes move by one translate on `/WeatherFX/Clouds`, which also recentres them by whole
  tiles so the camera never reaches an edge. The dome re-bakes every 250 m of drift.
- **Auto white balance** (`sky.white_balance`, 0.9): the dome, the sun and the moon are balanced
  by one set of gains toward the scene's white point, weighted to the direct beam as eyes and
  cameras are (`core/sky.white_balance_gains`, `scene_illuminant`). Morning clouds are white, not
  orange; the last minutes before sunset stay warm.
- `core/sky.sun_colour`: the visible sun's colour from its measured colour temperature against
  elevation (2,000 K on the horizon to 5,800 K overhead). `meteorology.beam_tint` is unchanged for
  the thermal model.
- **Cloud shadows under real-time** (`clouds.cast_shadow`): the sun is dimmed by the cloud's
  transmittance toward it at the camera. The path tracer's volumes cast real shadows.
- **A physically based atmosphere** (`core/atmosphere.py`), the technique behind Unreal's Sky
  Atmosphere (Hillaire 2020): transmittance, multiple-scattering and sky-view tables over a round
  planet, with Rayleigh, Mie and ozone. It is the new default (`sky.model = "atmosphere"`); the
  Preetham fit stays selectable. Twilight comes from the geometry, the horizon brightens because a
  grazing ray crosses a thousand kilometres of air, and the ground below the horizon is seen
  through the same air. A clear sky builds in about 0.2 s once the tables are cached.
- `sky.horizon_blend_deg`: fades the dome's ground into the sky over a band below the horizon,
  replacing the hard line at the observer's five-kilometre geometric horizon.
- `sky.aerial_perspective` (on by default): with fog off, RTX fog is driven by the sky's own haze,
  at the visibility the turbidity implies (`atmosphere.haze_visibility_m`) and in the sky's horizon
  colour, so distant geometry fades the way the sky does.
- `sky.hide_scene_lights` (on by default): while the sky is authored, every light outside
  `/WeatherFX` is hidden in the session layer, and restored on detach. A new stage's
  `defaultLight` was a second sun.

- **A rendered gallery** (`captures/gallery/`) and the script that produces it,
  `examples/capture_gallery.py`. Thirteen scenarios, two cameras, one stage with **no lights** —
  every photon comes from the sky this extension authors. The clocks are solved from the
  ephemeris rather than typed: "golden hour" is the hour at which the sun is genuinely four
  degrees up at that place on that date, and the full-moon frame is a real date found by search.
- `beam_tint` in `core/meteorology.py`: the colour of the direct beam at a given elevation, from
  Rayleigh plus an Angstrom aerosol on the Kasten-Young airmass, **balanced for daylight** and
  luminance-preserving. It is applied to the cloud's illumination and to the sun and moon lights,
  so a low sun is warm rather than merely dim.
- `core/sky.dome_exposure`: the dome-light intensity, metered on the sky's 99th percentile with a
  stated adaptation exponent. Lives in `core` so it is testable without a renderer.
- `clouds.march_steps`: samples per ray when the dome is baked. 64 remains the viewport default;
  a still that someone will look at closely wants more, because the step pattern is visible on a
  backlit cloud edge before anything else is.

### Changed
- **Exposure is an incident-light meter** (`dome_exposure(image, conditions)`): it exposes for the
  light on the ground, capped so the sky's highlights stay within three stops, and meters the sky
  away from a 25 degree cone around the sun and moon. The sky-highlight meter exposed for the
  moon's glow and left moonlit nights black; a full-moon night now sits about 3.6 stops under
  noon and a moonless one about 4. Midday is unchanged (dome intensity 0.025).
- The dome lit clouds by moonlight five times too dimly for the moon's illuminance; fixed.
- Dragging the site, the clock or the turbidity no longer stalls the UI: the distance haze's
  colour is computed in the background, and the stage is scanned for lights once, not per step.
- The dome's cloud layer is marched on `clouds.dome_rows` rows (256) and upsampled, instead of at
  the full dome resolution: a cloudy 1024-row bake went from about 90 s to about 12 s here. The
  sun and moon lights update immediately; the dome texture follows when its bake finishes.
- The dome exposure is re-anchored on the atmosphere's clear-noon 99th percentile (16,000 cd/m2,
  target 400), which keeps the measured clear-noon dome intensity of 0.025.
- **The README is a gallery and a pitch.** The reference material moved to `docs/GUIDE.md`,
  `docs/ARCHITECTURE.md` and `docs/LIMITATIONS.md`, and planned work is only in `ROADMAP.md`.

### Fixed
- **Black real-time frame.** RTX fog colours are in renderer units, where the authored dome is
  drawn at about 150; the fog was written at intensity 1, so in real-time (the path tracer ignores
  RTX fog) it fogged the sky at infinity, and everything distant, to black. The sky effect now
  publishes its horizon colour in renderer units (`context.sky_horizon_rgb`) and both the user's
  fog and the distance haze are scaled by it. Distance haze is now off by default.
- The renderer is read from the active viewport's own render mode before `/rtx/rendermode`,
  which the viewport menu does not update on every build (the cloud volumes stayed in a real-time
  viewport). Mode changes are logged.
- The quick first dome bake and the full bake of the same sky wrote the same file (the name kept
  only 18 characters of the key), so the full bake overwrote the texture the renderer was
  reading: "Unexpected data block y coordinate" from OpenEXR and a black dome. Each bake now gets
  its own file.
- All cloud volume tiles share one VDB file, each tile placed by a translate, instead of nine
  identical files: nine times less writing under the interpreter lock and nine times less texture
  memory.
- **Every sky in a session was the first one.** The baked environment map was written to one path
  per process, and a renderer caches a texture by its path — so a thirteen-scenario sweep
  rendered thirteen different skies as one. Overwriting the file while the loader still held it
  also produced `Unexpected data block y coordinate` from OpenEXR: a half-written file, read as a
  whole one. The name now carries the bake key.
- **The sun and the moon were not exposed with the dome.** The dome's intensity carried the
  exposure and the lights did not, and they are quoted in different units, so the sun sat about
  two and a half decades above the sky it shared a frame with: the sky rendered correctly and
  every surface the sun touched clipped to white.
- **The dome was metered on the median of the sky**, which fails in both directions. At a four
  degree sun the dim anti-solar half dragged the median down, the exposure up, and the whole
  solar side clipped. At civil twilight the median of the upper hemisphere is zero to float
  precision and the exposure ran to its clamp — a white frame at the darkest moment of the day.
- **The randomiser's seasonal draw was anchored on the new year** rather than the warmest day,
  putting the northern hemisphere's maximum in mid-January: a December at 59 degN drew 17 degC.
- Cloud illumination attributed the whole broadband extinction to a wavelength^-1.3 aerosol,
  which over-reddened everything; water vapour, ozone and the mixed gases absorb broadly and do
  not redden. Midday clouds came out correctly lit and visibly khaki.

### Added
- Sky: sun and moon from the real ephemeris (NOAA SPA, Meeus ch. 47), a Preetham-Shirley-Smits
  daylight sky with moonlight and starlight, baked to an HDR dome light with the sun and moon
  authored as `DistantLight`s. New `sky` state section and UI group.
- Clouds: a three-dimensional density field with per-level area morphology, multi-octave multiple
  scattering and a two-stream albedo envelope. New `clouds` state section and UI group.
- A minimal OpenEXR writer (`core/exr.py`) so the dome texture can carry the six decades between
  noon and a moonless night.
- A coherent random-weather generator (`core/random_weather.py`), `wx.randomize(seed)` and a
  *Randomize* button in the panel.
- Surface meteorology (`core/meteorology.py`): irradiance from the real sun elevation with
  Kasten-Czeplak cloud attenuation, turbidity-to-visibility, snow water equivalent derived from
  the flake population, and `diurnal_series` / `wx.surface_weather()` -- a full diurnal series for
  a thermal or sensor model, anchored on the state's own instant.

### Fixed
- The seasonal temperature draw was anchored on the new year rather than the warmest day, which
  put the northern hemisphere's maximum in mid-January. A December at 59 degN drew 17 degC.
- The clear-sky irradiance law lived in the viewport backend, where nothing outside Kit could
  test or reuse it. It is now `core.meteorology.clear_sky_irradiance` and the backend calls it.

## [0.1.0]
- Fog via RTX Simple Fog, driven by visibility in meters.
- Rain (Marshall-Palmer drop sizes, terminal velocity, exposure-length streaks).
- Snow with fall-speed jitter and sway.
- Wind with gusts, scene light dimming.
- Auto-generated UI panel, presets, JSON save/load, Python API.
- Backend interface for future lidar and radar support.
