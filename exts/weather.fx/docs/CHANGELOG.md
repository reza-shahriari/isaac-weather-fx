# Changelog

## [Unreleased]

### Added
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
- The dome's cloud layer is marched on `clouds.dome_rows` rows (256) and upsampled, instead of at
  the full dome resolution: a cloudy 1024-row bake went from about 90 s to about 12 s here. The
  sun and moon lights update immediately; the dome texture follows when its bake finishes.
- The dome exposure is re-anchored on the atmosphere's clear-noon 99th percentile (16,000 cd/m2,
  target 400), which keeps the measured clear-noon dome intensity of 0.025.
- **The README is a gallery and a pitch.** The reference material moved to `docs/GUIDE.md`,
  `docs/ARCHITECTURE.md` and `docs/LIMITATIONS.md`, and planned work is only in `ROADMAP.md`.

### Fixed
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
