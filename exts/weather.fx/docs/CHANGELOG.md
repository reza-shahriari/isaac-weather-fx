# Changelog

## [Unreleased]

### Added
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
