# Changelog

## [Unreleased]

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
