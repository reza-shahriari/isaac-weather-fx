# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An NVIDIA Isaac Sim (Kit 107+) extension that adds fog, rain, snow, wind and light dimming, controllable from a UI panel and from Python. The extension lives in `exts/weather.fx/` (Kit manifest in `config/extension.toml`, Python package `weather_fx`). Everything outside `exts/` (tests, examples, tools) is supporting material.

## Commands

```bash
pip install -e ".[dev]"
pytest                                   # all tests; no Isaac Sim needed
pytest tests/test_state.py::test_roundtrip_json   # single test
```

`pyproject.toml` puts `exts/weather.fx` on the pytest path, so tests import `weather_fx.core` directly. Only `core/` is testable outside Kit. Everything else (`runtime/`, `backends/`, `ui/`, `api.py`, `extension.py`) imports `omni`/`pxr`/`carb` and can only run inside Isaac Sim, via the Script Editor or `./python.sh examples/standalone_weather_demo.py [--headless --manual]`.

## Architecture

**The core/Kit boundary is strict.** `weather_fx/core/` is plain Python plus numpy with no Omniverse imports: state, presets, physics formulas, the particle volume math, and a tiny `Signal` observer. `weather_fx/__init__.py` imports the Kit extension class only when `omni.ext` is importable, so `import weather_fx.core` works in plain Python. Keep new pure logic in `core/` so it stays unit-testable, and put `omni`/`pxr` imports inside the runtime/backend layers (often lazily inside functions).

**`core/state.py` is the single source of truth.** `WeatherState` is made of section dataclasses (`general`, `fog`, `wind`, `rain`, `snow`, `lighting`, registered in `SECTION_TYPES`). Each field is declared with `param(...)`, which carries UI metadata (label, min/max, unit, `log_scale`, `choices`, `advanced`, `widget`). That metadata drives:
- the UI panel (`ui/window.py` and `ui/widgets.py` generate widgets from the dataclass fields),
- coercion and clamping in `with_updates` (`coerce_value` casts types, clamps to min/max and validates `choices`; `_cross_validate` fixes min/max pairs),
- JSON save/load (`to_dict` / `from_dict`, with `SCHEMA_VERSION`),
- the Python API (`configure(section={...})`).

Adding a parameter therefore means adding one `param(...)` field and reading it in the relevant effect; nothing else needs registering. `with_updates` is strict by default (unknown keys raise `KeyError`). `from_dict` is lenient by default (unknown keys or sections log a warning), so old or newer JSON still loads. `RainParams` and `SnowParams` both inherit from `PrecipitationParams`.

**Data flow is one way:** UI or script → `WeatherController` (`api.py`, a process-wide singleton from `get_controller()`) → `WeatherManager` (`runtime/manager.py`). The manager diffs the old and new state per section, calls `backend.apply_state(state, changed_sections)` on each registered backend, then emits `state_changed`. The UI subscribes to that signal, which keeps the panel and scripts in sync. Backends receive the live state and must not mutate it; listeners get copies. Backend and listener exceptions are logged and swallowed so one failure does not break the others. Keep that pattern.

**Time:** `runtime/ticker.py` subscribes to app updates (Kit 107+ API, with a fallback for older Kit). The manager only auto-steps when `general.time_source == "wall"`. With `"manual"`, callers drive `step(dt)` for deterministic synthetic data (paired with `general.seed`). `apply_preset` keeps the current `general` section (follow prim, time source, seed).

**Backends and effects** (`backends/base.py`): a `SensorBackend` (for example `ViewportBackend`) holds a list of `Effect`s. Both share the lifecycle `attach(context) → apply_state(state, changed) → update(dt, t) → detach()`. `detach()` must undo everything the backend or effect changed. `lidar.py` and `radar.py` are placeholders for planned backends that will read the same state. `runtime/context.py` (`WeatherContext`) is what backends may know about the app: stage, meters-per-unit, up axis (Y or Z), and the anchor position. Anchor priority: custom `anchor_provider` > `general.follow_prim` > active viewport camera > origin.

**Viewport effects must stay non-destructive:**
- Fog (`backends/viewport/fog.py`) writes RTX Simple Fog carb settings and restores the originals when disabled or detached. All setting paths live in `rtx_settings.py`. Paths missing at runtime are skipped with a warning. `tools/dump_rtx_fog_settings.py` dumps the real paths for verification. Fog is driven by visibility in meters (Koschmieder, 5% contrast) times the `density_calibration` fudge factor, because RTX fog density units are not physical.
- Rain and snow (`precipitation.py`) are a `PointInstancer` under `/WeatherFX` in the **session layer**. It follows the anchor using the toroidal wrapping volume from `core/particles.py`. Parameter changes that alter the particle distribution trigger resampling (see the `_*_SAMPLER_KEYS` tuples), while material and color changes only update the shader. Particle counts come from physics (Marshall-Palmer) thinned by `density_scale` and capped at `max_particles`.
- Sky (`sky.py`) bakes the dome from `core/atmosphere.py` (Hillaire 2020) in a background job
  (`core/jobs.LatestJob`) and hides the stage's own lights (`scene_lights.py`). Clouds are one
  `CloudField` (`core/clouds.py`, also marched by the separate thermal-camera repo, so keep its API
  stable) drawn two ways, chosen by `render_mode.py` from `/rtx/rendermode`: path-traced OpenVDB
  volumes (`clouds_volume.py`) or painted into the dome (real-time). Slow work never runs on the
  update thread unless `general.time_source == "manual"`. The clouds' drift is a function of
  weather time (`core/clouds.cloud_drift_at` / `DriftTrack`, evaluated by the manager each step),
  never a frame-by-frame sum, so a headless consumer can reproduce any moment. The field also
  carries per-genus microphysics (`CloudField.microphysics`), derived from its visible extinction.
- Exposure (`exposure.py`) turns on RTX histogram auto exposure while the viewport runs a real-time mode (Kit 110's RTX Real-Time is black at its fixed exposure) and restores it for the path tracer and on detach.
- Lighting (`lighting.py`) scales `UsdLux` intensities via session-layer opinions and remembers the original values to restore them.
- Never author into the user's root layer. Handle stage units (`meters_per_unit`) and the up axis wherever geometry or positions are computed.

## Extension points (from README)

- Custom effect: subclass `Effect`, then `wx.manager.get_backend("viewport").add_effect(...)`.
- Custom sensor: subclass `SensorBackend`, then `wx.register_backend(...)`.
- Presets: `core/presets.py` (built-ins plus `register_preset(name, overrides)`, where overrides is a partial state dict).

`ROADMAP.md` lists planned work (fog calibration, Fabric/USDRT particle writes, lidar and radar backends). Bump the version in `pyproject.toml`, `config/extension.toml`, `weather_fx/__init__.py` and `docs/CHANGELOG.md` together.
