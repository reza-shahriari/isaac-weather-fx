# Architecture

```
exts/weather.fx/weather_fx/
├── core/              pure Python + numpy, no Omniverse — all of it unit-tested
│   ├── state.py         dataclasses + UI metadata  ← the single source of truth
│   ├── celestial.py     NOAA solar position, Meeus lunar theory, refraction, twilight
│   ├── sky.py           Perez daylight, moonlight, starlight, the lat-long bake, exposure
│   ├── clouds.py        the 3-D density field, its morphology and its scattering
│   ├── meteorology.py   irradiance, visibility, precipitation rate, the diurnal series
│   ├── random_weather.py  regime-first random scenes with physical couplings
│   ├── presets.py       named presets; register your own
│   ├── physics.py       Koschmieder, Marshall-Palmer, terminal velocity, wind and gusts
│   ├── particles.py     the camera-following toroidal particle volume
│   └── exr.py           a minimal OpenEXR writer, because the sky needs HDR
├── runtime/
│   ├── manager.py       owns state, fans changes out to backends, steps time
│   ├── context.py       stage / units / up-axis / anchor for backends
│   └── ticker.py        app-update subscription (Kit 107+, with a fallback)
├── backends/
│   ├── base.py          SensorBackend + Effect interfaces
│   ├── viewport/        sky, fog, rain, snow, lighting
│   ├── lidar.py         planned
│   └── radar.py         planned
├── api.py             WeatherController — what scripts and the UI both call
├── ui/                the panel, generated from state.py's metadata
└── extension.py       Kit entry point and menu
```

## The core / Kit boundary is strict

`weather_fx/core/` is plain Python plus numpy with **no Omniverse imports**, and
`weather_fx/__init__.py` imports the Kit extension class only when `omni.ext` is importable. So
`import weather_fx.core` works in a bare Python process, which is why the whole physics layer has
tests that run in three seconds without a renderer.

Keep new pure logic in `core/`. Put `omni` / `pxr` / `carb` imports inside the runtime and backend
layers, often lazily inside functions.

## `core/state.py` is the single source of truth

`WeatherState` is made of section dataclasses — `general`, `sky`, `clouds`, `fog`, `wind`, `rain`,
`snow`, `lighting` — registered in `SECTION_TYPES`. Every field is declared with `param(...)`,
which carries its UI metadata: label, min/max, unit, `log_scale`, `choices`, `advanced`, `widget`.

That one declaration drives:

- the **UI panel** (`ui/window.py` generates widgets from the dataclass fields),
- **coercion and clamping** in `with_updates` (`coerce_value` casts, clamps to min/max, validates
  `choices`; `_cross_validate` fixes min/max pairs),
- **JSON** save and load (`to_dict` / `from_dict`, with `SCHEMA_VERSION`),
- the **Python API** (`configure(section={...})`).

Adding a parameter therefore means adding one `param(...)` field and reading it in the relevant
effect. Nothing else needs registering.

`with_updates` is strict by default — an unknown key raises. `from_dict` is lenient by default —
an unknown key or section logs a warning — so old and newer JSON still load.

## Data flow is one way

UI or script → `WeatherController` (`api.py`, a process-wide singleton from `get_controller()`) →
`WeatherManager` (`runtime/manager.py`). The manager diffs the old and new state per section,
calls `backend.apply_state(state, changed_sections)` on each registered backend, then emits
`state_changed`. The UI subscribes to that signal, which is what keeps the panel and scripts in
sync.

Backends receive the live state and **must not mutate it**; listeners get copies. Backend and
listener exceptions are logged and swallowed, so one failure does not break the others. Keep that
pattern.

## Time

`runtime/ticker.py` subscribes to app updates. The manager only auto-steps when
`general.time_source == "wall"`. With `"manual"`, callers drive `step(dt)` for deterministic
synthetic data, paired with `general.seed`. `apply_preset` keeps the current `general` section, so
the follow prim, the time source and the seed survive a preset change.

## Backends and effects

`backends/base.py` defines the lifecycle both share:
`attach(context) → apply_state(state, changed) → update(dt, t) → detach()`.

**`detach()` must undo everything the backend or effect changed.** `runtime/context.py`
(`WeatherContext`) is the whole of what a backend may know about the app: stage, meters-per-unit,
up axis (Y or Z) and the anchor position. Anchor priority: custom `anchor_provider` >
`general.follow_prim` > active viewport camera > origin.

### Viewport effects stay non-destructive

- **Sky** (`viewport/sky.py`) authors `/WeatherFX/Sky/{Dome,Sun,Moon}` in the **session layer**,
  bakes the environment map to a temporary EXR and re-bakes only when something the sky actually
  depends on changes (`_BAKE_KEYS`). A Z-up stage gets the dome *rotated* rather than the map
  resampled, because rotating a light is exact and resampling a texture is not.
- **Fog** writes RTX Simple Fog carb settings and restores the originals on disable or detach. All
  setting paths live in `rtx_settings.py`; paths missing at runtime are skipped with a warning.
- **Rain and snow** are a `PointInstancer` under `/WeatherFX` in the session layer, following the
  anchor through the toroidal wrapping volume in `core/particles.py`. Parameter changes that alter
  the particle distribution trigger resampling (the `_*_SAMPLER_KEYS` tuples); material and colour
  changes only update the shader.
- **Lighting** scales `UsdLux` intensities via session-layer opinions and remembers the originals.

**Never author into the user's root layer**, and handle stage units and the up axis wherever
geometry or positions are computed.

## Extending

**Add a parameter:** one `param(...)` field in `core/state.py`, then read it in your effect.

**Add a visual effect** (lens droplets, say):

```python
from weather_fx.backends.base import Effect

class LensDroplets(Effect):
    name = "lens_droplets"
    def apply_state(self, state, changed): ...
    def update(self, dt, t): ...
    def detach(self): ...

wx.manager.get_backend("viewport").add_effect(LensDroplets())
```

**Add a sensor** (lidar, radar, thermal…): subclass `SensorBackend`, implement
`apply_state / update / detach`, then `wx.register_backend(MyBackend(...))`. It reads the same
state as the viewport, so one UI or script change drives every sensor consistently.

**Add a preset:** `wx.register_preset("monsoon", {"rain": {"enabled": True, "rate_mm_h": 120}})`.

## Development

```bash
pip install -e ".[dev]"
pytest            # 164 tests, no Isaac Sim needed
```

`pyproject.toml` puts `exts/weather.fx` on the pytest path, so tests import `weather_fx.core`
directly. Only `core/` is testable outside Kit; everything else imports `omni` / `pxr` / `carb`
and runs inside Isaac Sim, via the Script Editor or
`./python.sh examples/standalone_weather_demo.py [--headless --manual]`.
