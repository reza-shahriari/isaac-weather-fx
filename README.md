# Isaac Weather FX

Fog, rain and snow for **NVIDIA Isaac Sim**, controllable from a UI panel **and** from Python.
Everything an RTX camera renders is affected: the viewport, Replicator render products and ROS camera publishers.

> Status: v0.1, viewport/camera only. Lidar and radar backends are planned (see [ROADMAP.md](ROADMAP.md)).

<!-- Add a GIF here: clear -> fog -> heavy rain -> blizzard -->

## Features

| Effect   | How it works | Main controls |
|----------|--------------|---------------|
| Fog      | RTX Simple Fog render settings, driven by **visibility in meters** (Koschmieder, 5% contrast) | visibility, color, start/end distance, height fog |
| Rain     | Camera-following `PointInstancer`; Marshall-Palmer drop sizes, terminal velocity per drop, streak length = speed x exposure | rate (mm/h), density scale, exposure, drop size range |
| Snow     | Same volume system with fall-speed jitter and sway | flake density, size, fall speed, sway |
| Wind     | Speed + direction with smooth gusts, applied to rain and snow | speed, direction, gusts |
| Lighting | Scales scene light intensities for overcast looks | light scale, include dome lights |
| Sky      | Sun and moon from the real ephemeris for a place, date and hour; a Perez daylight sky baked to an HDR dome, with moonlight and starlight at night | latitude, longitude, date, hour, turbidity, ground albedo, exposure |
| Clouds   | A genuine three-dimensional density field (not an extruded map), shaded with multiple scattering and marched by the dome bake | cover, genus, base height, thickness, optical depth, feature size |

Also included:
- Presets (`clear`, `haze`, `light_fog`, `dense_fog`, `drizzle`, `moderate_rain`, `heavy_rain`, `storm`, `light_snow`, `blizzard`) and JSON save/load.
- A **manual time source** for deterministic synthetic data generation (`seed` + `step(dt)`).
- A **coherent randomiser** (`wx.randomize(seed)` or the panel's *Randomize* button): draws a whole scene from one integer, with the couplings that make it believable -- rain scavenges aerosol so the air between the drops is *clearer*, fog needs a tiny dew-point spread and no wind, snow needs sub-zero air.
- A **surface-meteorology series** (`wx.surface_weather()`): air temperature, humidity, wind, cloud, irradiance, visibility and precipitation on a regular grid, so a thermal or sensor model integrates the same weather the camera is looking at. See [Weather as data](#weather-as-data).
- Non-destructive edits: prims and light changes live in the **session layer**, and render settings are restored when an effect is turned off. Your USD files are never modified.

## Install

1. Clone this repository.
2. In Isaac Sim: **Window > Extensions > ☰ (hamburger) > Settings > Extension Search Paths**, add `<repo>/exts`.
3. Search for **Weather FX** and enable it.
4. Open **Window > Weather FX**.

Tested target: Isaac Sim 5.x / 6.0 (Kit 107+). The update-loop code also falls back to the older Kit API.

## Use from Python

In the Script Editor, or in any script once the extension is enabled:

```python
from weather_fx import api as weather

wx = weather.get_controller()

wx.apply_preset("heavy_rain")
wx.set_fog(visibility_m=150, color=(0.8, 0.8, 0.85))
wx.set_wind(speed_mps=6, direction_deg=45, gust_strength=0.4)

# Several sections in one update
wx.configure(rain={"rate_mm_h": 40}, lighting={"enabled": True, "light_scale": 0.4})

wx.follow("/World/Robot/camera_link/Camera")   # particles follow a sensor camera
wx.save("my_weather.json"); wx.load("my_weather.json")
print(wx.stats())
wx.clear()
```

The UI panel and your scripts share the same controller, so a script change shows up in the panel immediately and vice versa (`wx.on_change(fn)` lets your own code listen too).

### Weather as data

The state describes an instant. A thermal model needs the hours before it, because a surface's
temperature is an integral of its history:

```python
series = wx.surface_weather(hours=48, step_s=1800)
print(series.describe())
# 2024-06-20 10:00 UTC + 48 h, 97 samples  air 13.8 .. 24.2 degC  RH 44 .. 100 %
#   peak DNI 872 W/m2  cloud 38 %  visibility 24.6 km
series.columns()["t_air_k"]     # numpy, SI, ready for a solver
```

The air temperature passes through the state's own value at the state's own hour, so the series
and a frame rendered at that instant describe the same moment. Dew point is held constant and
humidity is derived from it -- that is the way round that gives a night which actually saturates.
See `core/meteorology.py` for what varies, what is held constant, and why.

**Standalone scripts:** see [`examples/standalone_weather_demo.py`](examples/standalone_weather_demo.py). It enables the extension from a script and cycles through presets:

```bash
./python.sh examples/standalone_weather_demo.py              # with UI
./python.sh examples/standalone_weather_demo.py --headless --manual
```

**Deterministic data generation:**

```python
wx.set_general(time_source="manual", seed=42)
for frame in range(N):
    wx.step(1 / 30)
    # ... trigger your Replicator capture ...
```

**Robot cameras moved by physics:** with Fabric, a camera's USD transform may not update while the robot moves. Give the weather an explicit anchor:

```python
wx.set_anchor_provider(lambda: my_camera.get_world_pose()[0])
```

## First-run checklist (please do this once)

1. **Verify fog setting paths.** Run [`tools/dump_rtx_fog_settings.py`](tools/dump_rtx_fog_settings.py) in the Script Editor and compare the output with [`rtx_settings.py`](exts/weather.fx/weather_fx/backends/viewport/rtx_settings.py). Missing paths are skipped with a warning.
2. **Calibrate fog density.** RTX Simple Fog is an artistic model, so its density units are not guaranteed to match physical extinction. Place a dark target at distance `d`, set visibility `V`, and adjust **Fog > Density calibration** (Advanced) until the target's contrast is about `exp(-3.0 * d / V)`. At `d = V`, that is 5%.
3. **Check rain transparency** in your RTX mode. If `preview` drops look solid, try `material = "glass"` or lower the opacity.

## Architecture

```
exts/weather.fx/weather_fx/
├── core/            pure Python + numpy, no Omniverse (unit-tested)
│   ├── state.py       dataclasses + UI metadata  ← single source of truth
│   ├── presets.py     named presets, register your own
│   ├── physics.py     Koschmieder, Marshall-Palmer, terminal velocity, wind/gusts
│   └── particles.py   camera-following toroidal particle volume
├── runtime/
│   ├── manager.py     owns state, fans changes out to backends, time stepping
│   ├── context.py     stage / units / up-axis / anchor for backends
│   └── ticker.py      app-update subscription (Kit 107+ with fallback)
├── backends/
│   ├── base.py        SensorBackend + Effect interfaces
│   ├── viewport/      fog, rain, snow, lighting  (v0.1)
│   ├── lidar.py       planned
│   └── radar.py       planned
├── api.py           WeatherController (what scripts and the UI call)
├── ui/              panel generated from state.py metadata
└── extension.py     Kit entry point + menu
```

The data flow is one way: the UI or a script updates the `WeatherState` through the `WeatherController`. The `WeatherManager` then passes the new state to every registered backend, and it notifies listeners such as the UI.

## Extending

**Add a parameter:** add a field with `param(...)` to a section in `core/state.py`. It appears in the UI, presets, JSON and the API automatically. Then read it in your effect.

**Add a visual effect** (e.g. lens droplets):

```python
from weather_fx.backends.base import Effect

class LensDroplets(Effect):
    name = "lens_droplets"
    def apply_state(self, state, changed): ...
    def update(self, dt, t): ...
    def detach(self): ...

wx.manager.get_backend("viewport").add_effect(LensDroplets())
```

**Add a sensor** (lidar, radar, thermal...): subclass `SensorBackend`, implement `apply_state / update / detach`, then call `wx.register_backend(MyBackend(...))`. It reads the same state as the viewport, so one UI or script change drives every sensor consistently.

**Add a preset:** `wx.register_preset("monsoon", {"rain": {"enabled": True, "rate_mm_h": 120}})`.

## Known limitations (v0.1)

- Rain density is physically derived, then thinned by `density_scale` to a renderable count. This is a visual control, not physics.
- Snow density is a direct control. The **water-equivalent rate** is now derived from the flake population (`core/meteorology.precipitation_rate_mm_h`), but the particle count is still what you set.
- Precipitation does not reduce visibility by itself. Pair it with fog (the presets do this).
- Particles are real geometry, so they are also visible to RTX lidar. This may or may not be what you want.
- Particle positions are written through USD every frame. Large counts (>50k) will cost frame time; Fabric/USDRT writes are on the roadmap.
- Presets are art-directed starting points, not calibrated weather.
- The cloud field is a 60 m grid with 400 m features by default: right for a sky seen from the ground, soft if the camera is close enough to read individual turrets. Lower `cell_m` and `feature_m` and pay for the bake.
- `surface_weather` is a diurnal *model*, not a forecast: what is genuinely diurnal (temperature, humidity, irradiance) varies, and what the state has no basis to vary (wind, cover, visibility) is held constant.
- The regime weights, the diurnal swing and the turbidity-to-visibility curve are **estimated** -- a plausible spread for a mid-latitude site, not a climatology.

## Development

```bash
pip install -e ".[dev]"
pytest            # core tests run without Isaac Sim
```

## License

MIT
