# Using Weather FX

Everything the panel does, a script can do, and the two stay in sync because they share one
controller. This page is the long form; [the README](../README.md) has the sixty-second version.

---

## Install

1. Clone this repository.
2. In Isaac Sim: **Window → Extensions → ☰ → Settings → Extension Search Paths**, add `<repo>/exts`.
3. Search for **Weather FX** and enable it.
4. Open **Window → Weather FX**.

Tested target: Isaac Sim 5.x / 6.0 (Kit 107+). The update-loop code also falls back to the older
Kit API.

---

## The controller

In the Script Editor, or in any script once the extension is enabled:

```python
from weather_fx import api as weather

wx = weather.get_controller()
```

The UI panel and your scripts share that object, so a script change shows up in the panel
immediately and vice versa. `wx.on_change(fn)` lets your own code listen.

### Weather

```python
wx.apply_preset("heavy_rain")
wx.set_fog(visibility_m=150, color=(0.8, 0.8, 0.85))
wx.set_wind(speed_mps=6, direction_deg=45, gust_strength=0.4)
wx.set_rain(rate_mm_h=40)
wx.set_snow(number_density_m3=30, fall_speed_mps=1.2)

# Several sections in one update
wx.configure(rain={"rate_mm_h": 40}, lighting={"enabled": True, "light_scale": 0.4})
```

### Sky, sun and moon

```python
wx.set_site(latitude_deg=48.14, longitude_deg=11.58)
wx.set_time(date_utc="2024-06-21", hour_utc=18.5)
wx.set_sky(turbidity=3.4, star_intensity=1.2, dome_resolution=2048)
wx.set_clouds(enabled=True, cover=0.35, genus="cumulus",
              temperature_c=24.0, dewpoint_c=12.0)

print(wx.sky_conditions().describe())
# 2024-06-21 18:30 UTC  sun +5.7 deg el / 300 deg az  moon -6.1 deg, full (100 %)
#   turbidity 3.4  cloud 34 % cumulus
```

The sun and moon come from the real ephemeris — NOAA's solar position algorithm and Meeus's
truncated lunar theory — so the site and the clock are the only inputs. `base_m = 0` puts the
cloud base at the lifting condensation level implied by the temperature and the dew point, which
is why a whole field of cumulus has its bases on one level.

Cloud genera: `cumulus`, `congestus`, `stratocumulus`, `stratus`, `cirrus`.

### A whole scene from one integer

```python
print(wx.randomize(11))
# broken cumulus, larger and deeper
```

The draw is **regime first**, then the parameters within it, with the couplings that make a day
hang together: rain implies a deep low deck, reduced visibility and a *lower* turbidity, because
rain scavenges aerosol; fog implies a tiny dew-point spread, almost no wind and no convective
cloud above it; snow implies sub-zero air. Drawing each parameter independently produces days
that cannot happen, and a model trained on those learns the noise.

The clock is drawn in **local solar time** and converted, so "night" means night where the scene
is. Over 160 draws, 39 % land with the sun below the horizon.

```python
from weather_fx.core.random_weather import random_sequence
for state in random_sequence(seed=0, count=500):    # a dataset's worth, reproducible
    ...
```

### Weather as data

The state describes an instant, which is all a renderer needs. A *thermal* model needs the hours
before it, because a surface's temperature is an integral of its history:

```python
series = wx.surface_weather(hours=48, step_s=1800)
print(series.describe())
# 2024-06-20 10:00 UTC + 48 h, 97 samples  air 13.8 .. 24.2 degC  RH 44 .. 100 %
#   peak DNI 872 W/m2  cloud 38 %  visibility 24.6 km
series.columns()["t_air_k"]     # numpy, SI, ready for a solver
```

| column | what varies |
|---|---|
| `t_air_k` | sinusoid about a solved mean, peaking mid-afternoon in solar time, damped by cloud and fog, **floored at the dew point** |
| `rh_fraction` | derived, from a *constant* dew point and the varying air temperature |
| `dni_w_m2`, `dhi_w_m2` | computed per step from the real sun elevation, the turbidity and the cover |
| `cloud_fraction`, `wind_speed_m_s`, `visibility_m`, `precip_mm_h` | constant — one state carries one of each |

Two choices worth knowing about. The air temperature is floored at the dew point because
nocturnal cooling stalls at saturation: further cooling condenses water and releases its latent
heat instead of dropping the temperature, and a bare sinusoid will happily drive a humid night
4 K past it. And the **dew point is the constant, humidity is derived** — the other way round
gives a night that never saturates, so nothing ever forms dew or fog.

Irradiance under cloud uses Kasten–Czeplak, so full overcast keeps about **25 %** of the global
rather than the `1 − cover` a naive model gives, and broken cloud puts the *diffuse* above its
clear-sky value, because the bright side of a cumulus is a light source.

### Presets and files

```python
wx.list_presets()
# ['clear', 'haze', 'light_fog', 'dense_fog', 'drizzle', 'moderate_rain',
#  'heavy_rain', 'storm', 'light_snow', 'blizzard']
wx.register_preset("monsoon", {"rain": {"enabled": True, "rate_mm_h": 120}})
wx.save("my_weather.json"); wx.load("my_weather.json")
wx.clear()
```

### Deterministic data generation

```python
wx.set_general(time_source="manual", seed=42)
for frame in range(N):
    wx.step(1 / 30)
    # ... trigger your Replicator capture ...
```

### Robot cameras moved by physics

With Fabric, a camera's USD transform may not update while the robot moves. Give the weather an
explicit anchor:

```python
wx.follow("/World/Robot/camera_link/Camera")
wx.set_anchor_provider(lambda: my_camera.get_world_pose()[0])
```

---

## Standalone scripts

```bash
./python.sh examples/standalone_weather_demo.py              # with UI
./python.sh examples/standalone_weather_demo.py --headless --manual
./python.sh examples/capture_presets.py                      # one frame per preset
./python.sh examples/capture_gallery.py --asset /path/to/drone.usdc
```

`examples/capture_gallery.py` is what renders [the README gallery](../README.md#gallery). It is
worth reading as an example of driving the sky from code: the scene it builds has **no lights at
all**, every scenario's clock is solved from the ephemeris rather than typed, and the cameras are
aimed at whichever body is lighting the scene.

---

## First run: three things to check once

1. **Verify fog setting paths.** Run [`tools/dump_rtx_fog_settings.py`](../tools/dump_rtx_fog_settings.py)
   in the Script Editor and compare with
   [`rtx_settings.py`](../exts/weather.fx/weather_fx/backends/viewport/rtx_settings.py). Missing
   paths are skipped with a warning.
2. **Calibrate fog density.** RTX Simple Fog is an artistic model, so its density units are not
   guaranteed to match physical extinction. Place a dark target at distance `d`, set visibility
   `V`, and adjust **Fog → Density calibration** (Advanced) until the target's contrast is about
   `exp(-3.0 * d / V)`. At `d = V`, that is 5 %.
3. **Check rain transparency** in your RTX mode. If `preview` drops look solid, try
   `material = "glass"` or lower the opacity.

---

## Exposure, and why the sky is not "too bright"

The sky spans about **seven decades** of luminance between noon and a moonless night, and a
display spans two. The dome texture stays in honest cd/m², and the dome light's *intensity*
carries the exposure, which is metered on the **99th percentile of the sky** — not the median. A
median is the wrong statistic in both directions: at sunset the dim anti-solar half drags it down,
the exposure up, and the solar side clips; at civil twilight the median of the upper hemisphere is
zero to float precision.

The compression itself is a stated tone decision (`EXPOSURE_ADAPTATION = 0.83` in
`core/sky.py`): a full moon lands about three and a half stops below noon, rather than either
matching it or being black. `wx.set_sky(exposure_scale=...)` scales the result if you want a
different look.
