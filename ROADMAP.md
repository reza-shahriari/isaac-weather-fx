# Roadmap

Planned work lives here, not in the README. What the extension *does not* do today is written
down separately, in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Shipped

### v0.1 — the viewport
- [x] Fog (RTX Simple Fog) driven by visibility in metres
- [x] Rain, snow, wind with gusts, scene light dimming
- [x] Auto-generated UI, presets, JSON save/load, Python API, manual time source
- [x] Backend interface for other sensors

### v0.2 — the sky
- [x] Sun and moon from the real ephemeris (NOAA solar position, Meeus ch. 47, refraction)
- [x] Preetham–Shirley–Smits daylight, a moonlit night from the same distribution, starlight
- [x] A three-dimensional cloud field with per-level morphology and multiple-scattering shading
- [x] Daylight-balanced beam colour, so a low sun is warm rather than merely dim
- [x] HDR dome bake with highlight metering and a stated exposure adaptation
- [x] A coherent random-weather generator, in the panel and in the API
- [x] Surface meteorology: irradiance, visibility, precipitation rate, a diurnal series
- [x] A rendered gallery, and the script that produces it

## v0.3: realism and calibration
- [ ] Fog calibration script (target chart, automatic `density_calibration` fit)
- [ ] Optional fog mode using RTX Global Volumetric Effects
- [ ] Precipitation reduces visibility by itself, rather than needing fog paired with it
- [ ] Wet surfaces (roughness and darkening), puddles, snow accumulation
- [ ] Lens droplets and splashes as a camera effect
- [ ] Fabric/USDRT writes for 100 k+ particles
- [ ] Cloud: break the horizontal tiling, and a cheaper march so a finer grid is affordable
- [ ] Demo GIFs and a short benchmark table

## v0.4: other sensors
- [ ] Lidar backend: two-way Beer–Lambert attenuation, dropouts, fog backscatter returns
- [ ] Option to hide precipitation geometry from lidar
- [ ] Radar backend: rain attenuation and clutter
- [ ] Per-sensor weather overrides

## Later
- [ ] Thermal/IR band-dependent attenuation (LWIR vs visible in fog) inside this package —
      today an infrared consumer takes the cloud field and the geometry and does its own
      radiometry, which is the right split but leaves fog and precipitation unhandled
- [ ] Replicator randomizer (`rep.randomizer.weather(...)`)
- [ ] Isaac Lab event term for weather domain randomisation
- [ ] A real climatology behind the randomiser's regime weights, in place of the estimated spread
