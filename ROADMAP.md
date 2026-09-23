# Roadmap

## v0.1 (current): viewport
- [x] Fog (RTX Simple Fog) driven by visibility
- [x] Rain, snow, wind with gusts, light dimming
- [x] Auto-generated UI, presets, JSON, Python API, manual time source
- [x] Backend interface for other sensors

## v0.2: realism and calibration
- [ ] Fog calibration script (target chart + automatic `density_calibration` fit)
- [ ] Optional fog mode using RTX Global Volumetric Effects
- [ ] Precipitation adds haze (visibility coupled to rain rate / snow density)
- [ ] Wet surfaces (roughness/darkening), puddles, snow accumulation
- [ ] Lens droplets / splashes as a camera effect
- [ ] Fabric/USDRT writes for 100k+ particles
- [ ] Demo GIFs and a short benchmark table

## v0.3: other sensors
- [ ] Lidar backend: two-way Beer-Lambert attenuation, dropouts, fog backscatter returns
- [ ] Option to hide precipitation geometry from lidar
- [ ] Radar backend: rain attenuation and clutter
- [ ] Per-sensor weather overrides

## Later
- [ ] Thermal/IR band-dependent attenuation (LWIR vs visible in fog)
- [ ] Replicator randomizer (`rep.randomizer.weather(...)`)
- [ ] Isaac Lab event term for weather domain randomization
