# Implementation Plan

A step-by-step plan for every item in [ROADMAP.md](ROADMAP.md). Each step names the files it touches, the approach, how to verify it, and its risks. Steps within a milestone are in the order they should be done.

Two rules apply to every step:
- Pure math goes in `core/` with unit tests. Code that needs Omniverse (`omni`, `pxr`, `carb`) goes in `runtime/` or `backends/`.
- New parameters are added only through `param(...)` in `core/state.py`. The UI, JSON, presets and API pick them up automatically.

---

## v0.1.x: housekeeping (before any new features)

**0.1 Fix the current errors.** Get `pytest` green (`pip install -e ".[dev]"`, then `pytest`). Run the extension in Isaac Sim once and work through the first-run checklist in the README: fog setting paths, fog density calibration, rain transparency.

**0.2 Fill in the placeholders.** `config/extension.toml` still has `authors = ["<your name>"]` and a placeholder `repository` URL. The README is missing its demo GIF.

**0.3 Add a version constant check.** The version is repeated in `pyproject.toml`, `extension.toml`, `weather_fx/__init__.py` and `docs/CHANGELOG.md`. Add a small test that reads all four and asserts they match.

**0.4 Add CI.** A GitHub Actions workflow that runs `pytest` on Python 3.10 to 3.12. Only `core` is testable there, which is enough to catch most regressions.

---

## v0.2: realism and calibration

### 2.1 Precipitation adds haze
*Do this first: it is pure math and changes the presets.*

- **core/physics.py:** add `rain_extinction(rate_mm_h)` and `snow_extinction(number_density, mean_diameter)`. For rain, use an empirical visible-band fit (for example beta ≈ a·R^b in 1/km). For snow, use geometric optics: β ≈ 2 · N · π(D/2)².
- **core/state.py:** add `fog.precip_haze: bool` (default True) and `fog.precip_haze_scale: float` (advanced).
- **backends/viewport/fog.py:** compute the total extinction as fog + rain + snow. Enable RTX fog when fog is off but precipitation is on and `precip_haze` is set. `apply_state` must also react when the `rain`, `snow` or `general` sections change, not only `fog`.
- **core/presets.py:** re-tune the rain and snow presets, because the haze now comes partly from the precipitation itself. Update the "Known limitations" section of the README.
- **Tests:** extinction increases monotonically with rate and density, the value at the reference rate matches the chosen fit, and zero rate gives zero extinction.
- **Verify:** `wx.stats()["viewport"]["fog"]` reports the combined extinction.

### 2.2 Fog calibration script
- **tools/calibrate_fog.py** (run in the Script Editor): build a test stage with a row of black and white target panels at known distances under a fixed light. Add a render product and capture RGB through Replicator with the `LdrColor` annotator, stepping with the manual time source.
- **core/calibration.py** (pure, tested): given the measured contrast per distance and the chosen visibility, fit `density_calibration` so that contrast(d) ≈ exp(-3·d/V). This is a 1-D least-squares fit in log space.
- The script prints the fitted value and can optionally save it as the default. Store the result per RTX mode (RealTime / PathTracing), because the renderers differ. This could go in an optional `fog.calibration_by_mode` or a settings key.
- **Tests:** feed synthetic contrast curves with a known scale factor and check the fit recovers it.
- **Risk:** tone mapping and auto-exposure distort measured contrast. The script must fix the exposure and ideally read HDR (the `HdrColor` annotator).

### 2.3 Optional volumetric fog mode
- **core/state.py:** add `fog.mode: "simple" | "volumetric"` (advanced).
- **backends/viewport/rtx_settings.py:** add a `VOLUMETRIC_SETTINGS` map for RTX Global Volumetric Effects. Extend `tools/dump_rtx_fog_settings.py` to dump those paths too.
- **backends/viewport/fog.py:** split the setting writers into a `SimpleFogWriter` and a `VolumetricFogWriter` that share the `_write` / `_restore` bookkeeping (originals dict, missing-path warnings). Switching modes restores the old mode before the new one is applied.
- **Verify:** light shafts and lights look correct in fog. The frame-time cost appears in `stats()`.
- **Risk:** the volumetric setting names differ between Kit versions. Handle missing paths the same way simple fog does.

### 2.4 Fabric/USDRT particle writes (100k+ particles)
*Do this before the wet-surface and lens-droplet steps, which add more per-frame work.*

- **backends/viewport/precipitation.py:** separate the per-frame writer from the USD authoring. Keep USD authoring (session layer) for creating prims, materials and proto indices. Add a `FabricWriter` that uses `usdrt.Usd.Stage.Attach(stage_id)` to write `positions`, `scales` and `orientations` straight to Fabric every frame. Fall back to the current `UsdWriter` when `usdrt` is not available.
- **core/state.py:** add `general.writer: "auto" | "usd" | "fabric"` (advanced) and raise the `max_particles` limit.
- **Check:** does the RTX renderer pick up Fabric-only changes to a PointInstancer without USD notices? Test this first, in a spike, before building the rest.
- **Benchmark:** add `tools/benchmark_particles.py`, which measures ms/frame for 10k, 50k, 100k and 200k particles on each writer. The results feed the benchmark table in 2.7.
- **Risk:** a Fabric write followed by a USD write of the same attribute can conflict. Always write through one writer per session.

### 2.5 Wet surfaces, puddles, snow accumulation
*The largest v0.2 item. Build it in the three stages below.*

1. **Wetness (darkening and lower roughness):**
   - **core/state.py:** add a `surface` section: `wetness` (0 to 1, or automatic from rain rate and time), `auto_wetness`, `drying_time_s`, and `include_paths` / `exclude_paths`.
   - **core/physics.py:** a wetness accumulation model, dW/dt = k·R·(1−W) − W/τ_dry. Pure and tested.
   - **backends/viewport/surfaces.py (a new Effect):** find the bound materials under the include paths. Create session-layer overrides that multiply the albedo (darker) and scale the roughness down by the wetness. This works for `UsdPreviewSurface` and for `OmniPBR` inputs (`diffuse_tint`, `reflection_roughness_constant`). Remember the originals and restore them on detach, as `lighting.py` does.
   - **Risk:** materials with textures, or MDL materials without these inputs, cannot be tinted this way. Log them once and skip them.
2. **Puddles:** an optional decal or overlay on horizontal ground. Start with a flat mirror-like material overlay using a noise mask as opacity. This could be an MDL layered material, so the details belong in a research spike.
3. **Snow accumulation:** a snow cover amount grows with snow density and time. Blend upward-facing surfaces toward white, using a normal·up mask in an MDL layered material or a projected overlay mesh on the ground plane. Start with the ground plane only.

- **Verify:** switching surfaces off returns every material to its exact original values. Add a test for the restore bookkeeping using a fake stage, or a doc-tested helper.

### 2.6 Lens droplets and splashes as a camera effect
- **backends/viewport/lens.py (a new Effect):** the first version is geometry. Place small refracting quads or spheres (OmniGlass) just in front of the followed camera's near plane, parented to the camera prim in the session layer. Droplets spawn at a rate that depends on rain rate × how much the camera faces into the wind and rain, then run down the lens and expire.
- **core/lens.py:** the pure spawn, lifetime and slide model in 2-D lens coordinates, with tests.
- **Splashes:** short-lived, small particle bursts on the ground plane under the rain volume. Reuse `ParticleField` with a ground-hit check (ray casts are too expensive, so use the ground plane height from the up axis and a configured `ground_height_m`).
- **Alternative:** a post-process or render-product pass. It looks better but is much harder in Kit, so leave it for later.

### 2.7 Demo GIFs and a benchmark table
- Script the capture in `examples/record_demo.py`: manual time source, apply the preset sequence, capture frames through Replicator into `captures/` (already in `.gitignore`), and assemble the GIF with ffmpeg.
- Add the GIF and the benchmark table from 2.4 to the README.

**v0.2 release:** bump the version in all four places, update CHANGELOG, and tick the roadmap.

---

## v0.3: other sensors

### 3.0 Shared prerequisite: per-sensor state resolution
Do this before the lidar and radar backends so they are built on it.

- **core/state.py:** add `resolve(state, overrides: dict) -> WeatherState`, which applies a partial override dict on top of a state.
- **runtime/manager.py:** keep `overrides: Dict[backend_name, dict]`. `set_state` sends each backend `resolve(state, overrides[name])`, and `changed` is computed per backend from the resolved states. `update()` calls pass the resolved state through the context.
- **api.py:** add `wx.set_sensor_override("lidar", fog={"visibility_m": 50})` and `clear_sensor_override(name)`.
- **Tests:** resolution works, and the changed-sections calculation takes overrides into account.
- This also delivers the roadmap item "Per-sensor weather overrides".

### 3.1 Hide precipitation from lidar
- Particles are real geometry. Research how RTX lidar decides what it sees: prim-level visibility or render-product settings (for example the `omni:rtx:...` attributes that stop a prim being hit by non-visual sensors).
- **core/state.py:** add `rain.visible_to_lidar` and `snow.visible_to_lidar` (default True, which keeps current behavior).
- **precipitation.py:** set the matching attribute on the instancer.
- **Risk:** there may be no per-sensor-type visibility control. The fallback is a separate render-product layer, which is costly. Spike this first.

### 3.2 Lidar backend
- **core/lidar_model.py** (pure numpy, tested): the input is point arrays (range, intensity, return, direction) and a state. Steps:
  1. Two-way Beer-Lambert attenuation: `I *= exp(-2·β·r)`, with β = fog + rain + snow extinction (reuse 2.1), converted to the lidar wavelength (a scale factor; 905 nm and 1550 nm differ).
  2. Detection threshold: drop points whose attenuated intensity falls below `min_intensity`.
  3. Fog backscatter: add near-range false returns with a range distribution following Hahner et al., ICCV 2021.
  4. Rain and snow: random dropouts and scattered returns at a rate based on hydrometeor density. Skip this when the particles are already visible to the lidar (3.1), to avoid counting them twice.
  - Seeded with `general.seed` so results are deterministic.
- **backends/lidar.py:** `LidarBackend(sensor_prim_path)` attaches to the RTX lidar's point-cloud annotator (for example `RtxSensorCpuIsaacCreateRTXLidarScanBuffer`). It uses a Replicator writer, or a custom annotator node in the pipeline that applies `lidar_model` before publishing, so ROS and Replicator consumers see the weather.
- **core/state.py:** add a `lidar` section (wavelength, min_intensity, backscatter_strength, dropout_scale).
- **Examples:** `examples/lidar_weather_demo.py`.
- **Risk:** the hardest part is inserting processing into the annotator output inside the RTX sensor pipeline. Start with a spike that works as a post-process on a Replicator writer, then move it into the pipeline.

### 3.3 Radar backend
- **core/radar_model.py:** two-way rain attenuation with the ITU-R P.838 specific attenuation γ = k·R^α (dB/km) at the radar frequency (for example 77 GHz). Rain clutter: random detections near the sensor with low RCS and near-zero Doppler plus wind velocity. Fog and snow have a negligible effect by default.
- **backends/radar.py:** attach to the RTX radar detection output, using the same approach as the lidar backend.
- **core/state.py:** add a `radar` section (frequency_ghz, clutter_density, min_rcs).
- **Tests:** attenuation matches ITU tabulated values at reference rates.

**v0.3 release:** add a README section explaining how one state drives the camera, lidar and radar.

---

## Later

### L.1 Thermal/IR band-dependent attenuation
- **core/physics.py:** band-dependent extinction. Fog droplets attenuate LWIR (8 to 14 µm) much less than visible light when droplets are small, depending on the droplet size distribution. Add `extinction_for_band(visibility_m, band, droplet_radius_um)`, using the Kruse or Kim relations as a first approximation.
- The backend depends on how thermal cameras are simulated in Isaac Sim (probably material or emissive based). Research spike first.

### L.2 Replicator randomizer
- **weather_fx/replicator.py:** `rep.randomizer.register(weather)`, so users can write `rep.randomizer.weather(presets=[...], visibility=rep.distribution.uniform(50, 2000), ...)` inside `rep.trigger.on_frame()`. Internally it samples values and calls `wx.configure(...)` and `wx.step(...)` using the manual time source.
- Must use the manual time source and the seed, so captures are reproducible.

### L.3 Isaac Lab event term
- **weather_fx/isaaclab.py:** a function matching Isaac Lab's `EventTerm` signature `(env, env_ids, **params)` that randomizes weather per reset. Caveat: weather is global to the stage, not per environment. Document this, and randomize once per reset of all environments unless per-environment sensor overrides (3.0) make per-environment variation possible.

---

## Suggested order across milestones

0.1–0.4 → 2.1 → 2.4 (spike first) → 2.2 → 2.3 → 3.0 → 2.5 → 2.6 → 2.7 → v0.2 release → 3.1 spike → 3.2 → 3.3 → v0.3 release → L.2 → L.3 → L.1

3.0 (per-sensor overrides) comes early because it is small and changes the backend contract. It is cheaper to change that contract before more effects and backends are written against it.
