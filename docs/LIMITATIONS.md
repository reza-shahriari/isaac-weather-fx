# What this does not do

Written down rather than discovered. Planned work lives in [ROADMAP.md](../ROADMAP.md); this page
is about the model as it stands.

## Precipitation

- Rain density is physically derived (Marshall–Palmer), then **thinned by `density_scale`** to a
  renderable count. That is a visual control, not physics.
- Snow's particle count is a direct control. Its **water-equivalent rate** is derived from the
  flake population (`core/meteorology.precipitation_rate_mm_h`), but the count you set is the
  count you get.
- Precipitation does not reduce visibility by itself. Pair it with fog — the presets and the
  randomiser do.
- Particles are real geometry, so they are also visible to RTX lidar. That may or may not be what
  you want.
- Particle positions are written through USD every frame. Large counts (> 50 k) cost frame time.

## Fog

- RTX Simple Fog is an artistic model and its density units are not guaranteed to be physical.
  The `density_calibration` factor exists for that; see the calibration step in
  [the guide](GUIDE.md#first-run-three-things-to-check-once).

## Cloud

- The field is a **60 m grid with 400 m features** by default, with a 15 m detail noise eroding
  the edges (`detail_strength`). The body of a cloud is still 60 m-smooth: soft if the camera is
  close enough to read individual turrets. Lower `cell_m` and `feature_m` and pay for the bake.
- **Hero clouds** are path tracer only, one asset repeated (translates only, so every copy has the
  same orientation), and they do not shade the dome's sun or appear in the real-time dome. They
  are not part of `CloudField`: an infrared model gets them from `WeatherController.hero_clouds()`.
  The Disney cloud is CC BY-SA 3.0: credit it wherever it, or a render of it, is shown.
- The microphysics is **one effective size per genus**, derived water content, and a fixed ice
  fraction. No adiabatic growth of LWC or droplet size with height, no drizzle, and no vertical
  phase transition inside a mixed cloud (the ice fraction is uniform through it).
- The **voxel** deck (the dome and the path tracer's volumes) **tiles horizontally**. The domain is
  `cells × cell_m` across — 15 km at the defaults — and repeats beyond that, which is visible near
  the horizon if you look for it. The per-pixel cloudscape does not: each 32 km tile of the plane
  reads the weather map at its own hashed offset, cross-faded at the borders on the coverage's
  normal score (`core.cloudscape.sample_aperiodic`). That blend is baked into a texture five
  tiles a side, so the cover pattern repeats only every 160 km, twice the 80 km a camera sees.
  The bake is 52 MB and about 3 s per cloudscape. Evaluating the blend inside the march instead
  hung or crashed the kernel in Isaac Sim's CUDA context (CUDA 700, an invalid local-memory
  read) though never outside it; not root-caused.
- Cloud scattering is a multi-octave approximation wrapped in a two-stream albedo envelope, not a
  path trace. It gets the *energy* right and the directional detail approximately; it does not
  produce a fogbow or a glory.
- The march samples 64 times per ray by default. Backlit cloud edges show the step pattern before
  anything else does; raise `march_steps` for a still.

## Sky

- Radiance is returned in **cd/m²** — a photometric quantity, because a dome light wants nits.
  Nothing here is a radiometric claim. A sensor model that needs spectral radiance should take the
  *geometry* and the *cloud field* from this package and do its own radiometry.
- The exposure is an **adaptation**, stated in `core/sky.py`: seven decades compressed to about
  three and a half stops between noon and a full moon. It is a look, not a measurement.
- The sun and the moon are each a single `DistantLight`. No solar limb darkening, no lunar
  libration, no eclipse.
- Refraction is applied to the bodies' altitude (Saemundsson), not to the sky's own geometry.

## Surface weather

- `surface_weather` is a **diurnal model, not a forecast**. What is genuinely diurnal varies;
  what a single state has no basis to vary — wind, cover, visibility, precipitation — is held
  constant. It will not reproduce a frontal passage, a sea breeze or a nocturnal jet.

## Numbers that are estimated rather than measured

Marked as such in the code, and the first things to replace with real data for a specific site:

- the regime weights and parameter ranges in `core/random_weather.py` — a plausible spread for a
  mid-latitude land site, not a climatology;
- the diurnal temperature swing and its cloud damping;
- the turbidity-to-visibility curve, and the turbidity-to-aerosol-optical-depth curve behind the
  beam's colour;
- the built-in presets, which are art-directed starting points.
