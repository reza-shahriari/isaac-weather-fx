"""The sky: daylight, twilight, night, and the cloud in front of all three.

This is the piece that turns :mod:`weather_fx.core.celestial` and :mod:`weather_fx.core.clouds`
into something a renderer can bind and a sensor model can integrate. It is still pure numpy: the
environment map is generated here and *written* by the viewport backend, which is the only part
that needs Kit.

**Daylight** is the Preetham-Shirley-Smits analytic model (SIGGRAPH 1999): a Perez sky-luminance
distribution scaled to an absolute zenith luminance, with the CIE xy chromaticity distributed by
the same functional form. Two inputs, both of which the scene already has -- the sun's zenith
angle and the atmospheric turbidity.

**Night** is the moon and the stars, and it matters more than it looks. A night sky rendered black
is not dark, it is *empty*: a camera with any gain at all sees a moonlit landscape, and an
infrared sensor sees a sky that is cold but not uniform. The moon is a second sun four hundred
thousand times fainter, and its light is modelled the same way -- same Perez distribution, same
geometry, scaled to the moon's own illuminance and tinted by the eye's scotopic shift.

**Twilight** is the hard part and is handled by construction rather than by a blend: the daylight
model is evaluated at the sun's real position and multiplied by a smooth fall-off through the
civil-to-astronomical band, while the moon and star terms are added unconditionally. So the sun
setting is one continuous function, not a switch, and a time-lapse through it does not pop.

**Cloud** is composited by the march in :mod:`weather_fx.core.clouds`, which means the cloud in
the visible dome and the cloud an infrared band integrates are the same array. That is the
property the whole sky subsystem exists to guarantee, and it is the one the previous
implementation could not: it painted a dome from one model and marched another.

Absolute units: sky radiance is returned in **cd/m2** (a photometric quantity, because the dome
is a light and Kit wants nits), and the caller applies its own exposure. Nothing in this module
is a radiometric claim -- an infrared sensor takes the *geometry* and the *cloud field* from here
and does its own radiometry, because the visible photometry has no defensible LWIR analogue.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

import numpy as np

from weather_fx.core.celestial import (
    BodyPosition,
    MoonPosition,
    civil_twilight_fraction,
    moon_illuminance_lux,
    moon_position,
    sun_position,
)
from weather_fx.core import atmosphere
from weather_fx.core.clouds import CloudField, cloud_field_from_state
from weather_fx.core.meteorology import beam_tint

__all__ = [
    "MOONLIGHT_TINT",
    "STARLIGHT_LUX",
    "SkyConditions",
    "conditions_from_state",
    "latlong_directions",
    "perez",
    "sky_luminance_cd_m2",
    "sky_radiance_rgb",
    "dome_exposure",
    "scene_illuminant",
    "sun_colour",
    "white_balance_gains",
    "environment_map",
]

#: Colour of moonlight as a camera records it. The moon's own spectrum is sunlight off grey rock
#: and is very slightly *warmer* than the sun; the blue cast everyone associates with moonlight is
#: the human scotopic shift, not the light. A camera has no rods, so the tint here is mild and is
#: a look, not a measurement.
MOONLIGHT_TINT = (0.84, 0.90, 1.0)
#: Horizontal illuminance from a clear moonless night sky: starlight plus airglow, lux. About a
#: hundredth of a full moon, and the floor that stops a night render being a black frame.
STARLIGHT_LUX = 0.002
#: Luminous efficacy used to move between the sky's photometric units and an irradiance. Daylight
#: through the atmosphere, lm/W.
LUMINOUS_EFFICACY_DAYLIGHT = 105.0


#: Preetham table 2: the linear-in-turbidity fits of the five Perez coefficients, for each of the
#: three channels the model distributes. **All three are needed.** Reusing the luminance
#: coefficients for the chromaticity -- which is the shortcut every reimplementation reaches for,
#: because it looks like it only rescales a colour -- makes the sky get *bluer* with turbidity
#: instead of whiter. Haze whitening the sky is one of the two things turbidity is for.
_PEREZ_FITS = {
    "Y": ((0.1787, -1.4630), (-0.3554, 0.4275), (-0.0227, 5.3251),
          (0.1206, -2.5771), (-0.0670, 0.3703)),
    "x": ((-0.0193, -0.2592), (-0.0665, 0.0008), (-0.0004, 0.2125),
          (-0.0641, -0.8989), (-0.0033, 0.0452)),
    "y": ((-0.0167, -0.2608), (-0.0950, 0.0092), (-0.0079, 0.2102),
          (-0.0441, -1.6537), (-0.0109, 0.0529)),
}


def _perez_coefficients(turbidity: float, channel: str = "Y") -> Tuple[float, ...]:
    """Perez A-E for one channel, from Preetham's table 2 linear fits in turbidity."""
    t = float(turbidity)
    return tuple(slope * t + intercept for slope, intercept in _PEREZ_FITS[channel])


def perez(
    cos_theta: Any, gamma: Any, coefficients: Tuple[float, ...]
) -> np.ndarray:
    """The Perez sky distribution ``F(theta, gamma)``.

    ``cos_theta`` is the cosine of the angle from the zenith and ``gamma`` the angle from the sun.
    The ``1e-4`` floor on ``cos_theta`` is not cosmetic: at the horizon the ``B / cos_theta``
    term diverges, and a sky model that returns an infinity there bakes a ring of NaN into the
    environment map.
    """
    a, b, c, d, e = coefficients
    ct = np.maximum(np.asarray(cos_theta, dtype=np.float64), 1e-4)
    g = np.asarray(gamma, dtype=np.float64)
    return (1.0 + a * np.exp(b / ct)) * (
        1.0 + c * np.exp(d * g) + e * np.cos(g) ** 2
    )


def _zenith_luminance_cd_m2(turbidity: float, sun_zenith: float) -> float:
    """Preetham's absolute zenith luminance, kcd/m2 converted to cd/m2."""
    chi = (4.0 / 9.0 - turbidity / 120.0) * (math.pi - 2.0 * sun_zenith)
    return 1000.0 * ((4.0453 * turbidity - 4.9710) * math.tan(chi) - 0.2155 * sun_zenith + 2.4192)


def _zenith_chromaticity(turbidity: float, sun_zenith: float) -> Tuple[float, float]:
    """Preetham's zenith CIE x and y, by the published quartic in the sun's zenith angle."""
    t, z = float(turbidity), float(sun_zenith)
    t2, z2, z3 = t * t, z * z, z * z * z
    x = (
        (0.00166 * z3 - 0.00375 * z2 + 0.00209 * z) * t2
        + (-0.02903 * z3 + 0.06377 * z2 - 0.03202 * z + 0.00394) * t
        + (0.11693 * z3 - 0.21196 * z2 + 0.06052 * z + 0.25886)
    )
    y = (
        (0.00275 * z3 - 0.00610 * z2 + 0.00317 * z) * t2
        + (-0.04214 * z3 + 0.08970 * z2 - 0.04153 * z + 0.00516) * t
        + (0.15346 * z3 - 0.26756 * z2 + 0.06670 * z + 0.26688)
    )
    return float(x), float(y)


def _xyy_to_linear_rgb(luminance: Any, x: Any, y: Any) -> np.ndarray:
    """CIE xyY to linear sRGB. Clamped at zero: the gamut corners can go slightly negative."""
    lum = np.asarray(luminance, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.maximum(np.asarray(y, dtype=np.float64), 1e-6)
    big_x = lum * x / y
    big_z = lum * (1.0 - x - y) / y
    r = 3.2406 * big_x - 1.5372 * lum - 0.4986 * big_z
    g = -0.9689 * big_x + 1.8758 * lum + 0.0415 * big_z
    b = 0.0557 * big_x - 0.2040 * lum + 1.0570 * big_z
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, None)


def latlong_directions(height: int, up_axis: int = 1) -> np.ndarray:
    """Unit direction of every texel of a latitude-longitude environment map, in stage axes.

    Returns ``(height, 2 * height, 3)``. Pixel centres rather than edges, so neither pole is
    sampled exactly and the horizon falls on a row boundary for an even ``height``.

    **The layout is the renderer's, not a choice.** RTX's dome light takes the texture's polar
    angle from the stage's **+Z** and its azimuth from +X toward -Y, which on a Y-up stage puts
    the texture's poles on the horizon and its equator through the zenith. Building the map in
    elevation and azimuth and hoping is what fills a frame entirely with ground.
    """
    if height < 8 or height % 2:
        raise ValueError("height must be an even number of rows, at least 8")
    width = 2 * height
    theta = (np.arange(height, dtype=np.float64) + 0.5) * (math.pi / height)
    phi = (np.arange(width, dtype=np.float64) + 0.5) * (2.0 * math.pi / width)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    return np.stack([np.sin(t) * np.cos(p), -np.sin(t) * np.sin(p), np.cos(t)], axis=-1)


@dataclass
class SkyConditions:
    """Everything the sky needs, resolved from a :class:`~weather_fx.core.state.WeatherState`.

    Holding it as one object is what stops the dome, the sun light, the moon light and any sensor
    model reading four slightly different skies: they are all built from this.
    """

    when: datetime
    sun: BodyPosition
    moon: MoonPosition
    turbidity: float
    ground_albedo: Tuple[float, float, float]
    cloud: Optional[CloudField] = None
    star_intensity: float = 1.0
    exposure_scale: float = 1.0
    #: Samples per ray in the cloud march. Carried here rather than read from the state at the
    #: march, so every consumer of one `SkyConditions` integrates the cloud identically.
    march_steps: int = 64
    #: "atmosphere" (Hillaire 2020, :mod:`weather_fx.core.atmosphere`) or "preetham" (the 1999
    #: analytic fit, kept for comparison and for anything calibrated against it).
    model: str = "atmosphere"
    #: Degrees below the horizon over which the dome's ground fades into the sky (atmosphere only).
    horizon_blend_deg: float = 4.0
    #: Colour of the sunlit and of the self-shadowed parts of a cloud, as multipliers.
    cloud_lit_colour: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    cloud_shadow_colour: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: Rows of the latitude-longitude grid the cloud is marched on before being upsampled into the
    #: environment map. The march is the whole cost of a cloudy bake and a cloud's edges are soft,
    #: so a quarter of the map's rows loses little and costs a sixteenth.
    cloud_rows: int = 256
    #: How far the wind has carried the cloud field, in its own Y-up frame, metres. The observer
    #: sits at the field's origin, so the march starts at minus this.
    cloud_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def daylight(self) -> float:
        """How much of full daylight survives at this sun elevation. 1 by day, 0 at night."""
        return civil_twilight_fraction(self.sun.elevation_deg)

    @property
    def moon_lux(self) -> float:
        return moon_illuminance_lux(self.moon)

    def describe(self) -> str:
        """One line for a UI or a log. Worth having: every number here is derived, so a scene
        that looks wrong is usually a scene whose clock or place is wrong."""
        return (
            f"{self.when:%Y-%m-%d %H:%M} UTC  "
            f"sun {self.sun.elevation_deg:+.1f} deg el / {self.sun.azimuth_deg:.0f} deg az  "
            f"moon {self.moon.elevation_deg:+.1f} deg, {self.moon.phase_name} "
            f"({self.moon.illuminated_fraction * 100:.0f} %)  "
            f"turbidity {self.turbidity:.1f}"
            + (
                f"  cloud {self.cloud.measured_cover * 100:.0f} % {self.cloud.profile.name}"
                if self.cloud is not None
                else "  no cloud field built"
            )
        )


def conditions_from_state(state: Any, *, build_cloud: bool = True) -> SkyConditions:
    """Resolve a :class:`WeatherState` into the one object every sky consumer reads.

    The cloud field is the expensive part -- a few hundred milliseconds to synthesise -- so
    ``build_cloud=False`` gives a caller that only wants the sun and the moon a cheap answer.
    """
    sky, clouds = state.sky, state.clouds
    try:
        day = datetime.strptime(sky.date_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(
            f"sky.date_utc must be YYYY-MM-DD, got {sky.date_utc!r}"
        ) from exc
    when = day + timedelta(hours=float(sky.hour_utc))

    field = cloud_field_from_state(state) if build_cloud else None

    return SkyConditions(
        when=when,
        sun=sun_position(sky.latitude_deg, sky.longitude_deg, when),
        moon=moon_position(sky.latitude_deg, sky.longitude_deg, when),
        turbidity=float(sky.turbidity),
        ground_albedo=tuple(sky.ground_albedo),
        cloud=field,
        star_intensity=float(sky.star_intensity),
        exposure_scale=float(sky.exposure_scale),
        march_steps=int(clouds.march_steps),
        model=str(getattr(sky, "model", "atmosphere")),
        horizon_blend_deg=float(getattr(sky, "horizon_blend_deg", 4.0)),
        cloud_lit_colour=tuple(getattr(clouds, "lit_color", (1.0, 1.0, 1.0))),
        cloud_shadow_colour=tuple(getattr(clouds, "shadow_color", (1.0, 1.0, 1.0))),
        cloud_rows=int(getattr(clouds, "dome_rows", 256)),
    )


def sky_luminance_cd_m2(
    directions: Any, body: BodyPosition, turbidity: float, scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perez luminance and chromaticity about one body. Returns ``(Y, x, y)``.

    Used twice: once for the sun to make daylight, once for the moon to make a moonlit night.
    They are the same scattering problem with a source four hundred thousand times fainter, so
    reusing the distribution is not a shortcut -- it is the reason a moonlit sky is the same blue
    as a daylit one, which is a real and frequently-missed property of night photography.
    """
    d = np.asarray(directions, dtype=np.float64)
    up = d[..., 1]
    cos_theta = np.clip(up, 0.0, 1.0)
    sun_dir = body.direction()
    cos_gamma = np.clip(np.sum(d * sun_dir, axis=-1), -1.0, 1.0)
    gamma = np.arccos(cos_gamma)

    zenith_angle = math.radians(min(max(90.0 - body.elevation_deg, 0.0), 92.0))

    def channel(name: str) -> np.ndarray:
        """The Perez ratio for one channel: its value here over its value at the zenith."""
        coefficients = _perez_coefficients(turbidity, name)
        here = perez(cos_theta, gamma, coefficients)
        at_zenith = perez(1.0, zenith_angle, coefficients)
        return here / max(float(at_zenith), 1e-9)

    zenith_luminance = max(_zenith_luminance_cd_m2(turbidity, zenith_angle), 0.0)
    x0, y0 = _zenith_chromaticity(turbidity, zenith_angle)
    # Each channel is distributed by its *own* coefficients and anchored on its own zenith value.
    luminance = channel("Y") * zenith_luminance * scale
    x = np.clip(x0 * channel("x"), 0.05, 0.75)
    y = np.clip(y0 * channel("y"), 0.05, 0.75)
    return luminance, x, y


def sky_radiance_rgb(directions: Any, conditions: SkyConditions) -> np.ndarray:
    """Linear RGB sky luminance, cd/m2, for a set of stage-space directions.

    Daylight, moonlight and starlight are **added**, never switched between, so the whole day is
    one continuous function of the clock. Below the horizon the ground is returned instead: a
    Lambertian terrain lit by whatever is in the sky, which is what stops the lower half of an
    environment map being a black hemisphere the dome then lights the scene with.
    """
    d = np.asarray(directions, dtype=np.float64)
    if conditions.model == "atmosphere":
        return _atmosphere_radiance_rgb(d, conditions)
    up = d[..., 1]

    total = np.zeros(d.shape, dtype=np.float64)

    daylight = conditions.daylight
    if daylight > 0.0:
        luminance, x, y = sky_luminance_cd_m2(d, conditions.sun, conditions.turbidity, daylight)
        total += _xyy_to_linear_rgb(luminance, x, y)

    if conditions.moon.is_up and conditions.moon_lux > 0.0:
        # The moon's sky, scaled so the horizontal illuminance it produces is the moon's own.
        # `pi` converts an average radiance over the hemisphere into an illuminance.
        luminance, x, y = sky_luminance_cd_m2(d, conditions.moon, conditions.turbidity, 1.0)
        mean = float(np.mean(luminance[up > 0.0])) if np.any(up > 0.0) else 1.0
        moon_sky = luminance * (conditions.moon_lux / max(math.pi * mean, 1e-9))
        total += _xyy_to_linear_rgb(moon_sky, x, y) * np.asarray(MOONLIGHT_TINT)

    floor = STARLIGHT_LUX * conditions.star_intensity / math.pi
    total += floor * np.asarray(MOONLIGHT_TINT)

    if conditions.cloud is not None:
        total = _composite_cloud(d, total, conditions)

    # Below the horizon: Lambertian ground lit by the sky above it.
    below = up < 0.0
    if np.any(below):
        above = up > 0.0
        illuminance = (
            float(np.mean(total[above].sum(axis=-1) / 3.0)) * math.pi if np.any(above) else 0.0
        )
        albedo = np.asarray(conditions.ground_albedo, dtype=np.float64)
        ground = albedo * illuminance / math.pi
        # Fade into the horizon rather than meeting it at a hard line: distant ground is seen
        # through enough air to take the sky's colour (Koschmieder, at the model's own turbidity).
        haze = np.clip(1.0 - np.exp(-6.0 * np.abs(up[below])), 0.0, 1.0)[..., None]
        # The colour the ground fades *into* is the mean of the band just above the horizon, not
        # the single nearest texel: with cloud in the sky that one texel is as likely to be a
        # sunlit cumulus as sky, and using it paints a bright ring all the way round the horizon.
        band = (up > 0.0) & (up < 0.03)
        horizon = (
            total[band].mean(axis=0)
            if np.any(band)
            else (total[above].mean(axis=0) if np.any(above) else np.zeros(3))
        )
        total[below] = ground * haze + horizon * (1.0 - haze)

    return np.clip(total * conditions.exposure_scale, 0.0, None)


def _relative_azimuth(directions: np.ndarray, body: BodyPosition) -> np.ndarray:
    toward = body.direction()
    return np.arctan2(directions[..., 2], directions[..., 0]) - math.atan2(toward[2], toward[0])


def _atmosphere_radiance_rgb(d: np.ndarray, conditions: SkyConditions) -> np.ndarray:
    """The sky from :mod:`weather_fx.core.atmosphere`: sun, moon and stars through one medium.

    The ground below the horizon comes out of the same march -- lit terrain seen through the air
    in front of it -- so there is no separate ground model and no seam to hide.
    """
    atm = atmosphere.atmosphere_for(conditions.turbidity, conditions.ground_albedo)
    elevation = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    total = np.zeros(d.shape, dtype=np.float64)

    # The dome's ground is everything beyond the stage, seen from two metres up -- a geometric
    # horizon five kilometres away, too close for any haze, so the model alone draws a hard line
    # there. Real terrain has relief and depth, so its far edge melts into the sky: fade the
    # ground rows into the sky just above the horizon in the same direction, over a band.
    blend = math.radians(max(conditions.horizon_blend_deg, 0.0))

    def sample(view: "atmosphere.SkyView", relative_azimuth: np.ndarray) -> np.ndarray:
        values = view.radiance(elevation, relative_azimuth)
        if blend <= 0.0:
            return values
        depth = view.horizon_elevation - elevation
        below = depth > 0.0
        if not np.any(below):
            return values
        x = np.clip(depth[below] / blend, 0.0, 1.0)
        weight = (x * x * (3.0 - 2.0 * x))[..., None]  # smoothstep: 0 at the horizon, 1 past the band
        sky = view.radiance(np.full(x.shape, view.horizon_elevation + 1e-3),
                            relative_azimuth[below])
        values = np.array(values)
        values[below] = sky * (1.0 - weight) + values[below] * weight
        return values

    sun = conditions.sun
    # Below about -18 degrees nothing of the sun reaches even the upper air.
    if sun.elevation_deg > -18.0:
        view = atmosphere.sky_view(atm, sun.elevation_deg)
        total += sample(view, _relative_azimuth(d, sun)) * atmosphere.SOLAR_ILLUMINANCE_LUX

    moon = conditions.moon
    if moon.is_up and conditions.moon_lux > 0.0:
        view = atmosphere.sky_view(atm, moon.elevation_deg)
        # `moon_lux` is the horizontal illuminance at the ground; the table wants it at the top of
        # the air, at normal incidence.
        mu = math.sin(math.radians(max(moon.elevation_deg, 1.0)))
        t_ground = float(atmosphere.transmittance_to_space(0.0, mu, atm)[1])
        source = conditions.moon_lux / (mu * max(t_ground, 1e-3))
        total += (sample(view, _relative_azimuth(d, moon)) * source
                  * np.asarray(MOONLIGHT_TINT))

    total += STARLIGHT_LUX * conditions.star_intensity / math.pi * np.asarray(MOONLIGHT_TINT)

    if conditions.cloud is not None:
        above = d[..., 1] > 0.0
        if np.any(above):
            total[above] = _composite_cloud(d[above], total[above], conditions)
    return np.clip(total * conditions.exposure_scale, 0.0, None)


def _composite_cloud(
    directions: np.ndarray, sky: np.ndarray, conditions: SkyConditions
) -> np.ndarray:
    """March the cloud field along every direction and put it in front of the sky.

    The same march the infrared band uses, so the two bands cannot put cloud in different places.
    """
    transmittance, added = _cloud_terms(directions, conditions)
    return sky * transmittance[..., None] + added


def _cloud_terms(directions: np.ndarray, conditions: SkyConditions) -> Tuple[np.ndarray, np.ndarray]:
    """What the cloud does along each direction: ``(transmittance, added luminance)``.

    Kept as two terms so a caller can march coarsely and upsample them without blurring the sky
    behind the cloud: ``out = transmittance * sky + added``.
    """
    field = conditions.cloud
    assert field is not None
    origin = np.zeros(directions.shape, dtype=np.float64)
    origin[..., 0] = -conditions.cloud_offset_m[0]
    origin[..., 1] = 2.0
    origin[..., 2] = -conditions.cloud_offset_m[2]

    lit_by = conditions.sun if conditions.sun.elevation_deg > 0.0 else conditions.moon
    sun_direction = lit_by.direction() if lit_by.elevation_deg > 0.0 else None
    result = field.march(
        origin, directions, sun_direction=sun_direction, steps=int(conditions.march_steps)
    )

    added = np.zeros(directions.shape, dtype=np.float64)
    if result.radiance is not None:
        # The incident illuminance on the cloud tops, spread over the hemisphere: what the
        # albedo the march returns is a fraction *of*.
        # The beam that lights the cloud is the beam that reached it, and at a low sun that beam
        # is orange. A cloud is white; everything people photograph at sunset is this tint.
        reddening = sun_colour(lit_by.elevation_deg, conditions.turbidity)
        if lit_by is conditions.sun:
            source = 1.6e9 * math.sin(math.radians(max(lit_by.elevation_deg, 0.0)))
            source *= conditions.daylight
            tint = reddening
        else:
            # The same scale as the sun's line above, which is 1.6e4 per lux of *horizontal*
            # illuminance -- and moon_lux is horizontal. (It was 1e4/pi: five times too dark.)
            source = 1.6e4 * conditions.moon_lux
            tint = np.asarray(MOONLIGHT_TINT) * reddening
        added = (result.radiance[..., None] * source / math.pi * tint * 1e-4
                 * _cloud_colour(result.radiance, result.transmittance, conditions))
    return result.transmittance, added


def _cloud_colour(radiance: np.ndarray, transmittance: np.ndarray, conditions: SkyConditions) -> np.ndarray:
    """Per-pixel colour of the cloud: the shadow colour where it is self-shadowed, the lit colour
    where the sun reaches it, blended by how lit the march found it.

    "How lit" is the scattered radiance per unit of cloud opacity, which runs from the march's
    shadow floor (a fully shadowed interior) to one (a sunlit face).
    """
    from weather_fx.core.clouds import MS_SHADOW_FLOOR

    lit = np.asarray(conditions.cloud_lit_colour, dtype=np.float64)
    shadow = np.asarray(conditions.cloud_shadow_colour, dtype=np.float64)
    if np.allclose(lit, shadow):
        return lit
    litness = radiance / np.maximum(1.0 - transmittance, 1e-3)
    w = np.clip((litness - MS_SHADOW_FLOOR) / (1.0 - MS_SHADOW_FLOOR), 0.0, 1.0)[..., None]
    return shadow * (1.0 - w) + lit * w


#: Renderer-side level the *brightest part of the sky* is exposed to, in the units an RTX dome
#: light's `intensity` multiplies its texture by. Measured by sweeping the intensity on a demo
#: stage: a clear midday sky renders correctly at an intensity near 0.025 -- which also sets the
#: sun, exposed by the same factor, and so the brightness of sunlit ground. The two constants
#: below are anchored so that intensity is unchanged: 0.025 times the reference.
DOME_HIGHLIGHT_TARGET = 217.5
#: The metered 99th-percentile luminance of a clear midday sky, cd/m2 -- the anchor the adaptation
#: is written about. Measured from the atmosphere model at a 65 degree sun, turbidity 2.1, with the
#: cone around the sun excluded (METER_EXCLUSION_DEG). The Preetham fit, metered with the aureole
#: included, put it at 20,000.
DAYLIGHT_REFERENCE_CD_M2 = 8_700.0
#: How completely the exposure adapts to the scene. **A tone decision, stated rather than hidden.**
#: The sky spans seven decades between noon and a moonless night and a display spans two, so
#: something has to compress it: 1.0 exposes every frame to the same brightness and a moonlit
#: night looks like an overcast afternoon, while 0.0 holds one exposure for the whole day and
#: every frame but one is black or white. At 0.83 a full moon lands about three and a half stops
#: below noon -- dark, legible, and in the right order.
EXPOSURE_ADAPTATION = 0.83


#: Half-angle of the cone around the sun and the moon that the exposure meter ignores, degrees.
#: The aureole inside it is the brightest part of any sky, and at night it is *all* the meter saw:
#: a full moon's glow set the exposure and left the rest of the sky and the landscape black --
#: a moonless night measured brighter than a moonlit one.
METER_EXCLUSION_DEG = 25.0
#: Rec. 709 luminance weights, for linear sRGB.
LUMA = np.array([0.2126, 0.7152, 0.0722])


#: The direct sun's correlated colour temperature at a high sun, kelvin: what the viewport's
#: colours are balanced to, so a midday sun renders white.
SUN_CCT_HIGH_K = 5_800.0
#: The sun's colour temperature at the horizon, kelvin.
SUN_CCT_HORIZON_K = 2_000.0
#: Elevation scale of the approach from one to the other, degrees, at turbidity 2.8.
SUN_CCT_SCALE_DEG = 12.0


def _planck_srgb(cct_k: float) -> np.ndarray:
    """Linear sRGB of a blackbody at ``cct_k``, unit luminance (Kim et al. 2002 Planckian locus)."""
    t = float(np.clip(cct_k, 1_667.0, 25_000.0))
    if t <= 4_000.0:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2_222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4_000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return _xyy_to_linear_rgb(1.0, x, y)


def sun_colour(elevation_deg: float, turbidity: float = 2.8) -> np.ndarray:
    """Colour of direct sunlight as a camera balanced for a high sun records it. Unit luminance.

    From the sun's measured colour temperature, which climbs from about 2,000 K on the horizon
    through 3,300 K at 5 degrees and 4,100 K at 10 to 5,100 K at 20 and 5,800 K overhead; haze
    slows the climb. :func:`weather_fx.core.meteorology.beam_tint` computes the reddening from
    first principles at the three primaries' single wavelengths, which puts a 10 degree sun near
    3,000 K and turned every morning cloud orange; the visible rendering uses this instead, and
    the thermal model keeps ``beam_tint``, which it was calibrated with.
    """
    scale = SUN_CCT_SCALE_DEG * math.sqrt(max(float(turbidity), 1.0) / 2.8)
    el = max(float(elevation_deg), 0.0)
    cct = SUN_CCT_HIGH_K - (SUN_CCT_HIGH_K - SUN_CCT_HORIZON_K) * math.exp(-el / scale)
    rgb = _planck_srgb(cct) / _planck_srgb(SUN_CCT_HIGH_K)
    return rgb / float(rgb @ LUMA)


#: How much the sky's own light counts toward the white point, against the direct beam. Eyes and
#: cameras adapt to the brightest white in view -- a sunlit cloud, sunlit ground -- far more than
#: to the blue fill from the sky; weighting them equally cancels the warm sun against the blue sky
#: and leaves a morning's sunlit clouds orange, which nobody sees.
WHITE_POINT_SKY_WEIGHT = 0.25


def scene_illuminant(image: np.ndarray, conditions: SkyConditions,
                     sky_weight: float = 1.0) -> np.ndarray:
    """RGB illuminance on a horizontal surface: the sky in ``image`` plus the sun and moon beams.

    With ``sky_weight`` 1 it is the light the scene is lit by, which is what the exposure meters.
    With WHITE_POINT_SKY_WEIGHT it is the white point a camera's balance settles on. The beams use
    the same formulas the viewport's lights do.
    """
    from weather_fx.core.meteorology import clear_sky_irradiance

    directions = latlong_directions(image.shape[0])
    up = np.clip(directions[..., 1], 0.0, None)
    rows = image.shape[0]
    # Each texel's solid angle on a lat-long map: (pi/rows) * (2 pi / 2rows) * sin(polar angle).
    polar = (np.arange(rows) + 0.5) * (math.pi / rows)
    solid = (math.pi / rows) ** 2 * np.sin(polar)[:, None]
    sky = (image * (up * solid)[..., None]).sum(axis=(0, 1))

    total = sky.astype(np.float64) * float(sky_weight)
    sun = conditions.sun
    if sun.elevation_deg > 0.0:
        beam = (float(clear_sky_irradiance(sun.elevation_deg, conditions.turbidity)[0])
                * LUMINOUS_EFFICACY_DAYLIGHT * conditions.daylight
                * math.sin(math.radians(sun.elevation_deg)))
        total = total + sun_colour(sun.elevation_deg, conditions.turbidity) * beam
    moon = conditions.moon
    if moon.is_up and conditions.moon_lux > 0.0:
        tint = sun_colour(moon.elevation_deg, conditions.turbidity) * MOONLIGHT_TINT
        total = total + tint * conditions.moon_lux
    return total


def white_balance_gains(illuminant: Any, strength: float) -> np.ndarray:
    """Per-channel gains that pull ``illuminant`` toward neutral, luminance-preserving.

    ``strength`` 1 is a camera's full auto white balance (every light source rendered white);
    0 leaves the light as the physics made it. Real cameras and eyes sit in between -- a sunset
    still reads warm -- which is why the default is below 1.
    """
    ill = np.maximum(np.asarray(illuminant, dtype=np.float64), 1e-12)
    y = float(ill @ LUMA)
    if y <= 0.0 or strength <= 0.0:
        return np.ones(3)
    gains = (y / ill) ** float(np.clip(strength, 0.0, 1.0))
    # Keep the balanced illuminant's luminance where it was: white balance changes colour, not level.
    gains /= float((gains * ill) @ LUMA) / y
    return gains


#: Illuminance on the ground under a clear midday sky (65 degree sun, turbidity 2.1), lux, and the
#: dome intensity measured to render it correctly on the demo stage. The incident meter below is
#: anchored on this pair, so midday is exactly what it was.
INCIDENT_REFERENCE_LUX = 100_000.0
EXPOSURE_AT_REFERENCE = 0.025
#: How far the metered sky highlights may go above DOME_HIGHLIGHT_TARGET before they cap the
#: exposure: 8 is three stops. Exposing for a sunset's dim ground would otherwise blow out the sky.
HIGHLIGHT_HEADROOM = 8.0


def dome_exposure(image: np.ndarray, conditions: Optional[SkyConditions] = None) -> float:
    """The dome and light intensity that puts the scene where a camera would put it.

    With ``conditions`` it is an **incident-light meter**: it exposes for the light falling on the
    ground (sky plus sun or moon, :func:`scene_illuminant`), adapted by EXPOSURE_ADAPTATION, and
    capped so the sky's highlights -- metered away from the sun and moon -- stay within
    HIGHLIGHT_HEADROOM of the target. Metering the scene's light rather than the sky's brightest
    patch is what keeps a moonlit landscape readable: the sky meter exposed for the moon's glow and
    left everything else black.

    Without ``conditions`` it falls back to the sky-highlight meter alone.
    """
    highlight = _highlight_exposure(image, conditions)
    if conditions is None:
        return highlight
    lux = float(scene_illuminant(image, conditions) @ LUMA)
    incident = EXPOSURE_AT_REFERENCE * (
        INCIDENT_REFERENCE_LUX / max(lux, 1e-9)) ** EXPOSURE_ADAPTATION
    return float(np.clip(min(incident, HIGHLIGHT_HEADROOM * highlight), 1e-6, 1e9))


def _highlight_exposure(image: np.ndarray, conditions: Optional[SkyConditions] = None) -> float:
    """The dome-light intensity that puts ``image`` (cd/m2) where a camera would put it.

    **Meter on the highlights, not on the middle.** A median is the wrong statistic for a sky and
    it fails in both directions. At sunset the dim anti-solar half drags the median down, the
    exposure up, and the whole solar side clips to white -- measured, a 4 degree sun put the 99th
    percentile 2.4x above where noon puts it. At civil twilight the median of the upper hemisphere
    is *zero* to float precision and the exposure ran to its clamp, which is a white frame at the
    darkest moment of the day. The 99th percentile is stable through both, and it is also what a
    photographer meters.

    The percentile is taken over the **sky only**: the ground half of a lat-long map is far darker
    and including it drags the reference down and the intensity up.
    """
    directions = latlong_directions(image.shape[0])
    above = directions[..., 1] > 0.0
    if conditions is not None:
        cos_limit = math.cos(math.radians(METER_EXCLUSION_DEG))
        for body in (conditions.sun, conditions.moon):
            if body.elevation_deg > -METER_EXCLUSION_DEG:
                toward = np.asarray(body.direction(), dtype=np.float64)
                above &= (directions @ toward) < cos_limit
    luminance = image[above].mean(axis=-1) if np.any(above) else np.asarray([1.0])
    bright = float(np.percentile(luminance, 99.0))
    adapted = DAYLIGHT_REFERENCE_CD_M2 * (
        max(bright, 1e-9) / DAYLIGHT_REFERENCE_CD_M2
    ) ** EXPOSURE_ADAPTATION
    return float(np.clip(DOME_HIGHLIGHT_TARGET / max(adapted, 1e-9), 1e-6, 1e9))


def environment_map(conditions: SkyConditions, height: int = 1024) -> np.ndarray:
    """The latitude-longitude environment map: ``(height, 2 * height, 3)`` float32, cd/m2.

    Authored in *direction* space -- every texel becomes a stage-space unit vector and the sky is
    evaluated there -- so the texture's layout is the renderer's business and the physics never
    has to know it.

    Float32 and never float16. A half-float sky is the habit that ends up in a temperature path,
    and the encode discipline is cheaper to keep than to retrofit.
    """
    directions = latlong_directions(height)
    if conditions.cloud is None or conditions.cloud_rows >= height:
        return sky_radiance_rgb(directions, conditions).astype(np.float32)

    # The clear sky at full resolution; the cloud marched on a coarser grid and upsampled. The
    # march is the entire cost of a cloudy bake, and it scales with the texel count.
    clear = sky_radiance_rgb(directions, _without_cloud(conditions))
    rows = max(8, int(conditions.cloud_rows) // 2 * 2)
    coarse_dirs = latlong_directions(rows)
    up = coarse_dirs[..., 1] > 0.0
    transmit = np.ones(coarse_dirs.shape[:2])
    added = np.zeros(coarse_dirs.shape)
    transmit[up], added[up] = _cloud_terms(coarse_dirs[up], conditions)
    t_full = _upsample_latlong(transmit[..., None], height)[..., 0]
    a_full = _upsample_latlong(added, height) * conditions.exposure_scale
    above = directions[..., 1] > 0.0
    out = clear.copy()
    out[above] = clear[above] * t_full[above][..., None] + a_full[above]
    return np.clip(out, 0.0, None).astype(np.float32)


def _without_cloud(conditions: SkyConditions) -> SkyConditions:
    from dataclasses import replace

    return replace(conditions, cloud=None)


def _upsample_latlong(image: np.ndarray, height: int) -> np.ndarray:
    """Bilinear upsample of a lat-long map to ``height`` rows, wrapping in longitude."""
    rows, cols = image.shape[:2]
    width = 2 * height
    y = np.clip((np.arange(height) + 0.5) * rows / height - 0.5, 0.0, rows - 1.0)
    x = (np.arange(width) + 0.5) * cols / width - 0.5
    y0 = np.minimum(np.floor(y).astype(np.int64), rows - 2)
    fy = (y - y0)[:, None, None]
    x0 = np.floor(x).astype(np.int64)
    fx = (x - x0)[None, :, None]
    x0w, x1w = x0 % cols, (x0 + 1) % cols
    top = image[y0][:, x0w] * (1 - fx) + image[y0][:, x1w] * fx
    bottom = image[y0 + 1][:, x0w] * (1 - fx) + image[y0 + 1][:, x1w] * fx
    return top * (1 - fy) + bottom * fy
