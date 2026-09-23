"""Random weather that is actually weather.

Drawing each parameter independently produces states that cannot happen: fog at 30 m visibility
under a clear sky with no humidity, heavy rain with four oktas of fair-weather cumulus, a 1500 m
cloud base with a two-degree dew-point spread. A model trained on that learns the noise, and a
sensor study run on it measures nothing.

So the randomiser draws a **regime** first -- the kind of day -- and then draws the parameters
*within* it, with the couplings that make the day hang together:

* cloud base follows the temperature and dew point, because condensation happens at one height;
* rain implies low cloud, a deep one, reduced visibility and some wind, because that is what is
  producing the rain;
* fog implies a tiny dew-point spread, almost no wind, and no convective cloud above it;
* snow implies sub-zero air, which rules out the rain regime and caps the visibility;
* turbidity rises with haze and falls after rain, because the rain washes the aerosol out.

Everything is drawn from one seeded :class:`numpy.random.Generator`, so a scene is reproducible
from an integer, and the whole thing is pure Python plus numpy so a dataset generator can call it
in a loop without a renderer attached.

The regime weights and the parameter ranges are **ESTIMATED**: they are a plausible spread of
conditions for a mid-latitude site, not a climatology. Anything that needs a real distribution
should pass its own ``weights``.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from weather_fx.core.state import WeatherState

__all__ = ["REGIMES", "RegimeWeights", "random_state", "random_overrides", "describe_regime"]

#: The kinds of day this can produce, with their default relative likelihoods. Ordered from
#: clearest to worst, which is also roughly how often each occurs.
REGIMES: Dict[str, float] = {
    "clear": 0.18,
    "fair_cumulus": 0.22,
    "broken_cumulus": 0.16,
    "overcast": 0.14,
    "haze": 0.08,
    "rain": 0.10,
    "fog": 0.05,
    "snow": 0.04,
    "storm": 0.03,
}

RegimeWeights = Mapping[str, float]


def describe_regime(name: str) -> str:
    """One line a UI or a dataset manifest can carry."""
    return {
        "clear": "clear sky, good visibility",
        "fair_cumulus": "scattered fair-weather cumulus",
        "broken_cumulus": "broken cumulus, larger and deeper",
        "overcast": "overcast stratus or stratocumulus",
        "haze": "clear of cloud but hazy, high turbidity",
        "rain": "rain from a deep low cloud deck",
        "fog": "fog or very low stratus, tiny dew-point spread",
        "snow": "snow from a low deck, sub-zero air",
        "storm": "heavy rain, strong gusty wind, towering cloud",
    }.get(name, name)


def _pick(rng: np.random.Generator, weights: RegimeWeights) -> str:
    names = list(weights)
    probabilities = np.asarray([max(float(weights[n]), 0.0) for n in names], dtype=np.float64)
    total = probabilities.sum()
    if total <= 0.0:
        raise ValueError("regime weights must not all be zero")
    return str(rng.choice(names, p=probabilities / total))


def _random_date(rng: np.random.Generator, year: Optional[int]) -> str:
    chosen = int(year) if year is not None else int(rng.integers(2020, 2031))
    day = int(rng.integers(1, 367 if calendar.isleap(chosen) else 366))
    return (date(chosen, 1, 1) + timedelta(days=day - 1)).isoformat()


def _seasonal_temperature(rng: np.random.Generator, day_of_year: int, latitude: float) -> float:
    """A plausible surface temperature, degrees C, for the date and latitude.

    A sinusoid about a latitude-dependent annual mean with a latitude-dependent amplitude, plus a
    few degrees of day-to-day scatter. Crude, and deliberately so -- its job is to stop the
    randomiser producing snow in July at the equator, not to be a climate model. The hemisphere
    flip is the part that is easy to forget and immediately obvious when wrong.
    """
    phase = 2.0 * np.pi * (day_of_year - 15.0) / 365.25
    if latitude < 0.0:
        phase += np.pi
    warmth = np.cos(np.radians(min(abs(latitude), 89.0)))
    annual_mean = -12.0 + 40.0 * warmth
    amplitude = 2.0 + 16.0 * (1.0 - warmth)
    return float(annual_mean + amplitude * np.cos(phase) + rng.normal(0.0, 3.0))


@dataclass(frozen=True)
class _Draw:
    """The per-regime draw, kept as one object so the couplings are visible in one place."""

    overrides: Dict[str, Dict[str, Any]]
    regime: str


def random_overrides(
    seed: Optional[int] = None,
    *,
    weights: Optional[RegimeWeights] = None,
    regime: Optional[str] = None,
    latitude_deg: Optional[float] = None,
    longitude_deg: Optional[float] = None,
    year: Optional[int] = None,
    night_fraction: float = 0.35,
) -> Dict[str, Dict[str, Any]]:
    """A coherent random weather, as a dict of ``{section: {parameter: value}}``.

    Returned as *overrides* rather than as a whole state so a caller can apply them to whatever
    it already has -- which is what lets a runtime randomiser change the weather without
    disturbing the follow prim, the time source or the seed the rest of the scene is using.

    ``night_fraction`` is the probability the clock lands at night. The default over-samples the
    night deliberately: it is under a third of a real day at most latitudes but it is where a
    visible camera struggles and an infrared one does not, so a dataset wants more of it than
    chance would give.
    """
    rng = np.random.default_rng(seed)
    chosen = regime if regime is not None else _pick(rng, weights or REGIMES)
    if chosen not in REGIMES and regime is not None and weights is None:
        raise ValueError(f"unknown regime {chosen!r}; known: {sorted(REGIMES)}")

    latitude = float(latitude_deg) if latitude_deg is not None else float(rng.uniform(-60.0, 65.0))
    longitude = (
        float(longitude_deg) if longitude_deg is not None else float(rng.uniform(-180.0, 180.0))
    )
    date_utc = _random_date(rng, year)
    day_of_year = date.fromisoformat(date_utc).timetuple().tm_yday

    # The clock is drawn in *local solar* time and converted, so "night" means night where the
    # scene is rather than night in Greenwich -- a randomiser that forgets this puts the sun
    # overhead in half the scenes it labelled as dark.
    if rng.random() < night_fraction:
        local_hour = float(rng.uniform(19.5, 28.5)) % 24.0
    else:
        local_hour = float(rng.uniform(6.0, 19.0))
    hour_utc = (local_hour - longitude / 15.0) % 24.0

    temperature = _seasonal_temperature(rng, day_of_year, latitude)
    overrides: Dict[str, Dict[str, Any]] = {
        "sky": {
            "enabled": True,
            "latitude_deg": latitude,
            "longitude_deg": longitude,
            "date_utc": date_utc,
            "hour_utc": hour_utc,
        },
        "clouds": {"seed": int(rng.integers(0, 1_000_000))},
        "fog": {},
        "rain": {"enabled": False},
        "snow": {"enabled": False},
        "wind": {},
    }

    # --- the couplings, one regime at a time ------------------------------------------------
    if chosen == "clear":
        spread = float(rng.uniform(8.0, 20.0))
        overrides["clouds"].update(enabled=False, cover=0.0)
        overrides["sky"]["turbidity"] = float(rng.uniform(1.9, 3.0))
        overrides["fog"].update(enabled=False)
        overrides["wind"].update(speed_mps=float(rng.uniform(0.0, 6.0)))

    elif chosen in ("fair_cumulus", "broken_cumulus"):
        broken = chosen == "broken_cumulus"
        spread = float(rng.uniform(6.0, 14.0)) if not broken else float(rng.uniform(4.0, 9.0))
        overrides["clouds"].update(
            enabled=True,
            genus="congestus" if broken and rng.random() < 0.4 else "cumulus",
            cover=float(rng.uniform(0.35, 0.65) if broken else rng.uniform(0.08, 0.35)),
            base_m=0.0,
            temperature_c=temperature,
            dewpoint_c=temperature - spread,
        )
        overrides["sky"]["turbidity"] = float(rng.uniform(2.2, 4.2))
        overrides["fog"].update(enabled=False)
        overrides["wind"].update(
            speed_mps=float(rng.uniform(1.0, 9.0)), gust_strength=float(rng.uniform(0.0, 0.4))
        )

    elif chosen == "overcast":
        spread = float(rng.uniform(1.5, 5.0))
        overrides["clouds"].update(
            enabled=True,
            genus="stratus" if rng.random() < 0.45 else "stratocumulus",
            cover=float(rng.uniform(0.88, 1.0)),
            base_m=0.0,
            temperature_c=temperature,
            dewpoint_c=temperature - spread,
        )
        overrides["sky"]["turbidity"] = float(rng.uniform(3.0, 5.5))
        overrides["fog"].update(enabled=False)
        overrides["wind"].update(speed_mps=float(rng.uniform(2.0, 11.0)))

    elif chosen == "haze":
        spread = float(rng.uniform(5.0, 12.0))
        visibility = float(np.exp(rng.uniform(np.log(2500.0), np.log(9000.0))))
        overrides["clouds"].update(enabled=rng.random() < 0.4, cover=float(rng.uniform(0.0, 0.2)))
        # Haze *is* high turbidity; the two must move together or the sky looks clear while the
        # fog volume says otherwise.
        overrides["sky"]["turbidity"] = float(rng.uniform(5.0, 9.5))
        overrides["fog"].update(enabled=True, visibility_m=visibility)
        overrides["wind"].update(speed_mps=float(rng.uniform(0.0, 4.0)))

    elif chosen in ("rain", "storm"):
        storm = chosen == "storm"
        spread = float(rng.uniform(0.5, 3.0))
        rate = float(np.exp(rng.uniform(np.log(8.0), np.log(60.0)))) if storm else float(
            np.exp(rng.uniform(np.log(0.5), np.log(12.0)))
        )
        overrides["clouds"].update(
            enabled=True,
            genus="congestus" if storm else "stratocumulus",
            cover=float(rng.uniform(0.9, 1.0)),
            base_m=0.0,
            temperature_c=max(temperature, 1.0),
            dewpoint_c=max(temperature, 1.0) - spread,
        )
        overrides["rain"].update(enabled=True, rate_mm_h=rate)
        # Rain scavenges aerosol, so the air between the drops is *clearer* than a hazy dry day
        # even though the visibility through the rain is worse. Both are true at once and a
        # randomiser that only lowers visibility gets the colour of the sky wrong.
        overrides["sky"]["turbidity"] = float(rng.uniform(2.0, 3.4))
        overrides["fog"].update(
            enabled=True, visibility_m=float(np.clip(12000.0 / max(rate, 0.5) ** 0.6, 300.0, 9000.0))
        )
        overrides["wind"].update(
            speed_mps=float(rng.uniform(8.0, 24.0)) if storm else float(rng.uniform(1.0, 10.0)),
            gust_strength=float(rng.uniform(0.3, 0.8)) if storm else float(rng.uniform(0.0, 0.3)),
        )

    elif chosen == "fog":
        # Fog is a saturated surface layer: the spread is what *makes* it, so it is drawn first
        # and everything else follows. Convective cloud above fog is not a thing -- there is no
        # surface heating to drive it -- so the cloud stays off.
        spread = float(rng.uniform(0.0, 0.6))
        overrides["clouds"].update(enabled=False, cover=0.0)
        overrides["sky"]["turbidity"] = float(rng.uniform(3.5, 7.0))
        overrides["fog"].update(
            enabled=True,
            visibility_m=float(np.exp(rng.uniform(np.log(30.0), np.log(900.0)))),
            height_fog=True,
            base_height_m=0.0,
        )
        overrides["wind"].update(speed_mps=float(rng.uniform(0.0, 2.0)), gust_strength=0.0)

    elif chosen == "snow":
        # Snow needs sub-zero air, so the seasonal draw is overridden rather than hoped for.
        temperature = float(rng.uniform(-14.0, 0.5))
        spread = float(rng.uniform(0.5, 3.0))
        overrides["clouds"].update(
            enabled=True,
            genus="stratocumulus",
            cover=float(rng.uniform(0.85, 1.0)),
            base_m=0.0,
            temperature_c=temperature,
            dewpoint_c=temperature - spread,
        )
        overrides["snow"].update(
            enabled=True, number_density_m3=float(np.exp(rng.uniform(np.log(0.5), np.log(60.0))))
        )
        overrides["sky"]["turbidity"] = float(rng.uniform(2.2, 4.0))
        overrides["fog"].update(
            enabled=True, visibility_m=float(np.exp(rng.uniform(np.log(200.0), np.log(6000.0))))
        )
        overrides["wind"].update(speed_mps=float(rng.uniform(0.0, 12.0)))

    else:  # pragma: no cover - `_pick` only returns keys of `weights`
        raise ValueError(f"unhandled regime {chosen!r}")

    overrides["wind"].setdefault("direction_deg", float(rng.uniform(-180.0, 180.0)))
    return overrides


def random_state(
    seed: Optional[int] = None,
    *,
    base: Optional[WeatherState] = None,
    **kwargs: Any,
) -> WeatherState:
    """A whole :class:`WeatherState` for a random day, applied on top of ``base``.

    The ``general`` section is never touched, so the follow prim, the time source and the seed a
    caller set stay set -- randomising the weather must not quietly take a deterministic data
    run off its manual clock.
    """
    state = (base or WeatherState()).copy()
    for section, values in random_overrides(seed, **kwargs).items():
        if values:
            state = state.with_updates(section, **values)
    return state


def random_sequence(
    count: int, seed: Optional[int] = None, **kwargs: Any
) -> Sequence[Dict[str, Dict[str, Any]]]:
    """``count`` independent random weathers from one seed, for a dataset run.

    Drawn from a single seeded generator's child streams rather than from ``seed + i``: nearby
    integer seeds give correlated first draws in most generators, which is how a "random" dataset
    ends up with its first frame the same kind of day every time.
    """
    root = np.random.SeedSequence(seed)
    return [random_overrides(int(child.generate_state(1)[0]), **kwargs) for child in root.spawn(count)]
