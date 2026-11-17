"""Forecast Grand Theft Auto VI's launch-day Twitch viewers, hour by hour.

Given one case (`/input/<id>.in`): the same category's viewers up to the end of release day's
first hour, plus 38-42 comparable games' hourly curves from 14 days before their release day to
3 days after, the program writes one forecast per case: for hours 2..24 of the release day, the
5%, 25%, 50%, 75% and 95% quantiles of that hour's viewers.

How the forecast is made
------------------------
Every comparable is summarised by two numbers that the case does hold before the hour runs:

  scale  = ln(hour 1's viewers)                how big the launch's live audience is
  jump   = ln(hour 1's viewers / pre-launch)   how much of the ramp already happened by hour 1

The curve each comparable drew after its own hour 1 is its ratio to its own hour 1, hour by hour.
A launch's later hours are then the weighted median of those ratios: comparables whose scale is
close to this launch's count most, since a mega-launch and a small one follow different curves,
and a weak term keeps launches whose ramp had reached a similar stage close too. The spread comes
from the same weighted sample: its interquantile range sets a Gaussian width on the log scale, so
the bands widen where launches disagree and narrow where they agree, and a floor keeps an hour
whose comparables happen to agree from claiming certainty it does not have.

Everything is deterministic: no clock, no randomness, no network, and the comparables are read in
the order the case lists them, so two runs on one case write the same bytes.

Run it on your machine:  MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level; the spread is modelled as Gaussian in log viewers.
Z = [-1.6449, -0.6745, 0.0, 0.6745, 1.6449]
HOURS = list(range(2, 25))

# Similarity kernel, in natural-log units. A launch twice as big as another is 0.69 apart.
SIGMA_SCALE = 0.6
SIGMA_JUMP = 3.0
# Launches of this audience size run a little above the pooled median ratio; a small shift down.
BIAS = -0.05
# Multiplier on the sampled interquantile spread, and its floor in log units.
INFLATE = 1.0
SIGMA_FLOOR = 0.2
# Fewer usable comparables than this and the hour falls back to the pooled ratio alone.
MIN_ROWS = 4
MAX_VIEWERS = 1_000_000_000


def hour_of(iso: str, day_start: str) -> int:
    """The whole hour between `iso` and the case's day_start.

    Both are `YYYY-MM-DDTHH:MM:SS[.fff]Z`, so the arithmetic is done on the digits alone: the
    cases carry a few hundred points per comparable and strptime is far too slow for them.
    """

    def seconds(stamp: str) -> int:
        date, _, clock = stamp.partition("T")
        year, month, day = (int(part) for part in date.split("-"))
        hour, minute, rest = clock.split(":")
        # Days from the civil calendar, Howard Hinnant's algorithm: exact for every date, and fast.
        year_shifted = year - (1 if month <= 2 else 0)
        era = year_shifted // 400
        year_of_era = year_shifted - era * 400
        day_of_year = (153 * (month + (-3 if month > 2 else 9)) + 2) // 5 + day - 1
        day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
        days = era * 146097 + day_of_era
        return days * 86400 + int(hour) * 3600 + int(minute) * 60 + int(rest[:2])

    return round((seconds(iso) - seconds(day_start)) / 3600.0)


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def summarise(comparable: dict) -> dict | None:
    """One comparable's hourly curve and the two numbers the kernel uses."""
    day_start = comparable.get("day_start")
    hourly = comparable.get("hourly") or []
    if not day_start or not hourly:
        return None
    by_hour: dict[int, float] = {}
    for point in hourly:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            by_hour[hour_of(point["from"], day_start)] = float(viewers)
        except (KeyError, ValueError):
            continue
    first = by_hour.get(1)
    if not first or first <= 0:
        return None
    before = [by_hour[h] for h in range(-6, 1) if h in by_hour and by_hour[h] > 0]
    pre = median(before) or first
    return {
        "name": comparable.get("name") or "",
        "at": by_hour,
        "p1": first,
        "scale": math.log(first),
        "jump": math.log(first / max(1.0, pre)),
    }


def weighted_quantile(rows: list[tuple[float, float]], p: float) -> float:
    """The p-quantile of `rows` as (value, weight) pairs, weights read as a step CDF."""
    ordered = sorted(rows)
    total = math.fsum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = p * total
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= target:
            return value
    return ordered[-1][0]


def forecast_case(case: dict) -> dict:
    known = next((row for row in case.get("known") or [] if row.get("hour") == 1), None)
    if known is None:
        raise ValueError("the case has no hour-1 print")
    p1 = max(1.0, float(known.get("viewers") or 1))
    target_scale = math.log(p1)

    # The launch's own pre-release baseline, so its ramp stage can be read the same way.
    own_history = [row["viewers"] for row in case.get("history_hourly") or [] if row.get("viewers")]
    own_pre = median(own_history[-6:]) if own_history else None
    target_jump = math.log(p1 / max(1.0, own_pre)) if own_pre else 0.0

    comparables = []
    for raw in case.get("comparables") or []:
        made = summarise(raw)
        if made is not None:
            comparables.append(made)

    hours = []
    for hour in HOURS:
        rows = []
        for comp in comparables:
            viewers = comp["at"].get(hour)
            if viewers is None or viewers <= 0:
                continue
            distance = ((comp["scale"] - target_scale) / SIGMA_SCALE) ** 2 + (
                (comp["jump"] - target_jump) / SIGMA_JUMP
            ) ** 2
            rows.append((math.log(viewers) - math.log(comp["p1"]), math.exp(-distance / 2.0)))

        if len(rows) < MIN_ROWS:
            # No usable comparables: hold hour 1 with the same widening the house baseline uses.
            centre, sigma = 0.0, 0.15 * math.sqrt(hour - 1)
        else:
            centre = weighted_quantile(rows, 0.5) + BIAS
            spread = (weighted_quantile(rows, 0.75) - weighted_quantile(rows, 0.25)) / 1.349
            sigma = max(SIGMA_FLOOR, spread) * INFLATE

        quantiles = []
        for z in Z:
            q = p1 * math.exp(centre + z * sigma)
            quantiles.append(min(MAX_VIEWERS, max(0.01, q)))
        quantiles.sort()
        hours.append({"hour": hour, "q": [round(q, 2) for q in quantiles]})

    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    sys.exit(main())
