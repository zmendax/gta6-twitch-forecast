# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/zmendax/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/zmendax/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day: five quantiles per hour from the curves 38-42 comparable launches drew after their own first hour, weighted by how big and how late-ramping this launch is.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Every `<id>.in` in the input directory becomes `<id>.out`: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, …]}`, hours 2 to 24, quantiles non-decreasing, viewers above 0.

One file, Python 3.13 and its standard library, no network, no dependencies, and no clock or randomness: two runs on one case write the same bytes.

## How the forecast is made

Each comparable launch is reduced to two numbers the case already holds by the checkpoint — the size of its hour-1 audience, and how much of its ramp had already happened by hour 1. Every comparable's later hours are taken as a ratio to its own hour 1, so the curves line up whatever the game's scale.

For each hour of the release day the forecast is the weighted median of those ratios: a Gaussian kernel on log audience size decides whose curves count most (a mega-launch does not follow a small one's curve), with a weaker term on the size of the pre-launch ramp, since a launch whose audience already peaked at hour 1 declines afterwards while one that had not yet begun climbs. The spread comes from the same weighted sample — its interquantile range, floored so a lucky agreement among comparables never reads as certainty — and the five quantiles are that width around the centre on the log scale.

`dev/` holds the search that chose those settings: `dev/cv3.mjs` and `dev/cv2.mjs` hold out each comparable launch in turn, forecast it from the others, and score the result with this contest's own pinball loss on the log scale; `dev/diag.mjs` breaks the loss down hour by hour; `dev/explore.mjs` prints the comparable curves and their spread. Each takes the kit's `input` directory as an argument.

---

<sub>Entered on [make.fun](https://make.fun/?ref=5pse4ke5&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
