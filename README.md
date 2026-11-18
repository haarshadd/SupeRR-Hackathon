# SupeRR Selector 2.0 — Round 1: Pressure Performance Rankings

This repository contains the Round 1 analysis pipeline for identifying the **Top 10 batters and Top 10 bowlers who perform best under pressure** in T20 cricket.

The project estimates the importance of each delivery from the match state, identifies high-pressure deliveries, credits the batter and bowler for performance relative to expectation, and ranks eligible players using a confidence-aware score.

> **Scope:** This is a pressure-performance ranking system for Round 1. It is not an auction shortlist, retention recommendation, or player-availability tool.

## 1. What the pipeline does

1. Loads ball-by-ball, match, competition, and competition-to-team/player mapping data.
2. Removes Super Overs and matches that cannot be modelled cleanly, including tied, abandoned/no-result, and revised-conditions/DLS matches.
3. Reconstructs the match state before each delivery: score, wickets lost, legal balls used, balls remaining, and chase target.
4. Estimates the batting team's win probability at each match state using separate first- and second-innings models.
5. Calculates delivery leverage: the expected size of the win-probability change that the next legal delivery could cause.
6. Marks pressure deliveries using the configured first-innings collapse rule and chase leverage threshold.
7. Calculates player-attributed performance above expectation, applies opposition-strength adjustment, and generates separate batter and bowler rankings.
8. Saves the Top 10 CSVs, full eligible-player ratings, model diagnostics, and validation outputs under `outputs/pressure/`.

## 2. Data required

Place these four CSV files in a directory named `data/` at the repository root:

```text
data/
├── ball_by_ball.csv
├── matches.csv
├── competitions.csv
└── competition_team_player_mapping.csv
```

The files must use the column names and identifiers expected by `rr_scout/data.py`. The data pack is **not included** in this code bundle. The mapping CSV is required as part of the input pack even though player names used in the rankings are derived from the ball-by-ball striker and bowler fields.

## 3. Installation and execution

Use Python 3.10 or newer. From the repository root, create and activate a virtual environment if desired, then run:

```bash
python -m pip install -r requirements.txt
python run_pressure.py
```

The main pipeline writes its results to `outputs/pressure/`. It also creates `ball_pressure.csv.gz`, the delivery-level pressure dataset used by the validation suite.

To run the validation suite after the main pipeline has completed:

```bash
python -m rr_scout.validation
```

Run commands from the repository root so that Python can import the `rr_scout` package and the configured relative paths resolve correctly.

## 4. Pressure methodology

### 4.1 Match-state and win-probability model

The model estimates the probability that the batting team wins, separately for innings 1 and innings 2. It uses match-state features including balls remaining, wickets lost, scoring rate, runs required, required rate, and match format.

Predictions are generated using **five-fold out-of-fold cross-fitting by match**. Each match is evaluated by a model that was not trained on that match, reducing the risk of using a player's own match outcome to construct the win probabilities used to assess that player's performance.

The model reports AUC and log loss, alongside a constant-probability baseline log loss, in `wp_model_metrics.csv`.

### 4.2 Delivery leverage

For a given match state, leverage is the expected absolute change in win probability caused by the next legal delivery, averaged over possible next-ball outcomes and their estimated probabilities. The implementation keeps runs and wickets together in the outcome categories, so a wicket with completed runs is not treated as an ordinary wicket-free delivery.

A higher leverage value means more is at stake on that delivery; it does not by itself mean that the delivery is high pressure or that a player performed well.

### 4.3 Which deliveries count as high pressure?

The project uses different rules for the two innings:

- **First innings — collapse pressure:** a delivery is marked high pressure when at least **4 wickets have already fallen within the first 84 legal balls** of the innings.
- **Second innings — chase leverage:** a legal delivery is marked high pressure when its leverage is at or above the **75th percentile** of legal second-innings leverage (the top 25%).

The first-innings rule is used because the first-innings win-probability model is less informative early in an innings. The chase rule is based on delivery leverage.

The configured values are in `rr_scout/config.py`:

```python
HIGH_PRESSURE_QUANTILE = 0.75
FIRST_INN_COLLAPSE_WKTS = 4
FIRST_INN_COLLAPSE_BALLS = 84
```

### 4.4 Player credit and ranking

- Batters receive credit for deliveries they faced; wides are excluded from batter-ball attribution.
- Bowlers receive credit for deliveries they bowled. Bowler wicket credit excludes dismissals not credited to the bowler, such as run-outs.
- Performance is measured as win-probability added above the expected contribution of an average delivery from the same state.
- The rating combines an absolute pressure-performance measure with the difference between a player's pressure and ordinary performance.
- The configured blend is **75% absolute pressure score and 25% pressure-versus-ordinary score**.
- Pressure estimates are shrunk toward the eligible-player population mean using **150 pseudo-balls** (`PRESSURE_SHRINK_K = 150`).
- A match-level bootstrap is used to estimate uncertainty. Players are ranked by the **20th-percentile lower confidence bound** of the blended score, rather than by the point estimate alone.

The current minimum eligibility settings are:

| Role | Minimum high-pressure balls | Minimum matches containing pressure balls |
|---|---:|---:|
| Batter | 150 | 15 |
| Bowler | 150 | 15 |

These thresholds, the blend weights, shrinkage, bootstrap count, and other parameters are centrally configured in `rr_scout/config.py`.

### 4.5 Opposition-strength adjustment

The configured ranking uses opposition adjustment. Team batting and bowling effects are estimated separately within the T20I, IPL, and emerging-T20 pools. Each delivery is then associated with the strength tier of the opposition faced. The expected delivery outcome is conditioned on that tier, so performance against stronger opposition is assessed against a different expectation from performance against weaker opposition.

Unadjusted ratings are also saved to allow comparison. The pressure flag itself is based on neutral phase-level leverage probabilities; opposition adjustment changes player credit expectations, not which deliveries qualify as high pressure.

## 5. Main outputs

All primary outputs are written to `outputs/pressure/`.

| File | Description |
|---|---|
| `top10_bat_all_players.csv` | Final Top 10 batter ranking. |
| `top10_bowl_all_players.csv` | Final Top 10 bowler ranking. |
| `ratings_bat.csv` | Full eligible batter ranking with opposition adjustment. |
| `ratings_bowl.csv` | Full eligible bowler ranking with opposition adjustment. |
| `ratings_bat_unadjusted.csv` | Batter ratings without opposition adjustment. |
| `ratings_bowl_unadjusted.csv` | Bowler ratings without opposition adjustment. |
| `ball_pressure.csv.gz` | Delivery-level pressure flags, leverage, win-probability values, and player-credit fields. Generated by `run_pressure.py`; used by validation. |
| `wp_model_metrics.csv` | First- and second-innings win-probability model metrics. |
| `team_effects.csv` | Estimated team batting and bowling effects used for opposition adjustment. |
| `validation_summary.csv` | Combined ranking stability and sensitivity results. |
| `validation_*_baseline.csv` | Baseline eligible-player rankings for batters and bowlers. |
| `validation_*_half_a.csv`, `validation_*_half_b.csv` | Match-level split-half rankings. |
| `validation_*_strict_sample.csv` | Rankings under the 150-pressure-ball / 15-match sample requirement. |

The two `top10_*` files contain the final ten-player lists. The `ratings_*` files retain the wider eligible-player rankings and supporting metrics.

## 6. Validation performed

The validation module evaluates the configured rankings using:

- **Match-level split-half stability:** compares rankings from two disjoint groups of matches.
- **Pressure-rule sensitivity:** tests alternative chase leverage quantiles and first-innings collapse thresholds.
- **Strict sample-size sensitivity:** checks the ranking under the 150-pressure-ball and 15-pressure-match requirements.
- **Saved pressure-rule verification:** checks that the pressure flags in `ball_pressure.csv.gz` agree with the current configuration before calculating validation rankings.

Interpret split-half results carefully. The eligible player pool becomes substantially smaller when the matches are divided into halves, so overlap and rank correlation can be affected by sparse samples. Validation is a robustness analysis, not proof that the rankings are universally reproducible across every competition or season.

## 7. Data and modelling limitations

- **Approximate extras reconstruction:** the source data stores byes and leg-byes with `runs = 0`. The pipeline uses the configured `UNRECORDED_EXTRA_RUNS = 1` approximation for these deliveries and adds the numeric `extras_2_runs` field. Reconstructed team totals are therefore approximate where the source does not record the runs directly.
- **Excluded match types:** Super Overs, tied matches, no-result/abandoned matches, and matches with revised playing conditions are excluded from the modelled sample.
- **First-innings pressure is rule-based:** it is defined by the configured early-innings collapse condition rather than the chase leverage threshold.
- **Model scope:** the win-probability model is trained on the supplied dataset. Its performance metrics describe this dataset and should not be interpreted as a guarantee of accuracy on unseen competitions or future seasons.
- **Sample-size effects:** players with limited pressure exposure are excluded by the minimum eligibility criteria; the remaining rankings still depend on the competitions and matches represented in the supplied data.
- **Ranking interpretation:** the score is a model-based estimate of performance under the defined pressure rules. It is not a direct measure of overall player quality, a forecast of future performance, or an auction valuation.
- **Coverage and balance:** the pressure definition and match mix influence which phases, bowling styles, and player roles appear in the pressure sample. The rankings should be read alongside the validation and supporting CSVs rather than treated as context-free comparisons.

## 8. Repository structure

```text
.
├── data/                         # Add the four source CSVs here (not bundled)
├── outputs/
│   └── pressure/                 # Generated rankings and diagnostics
├── rr_scout/
│   ├── config.py                 # Central configuration
│   ├── data.py                   # Input loading and match-state reconstruction
│   ├── win_prob.py               # Cross-fitted win-probability model
│   ├── pressure.py               # Delivery leverage and pressure flags
│   ├── opposition.py             # Opposition-strength adjustment
│   ├── ratings.py                # Player credits, shrinkage, bootstrap rankings
│   └── validation.py              # Ranking validation and sensitivity analysis
├── run_pressure.py               # Main Round 1 pipeline
└── requirements.txt              # Python dependencies
```

Some modules and files in the bundle are retained from the wider project repository and are not required to run the Round 1 pressure-ranking entry point. Use `run_pressure.py` for the Round 1 analysis described here.

## 9. Reproducibility note

The reported rankings depend on the exact input data pack and the configuration in `rr_scout/config.py`. To reproduce them, use the same source CSVs and configuration. The data pack and generated delivery-level file `ball_pressure.csv.gz` are not included in the minimal source bundle; the main pipeline regenerates the latter from the four input CSVs.
