# RR auction scout

Finds auction targets for Rajasthan Royals from the hackathon data pack, starting from a
hand-curated list of "core" players per IPL team (`cores.csv`).

## Run
```
pip install -r requirements.txt
# put the data pack CSVs in data/   (ball_by_ball.csv, matches.csv, competitions.csv,
#                                    competition_team_player_mapping.csv)
python run_pipeline.py
```
Everything lands in `outputs/`.

## Layout
| File | Job |
|---|---|
| `cores.csv` | Your curated core players (team, player). Spelling is fuzzy-matched to the data; the run fails loudly if a name has no match. |
| `watchlist.csv` | Players to profile regardless of any filter (e.g. Vignesh Puthur). |
| `rr_scout/config.py` | All thresholds and pool rules. |
| `rr_scout/data.py` | Load + clean ball-by-ball; derived columns (legal ball, phase, bowler runs, bowler wickets). |
| `rr_scout/features.py` | Shrunk style profiles + value-over-average for every batter and bowler. |
| `rr_scout/players.py` | Segment (capped / uncapped / excluded), activity, 2026 squad, IPL-only ball counts. |
| `rr_scout/similarity.py` | Z-scores, cosine similarity, candidate pools, replacement ranking. |
| `rr_scout/gaps.py` | What the RR core covers/misses; gap modules (spin, death, powerplay, middle-over batter). |
| `rr_scout/contention.py` | Proxy for rival demand (similar rival core player; rivals lacking the niche). |
| `run_pipeline.py` | Orchestrates the above and writes the outputs. |

## Outputs
`shortlist_10.csv` (5 replacements + 5 gap-fillers) - `replacement_candidates.csv` (top 8 per core player, established and local) -
`gap_candidates.csv` - `local_gems.csv` - `rr_core_coverage.csv` - `rr_core_percentiles.csv` - `rival_niche_gaps.json` - `watchlist_report.csv`.

## Method in one paragraph
Each player gets a style vector (strike rate / economy by phase, pace-vs-spin, boundary and dot rates, usage), shrunk toward
the league mean so small samples do not look like stars. Replacement fit = cosine similarity to a core player, weighted by value
(runs above average per 100 balls, 6 runs per wicket) and by sample confidence `n/(n+150)`. Gap fit = percentile score on the factor the
core lacks. Rival demand = how many rival cores already own a similar player, plus how many rivals lack that niche.

## Known limits
- Extras runs (byes, leg-byes, wide overthrows) are not recorded, so totals are approximate.
- No auction prices, retention lists or nationality in the pack. "Local" here means *no senior international squad record in the pack*
  (so it can include overseas veterans and Indian players capped only in Tests). Verify before relying on it.
- `Pakistan`/`Pakistan A` players are excluded as IPL-ineligible.
- Value metric and weights are heuristics, not validated against prices.
- Local players have small samples: always read `n` and `conf` next to the score.
