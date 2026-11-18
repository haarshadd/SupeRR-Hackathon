"""Pressure = how much the next ball can move the match (leverage), measured from win probability.

leverage(state) = sum over next-ball outcomes o of  P(o | phase) * |WP(state after o) - WP(state now)|

Each ball also gets exp_wpa (the win-probability change an AVERAGE delivery would produce from that state)
and wpae = wpa - exp_wpa ("win probability added over expectation"), which is what ratings credit.

Opposition adjustment: when the delivery table carries the opposing-strength buckets (see opposition.py),
two more expectations are computed with outcome probabilities conditional on (group, bucket, phase):
  wpae_bat   expectation vs the strength of the BOWLING side  -> credit for batters
  wpae_bowl  expectation vs the strength of the BATTING side  -> credit for bowlers
Leverage and the pressure flag always use the neutral (phase-only) probabilities, so which balls count as
"high pressure" does not depend on the adjustment.

* Chases (innings 2): high pressure = top (1 - HIGH_PRESSURE_QUANTILE) of leverage.
* First innings: the first-innings win-probability model is weak (a score says little about who wins
  until late), so leverage there is unreliable. First-innings pressure is a simple collapse flag instead:
  >= FIRST_INN_COLLAPSE_WKTS wickets down inside the first FIRST_INN_COLLAPSE_BALLS legal balls.
"""
import numpy as np
import pandas as pd
from . import config as C




# Keep runs and wickets as joint delivery outcomes.
# This preserves cases such as a run-out with one or two completed runs.
OUTCOME_CODES = (
    [str(i) for i in range(7)]
    + [f"W{i}" for i in range(7)]
)


def _codes(L):
    """
    Encode the complete outcome of a legal delivery.

    Examples:
      0  = dot ball, no wicket
      1  = one run, no wicket
      W0 = wicket, zero runs
      W1 = wicket, one run
      W2 = wicket, two runs

    Uses team_runs so extras are included in the reconstructed outcome.
    """
    runs = (
        pd.to_numeric(L["team_runs"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=6)
        .round()
        .astype(int)
        .to_numpy()
    )

    wickets = L["is_wicket"].fillna(False).astype(bool).to_numpy()

    return np.where(
        wickets,
        "W" + runs.astype(str),
        runs.astype(str)
    )





def outcome_probs(s):
    """Next-legal-ball outcome distribution by phase."""
    L = s[s.legal]

    P = pd.crosstab(
        L.phase.values,
        _codes(L),
        normalize="index"
    ).reindex(columns=OUTCOME_CODES).fillna(0.0)

    return P.div(P.sum(axis=1), axis=0)




def outcome_table(s, bucket_col):
    """Outcome distribution by pool, opposing-strength bucket and phase."""
    L = s[s.legal]

    P = pd.crosstab(
        [
            L.pool.values,
            L[bucket_col].values,
            L.phase.astype(str).values
        ],
        _codes(L),
        normalize="index"
    )

    P = P.reindex(columns=OUTCOME_CODES).fillna(0.0)

    return P.div(P.sum(axis=1), axis=0)



def _lookup(P, d, bucket_col, neutral):
    idx = pd.MultiIndex.from_arrays([d.pool.values, d[bucket_col].values, d.phase.astype(str).values])
    out = np.array(P.reindex(idx).values, dtype=float)
    miss = np.isnan(out).any(axis=1)
    out[miss] = neutral[miss]
    return out



def compute_pressure(s, wp):
    """Return one row per delivery with win probabilities, leverage and pressure flag."""

    # Remove deliveries beyond the normal 120-ball innings.
    s = s[s.legal_before < C.BALLS_PER_INNINGS].copy()

    # Neutral outcome probabilities.
    P = outcome_probs(s)

    # Optional opposition-adjusted outcome probabilities.
    adj = "bucket_faced_by_bat" in s.columns

    if adj:
        P_bat = outcome_table(s, "bucket_faced_by_bat")
        P_bowl = outcome_table(s, "bucket_faced_by_bowl")

    out = []

    # Process BOTH innings independently.
    for inn in (1, 2):

        d = s[s.innings == inn].copy()

        if d.empty:
            continue

        def args(r, w, legal_balls):
            return (
                inn,
                r,
                w,
                legal_balls,
                d.target.values,
                d.t20i.values,
                d.match_id.values
            )

        # Actual match state before the delivery.
        before = wp.predict(
            *args(
                d.runs_before.values,
                d.wkts_before.values,
                d.legal_before.values
            )
        )

        # Actual match state after the delivery.
        after = wp.predict(
            *args(
                d.runs_before.values + d.team_runs.values,
                d.wkts_before.values + d.is_wicket.values.astype(int),
                d.legal_before.values + d.legal.values.astype(int)
            )
        )

        # Counterfactual state: same runs and balls, but no wicket.
        # Used to avoid assigning non-bowler wickets to the bowler,
        # or a non-striker's run-out to the striker.
        after_no_wicket = wp.predict(
            *args(
                d.runs_before.values + d.team_runs.values,
                d.wkts_before.values,
                d.legal_before.values + d.legal.values.astype(int)
            )
        )

        lev = np.zeros(len(d))
        exp = np.zeros(len(d))
        exp_b = np.zeros(len(d))
        exp_w = np.zeros(len(d))

        # Expected outcome probabilities for the current phase.
        pp = P.loc[d.phase.astype(str).values].values

        if adj:
            pp_b = _lookup(
                P_bat, d, "bucket_faced_by_bat", pp
            )
            pp_w = _lookup(
                P_bowl, d, "bucket_faced_by_bowl", pp
            )

        # Evaluate every possible joint outcome:
        # 0-6 runs without a wicket, and W0-W6 with a wicket.
        for k, o in enumerate(OUTCOME_CODES):

            if o.startswith("W"):
                r = int(o[1:])
                w = 1
            else:
                r = int(o)
                w = 0

            # Hypothetical state after one legal delivery.
            hyp = wp.predict(
                *args(
                    d.runs_before.values + r,
                    d.wkts_before.values + w,
                    d.legal_before.values + 1
                )
            )

            # Expected leverage and expected WP movement.
            lev += pp[:, k] * np.abs(hyp - before)
            exp += pp[:, k] * (hyp - before)

            if adj:
                exp_b += pp_b[:, k] * (hyp - before)
                exp_w += pp_w[:, k] * (hyp - before)

        # Without opposition adjustment, both expectations
        # use the same neutral outcome distribution.
        if not adj:
            exp_b = exp.copy()
            exp_w = exp.copy()

                # Actual win-probability added.
        wpa = after - before
        wpa_no_wicket = after_no_wicket - before

        has_wicket = (
            d.is_wicket.fillna(False).astype(bool).to_numpy()
        )
        striker_dismissed = (
            d.dis_bat.fillna(False).astype(bool).to_numpy()
        )
        bowler_gets_wicket = (
            d.bowler_wkt.fillna(False).astype(bool).to_numpy()
        )

        # Attribute wicket impact only to the player responsible:
        # - Batter: only when the striker was dismissed.
        # - Bowler: only for a bowler-creditable dismissal.
        wpae_bat_player = np.where(
            has_wicket & ~striker_dismissed,
            wpa_no_wicket - exp_b,
            wpa - exp_b
        )

        wpae_bowl_player = np.where(
            has_wicket & ~bowler_gets_wicket,
            wpa_no_wicket - exp_w,
            wpa - exp_w
        )

        # Append each delivery EXACTLY ONCE.
        out.append(
            d.assign(
                wp_before=before,
                wp_after=after,
                wpa=wpa,
                leverage=lev,
                exp_wpa=exp,
                wpae=wpa - exp,
                wpae_bat=wpa - exp_b,
                wpae_bowl=wpa - exp_w,
                wpae_bat_player=wpae_bat_player,
                wpae_bowl_player=wpae_bowl_player,
                wpa_no_wicket=wpa_no_wicket
            )
        )

    if not out:
        raise ValueError("No first- or second-innings deliveries available.")

    r = pd.concat(out, ignore_index=True)

    # Guard against accidental delivery duplication or loss.
    expected_rows = int(s.innings.isin([1, 2]).sum())

    if len(r) != expected_rows:
        raise RuntimeError(
            f"Delivery count mismatch: expected {expected_rows}, "
            f"generated {len(r)}"
        )

    # High-pressure chase threshold.
    chase_legal = r.loc[
        (r.innings == 2) & r.legal.astype(bool),
        "leverage"
    ]

    if chase_legal.empty:
        raise ValueError("No legal chase deliveries available.")


    # Calculate separate leverage thresholds for each innings.
    # This avoids comparing leverage values from two different
    # win-probability models on a shared scale.

    first_legal = r.loc[
        (r.innings == 1) & r.legal.astype(bool),
        "leverage"
    ]

    chase_legal = r.loc[
        (r.innings == 2) & r.legal.astype(bool),
        "leverage"
    ]

    if first_legal.empty or chase_legal.empty:
        raise ValueError("Both innings must contain legal deliveries.")

    first_thr = first_legal.quantile(C.HIGH_PRESSURE_QUANTILE)
    chase_thr = chase_legal.quantile(C.HIGH_PRESSURE_QUANTILE)


    # DIAGNOSTIC: inspect leverage distribution by innings and phase.
    # This does not change pressure classification.

    print("\nLeverage distribution by innings and phase:")

    for inn in (1, 2):
        for phase in ("PP", "MID", "DEATH"):
            values = r.loc[
                (r.innings == inn)
                & (r.phase.astype(str) == phase)
                & r.legal.astype(bool),
                "leverage"
            ]

            if values.empty:
                continue

            print(
                f"Innings {inn} | {phase}: "
                f"count={len(values)}, "
                f"mean={values.mean():.5f}, "
                f"median={values.median():.5f}, "
                f"p75={values.quantile(0.75):.5f}, "
                f"p90={values.quantile(0.90):.5f}, "
                f"p99={values.quantile(0.99):.5f}"
            )

    print(
        "\nFirst innings global threshold:",
        round(float(first_thr), 5)
    )

    print(
        "Chase global threshold:",
        round(float(chase_thr), 5)
    )


    # Apply each innings' own threshold.

    first_pressure = (
        (r.innings == 1)
        & r.legal.astype(bool)
        & (r.leverage >= first_thr)
    )

    chase_pressure = (
        (r.innings == 2)
        & r.legal.astype(bool)
        & (r.leverage >= chase_thr)
    )


    r["pressure_tier"] = np.select(
        [first_pressure, chase_pressure],
        ["first_innings_leverage", "chase_leverage"],
        default="ordinary"
    )

    r["high_pressure"] = r["pressure_tier"] != "ordinary"

        # Chases: pressure is the top 25% of legal-ball leverage.
    chase_legal = r.loc[
        (r.innings == 2) & r.legal.astype(bool),
        "leverage"
    ]

    if chase_legal.empty:
        raise ValueError("No legal chase deliveries available.")

    chase_thr = chase_legal.quantile(C.HIGH_PRESSURE_QUANTILE)

    # First innings: use the explicit collapse rule.
    # The first-innings WP model has weak predictive performance,
    # so its leverage estimates are not used to classify pressure.
    first_pressure = (
        (r.innings == 1)
        & (r.wkts_before >= C.FIRST_INN_COLLAPSE_WKTS)
        & (r.legal_before < C.FIRST_INN_COLLAPSE_BALLS)
    )

    # Second innings: classify high-leverage states.
    chase_pressure = (
        (r.innings == 2)
        & (r.leverage >= chase_thr)
    )

    print(
        "\nFirst-innings pressure rule: "
        f"{C.FIRST_INN_COLLAPSE_WKTS}+ wickets lost "
        f"within the first {C.FIRST_INN_COLLAPSE_BALLS} legal balls."
    )
    print(
        f"Chase leverage threshold (top "
        f"{1 - C.HIGH_PRESSURE_QUANTILE:.0%}): "
        f"{chase_thr:.5f}"
    )

    r["pressure_tier"] = np.select(
        [first_pressure, chase_pressure],
        ["first_innings_collapse", "chase_leverage"],
        default="ordinary"
    )

    r["high_pressure"] = r["pressure_tier"] != "ordinary"

    # Preserve compatibility with run_pressure.py.
    r.attrs["leverage_threshold"] = float(chase_thr)
    r.attrs["first_innings_pressure_rule"] = (
        f"{C.FIRST_INN_COLLAPSE_WKTS}+ wickets "
        f"within {C.FIRST_INN_COLLAPSE_BALLS} legal balls"
    )


    return r.sort_values(
        ["match_id", "innings", "innings_ball_num"]
    )


KEEP = ["match_id", "innings", "innings_ball_num", "overid", "phase", "batting_team_id", "bowling_team_id", "striker_id",
        "bowler_id", "runs", "team_runs", "is_wicket", "legal", "bowler_wkt", "dis_bat", "runs_before", "wkts_before",
        "legal_before", "target", "t20i", "comp_name", "wide", "nb", "rt", "wp_before", "wp_after", "wpa",
        "exp_wpa", "wpae", "wpae_bat", "wpae_bowl","wpae_bat_player", "wpae_bowl_player", "wpa_no_wicket", "leverage", "pressure_tier", "high_pressure"]