
"""
Round 1 pressure-ranking validation.

Run from the project root:
    python -m rr_scout.validation

Uses the saved ball_pressure.csv.gz file. Does not depend on
auction availability or modify the main pipeline.
"""

import numpy as np
import pandas as pd

from . import config as C
from .data import load_raw, prepare_balls, player_names
from .ratings import rate_players


def _as_bool(series, column_name):
    """Convert CSV boolean representations safely."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    values = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }

    ignored = {"", "nan", "none", "null"}
    unknown = set(values.unique()) - set(mapping) - ignored

    if unknown:
        raise ValueError(
            f"Unexpected boolean values in {column_name}: "
            f"{sorted(unknown)}"
        )

    return values.map(mapping).fillna(False).astype(bool)


def load_pressure_data():
    """Load and validate the saved delivery-level pressure data."""
    path = C.PRESSURE_OUT_DIR / "ball_pressure.csv.gz"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run the pressure pipeline first."
        )

    r = pd.read_csv(path)

    required = [
        "match_id", "innings", "striker_id", "bowler_id",
        "leverage", "high_pressure", "legal", "wide",
        "is_wicket", "runs", "t20i", "wpae",
        "wpae_bat", "wpae_bowl", "rt", "dis_bat",
        "bowler_wkt", "wkts_before", "legal_before",
        "wpa_no_wicket", "wpae_bat_player",
        "wpae_bowl_player"
    ]

    missing = [col for col in required if col not in r.columns]

    if missing:
        raise ValueError(
            f"Pressure file is missing required columns: {missing}"
        )

    for col in [
        "high_pressure", "legal", "wide",
        "is_wicket", "t20i", "dis_bat", "bowler_wkt"
    ]:
        r[col] = _as_bool(r[col], col)

    for col in [
        "innings", "leverage", "wpae",
        "wpae_bat", "wpae_bowl", "wpa_no_wicket",
        "wpae_bat_player", "wpae_bowl_player",
        "runs", "rt", "wkts_before", "legal_before"
    ]:
        r[col] = pd.to_numeric(r[col], errors="coerce")

    r = r.dropna(
        subset=[
            "match_id", "innings", "striker_id",
            "bowler_id", "leverage"
        ]
    ).copy()

    if r.empty:
        raise ValueError("The pressure file contains no usable deliveries.")
    validate_saved_pressure_rule(r)

    return r


def load_player_names():
    """Recreate the player ID-to-name mapping from the source data."""
    raw = load_raw()
    balls = prepare_balls(raw)
    return player_names(balls)


def rank(r, role, names, adjust):
    """Calculate a player ranking using the existing ratings module."""
    result = rate_players(
        r,
        role,
        names,
        adjust=adjust
    )

    if not result.empty:
        result = result.sort_values("rank")

    return result


def top_ids(table, n=10):
    """Return the IDs of the top n ranked players."""
    if table.empty:
        return set()

    return set(table.head(n).index)


def top_names(table, n=10):
    """Return readable names for the top n players."""
    if table.empty:
        return ""

    return "; ".join(
        table.head(n)["name"].fillna("Unknown").astype(str)
    )


def rank_correlation(table_a, table_b):
    """Spearman correlation between rankings of commonly eligible players."""
    common = table_a.index.intersection(table_b.index)

    if len(common) < 2:
        return np.nan

    return float(
        table_a.loc[common, "rank"].corr(
            table_b.loc[common, "rank"],
            method="spearman"
        )
    )

def rank_strict_sample(r, role, names):
    """Sensitivity ranking with stricter player sample requirements."""
    old_bat = C.MIN_PRESSURE_BALLS_BAT
    old_bowl = C.MIN_PRESSURE_BALLS_BOWL
    old_matches = C.MIN_PRESSURE_MATCHES

    try:
        C.MIN_PRESSURE_BALLS_BAT = 150
        C.MIN_PRESSURE_BALLS_BOWL = 150
        C.MIN_PRESSURE_MATCHES = 15

        return rank(
            r,
            role,
            names,
            adjust=C.OPPOSITION_ADJUST
        )
    finally:
        C.MIN_PRESSURE_BALLS_BAT = old_bat
        C.MIN_PRESSURE_BALLS_BOWL = old_bowl
        C.MIN_PRESSURE_MATCHES = old_matches

def comparison_row(role, test, scenario, reference, candidate):
    """Summarise a candidate ranking against a reference ranking."""
    ref_top = top_ids(reference)
    candidate_top = top_ids(candidate)

    overlap = len(ref_top & candidate_top)

    return {
        "role": role,
        "test": test,
        "scenario": scenario,
        "eligible_players": len(candidate),
        "top10_overlap": overlap,
        "spearman_common": rank_correlation(
            reference,
            candidate
        ),
        "top10_names": top_names(candidate)
    }


def split_matches(r):
    """Create reproducible, non-overlapping match-level halves."""
    ids = r.match_id.drop_duplicates().to_numpy().copy()

    if len(ids) < 2:
        raise ValueError("At least two matches are needed for split validation.")

    rng = np.random.RandomState(C.RANDOM_SEED + 100)
    rng.shuffle(ids)

    cut = len(ids) // 2

    half_a = set(ids[:cut])
    half_b = set(ids[cut:])

    if half_a & half_b:
        raise RuntimeError("Match leakage detected between validation halves.")

    return half_a, half_b


def pressure_variant(
    r,
    chase_quantile,
    collapse_wickets,
    collapse_balls
):
    """
    Reconstruct pressure flags using the actual Round 1 methodology.

    First innings:
        Collapse rule based on wickets lost and legal balls elapsed.

    Second innings:
        Leverage threshold estimated from legal deliveries,
        then applied to all deliveries in the innings.
    """
    if not 0 < chase_quantile < 1:
        raise ValueError("Chase quantile must be between 0 and 1.")

    if collapse_wickets < 1 or collapse_balls < 1:
        raise ValueError("Collapse thresholds must be positive.")

    result = r.copy()

    chase_legal = result.loc[
        (result.innings == 2) & result.legal,
        "leverage"
    ]

    if chase_legal.empty:
        raise ValueError("No legal chase deliveries found.")

    chase_threshold = float(
        chase_legal.quantile(chase_quantile)
    )

    first_pressure = (
        (result.innings == 1)
        & (result.wkts_before >= collapse_wickets)
        & (result.legal_before < collapse_balls)
    )

    chase_pressure = (
        (result.innings == 2)
        & (result.leverage >= chase_threshold)
    )

    result["high_pressure"] = (
        first_pressure | chase_pressure
    )

    return result, {
        "chase_threshold": chase_threshold,
        "collapse_wickets": collapse_wickets,
        "collapse_balls": collapse_balls
    }


def validate_saved_pressure_rule(r):
    """Ensure saved pressure flags match the configured methodology."""

    expected, _ = pressure_variant(
        r,
        C.HIGH_PRESSURE_QUANTILE,
        C.FIRST_INN_COLLAPSE_WKTS,
        C.FIRST_INN_COLLAPSE_BALLS
    )

    mismatches = (
        expected.high_pressure.astype(bool)
        != r.high_pressure.astype(bool)
    ).sum()

    if mismatches:
        raise ValueError(
            f"{mismatches} saved pressure flags do not match "
            "the configured pressure methodology. "
            "Regenerate ball_pressure.csv.gz using run_pressure.py."
        )

    print("Saved pressure flags: verified against configured rules.")


def run_validation():
    """Run the Round 1 player-ranking validation suite."""
    print("=" * 65)
    print("ROUND 1 — PRESSURE RANKING VALIDATION")
    print("=" * 65)

    r = load_pressure_data()
    names = load_player_names()

    print(f"\nLoaded deliveries: {len(r):,}")
    print(f"Matches: {r.match_id.nunique():,}")
    print(f"Players in name mapping: {len(names):,}")

    half_a, half_b = split_matches(r)

    print(
        f"Split A: {len(half_a)} matches | "
        f"Split B: {len(half_b)} matches"
    )

    rows = []
    roles = [
        ("bat", "BATTERS"),
        ("bowl", "BOWLERS")
    ]

    for role, label in roles:
        print("\n" + "=" * 65)
        print(f"{label}")
        print("=" * 65)

        # -------------------------------------------------
        # 1. Baseline ranking
        # -------------------------------------------------
        baseline = rank(
            r,
            role,
            names,
            adjust=C.OPPOSITION_ADJUST
        )

        baseline.to_csv(
            C.PRESSURE_OUT_DIR /
            f"validation_{role}_baseline.csv"
        )

        print(f"\nBaseline eligible players: {len(baseline)}")
        print("Baseline Top 10:")
        print(top_names(baseline))

        rows.append({
            "role": role,
            "test": "baseline",
            "scenario": "configured",
            "eligible_players": len(baseline),
            "top10_overlap": 10 if len(baseline) >= 10 else len(baseline),
            "spearman_common": 1.0,
            "top10_names": top_names(baseline)
        })

        # -------------------------------------------------
        # 2. Match-level split stability
        # -------------------------------------------------
        data_a = r[r.match_id.isin(half_a)].copy()
        data_b = r[r.match_id.isin(half_b)].copy()

        ranking_a = rank(
            data_a, role, names,
            adjust=C.OPPOSITION_ADJUST
        )

        ranking_b = rank(
            data_b, role, names,
            adjust=C.OPPOSITION_ADJUST
        )

        ranking_a.to_csv(
            C.PRESSURE_OUT_DIR /
            f"validation_{role}_half_a.csv"
        )

        ranking_b.to_csv(
            C.PRESSURE_OUT_DIR /
            f"validation_{role}_half_b.csv"
        )

        rows.append(
            comparison_row(
                role, "split_vs_baseline", "half_A",
                baseline, ranking_a
            )
        )

        rows.append(
            comparison_row(
                role, "split_vs_baseline", "half_B",
                baseline, ranking_b
            )
        )

        rows.append(
            comparison_row(
                role, "split_A_vs_B", "independent_halves",
                ranking_a, ranking_b
            )
        )

        print("\nSplit-sample stability:")
        print(
            "  Half A eligible:", len(ranking_a),
            "| Half B eligible:", len(ranking_b)
        )

        print(
            "  Top 10 overlap A vs B:",
            len(top_ids(ranking_a) & top_ids(ranking_b)),
            "/ 10"
        )

        print(
            "  Spearman rank correlation A vs B:",
            round(rank_correlation(ranking_a, ranking_b), 3)
            if np.isfinite(rank_correlation(ranking_a, ranking_b))
            else "insufficient common players"
        )

                # -------------------------------------------------
        # 3. Pressure-rule sensitivity
        # -------------------------------------------------
        print("\nPressure-rule sensitivity:")

        base_wickets = C.FIRST_INN_COLLAPSE_WKTS
        base_balls = C.FIRST_INN_COLLAPSE_BALLS
        base_quantile = C.HIGH_PRESSURE_QUANTILE

        scenarios = [
            (
                "chase_top_30_percent",
                0.70,
                base_wickets,
                base_balls
            ),
            (
                "chase_top_20_percent",
                0.80,
                base_wickets,
                base_balls
            ),
            (
                f"collapse_{max(1, base_wickets - 1)}_wkts_{base_balls}_balls",
                base_quantile,
                max(1, base_wickets - 1),
                base_balls
            ),
            (
                f"collapse_{base_wickets}_wkts_{max(1, base_balls - 12)}_balls",
                base_quantile,
                base_wickets,
                max(1, base_balls - 12)
            ),
            (
                f"collapse_{base_wickets}_wkts_{min(C.BALLS_PER_INNINGS, base_balls + 12)}_balls",
                base_quantile,
                base_wickets,
                min(C.BALLS_PER_INNINGS, base_balls + 12)
            )
        ]

        for scenario, q, wickets, balls in scenarios:

            variant, thresholds = pressure_variant(
                r,
                q,
                wickets,
                balls
            )

            candidate = rank(
                variant,
                role,
                names,
                adjust=C.OPPOSITION_ADJUST
            )

            candidate.to_csv(
                C.PRESSURE_OUT_DIR /
                f"validation_{role}_{scenario}.csv"
            )

            rows.append(
                comparison_row(
                    role,
                    "pressure_rule",
                    scenario,
                    baseline,
                    candidate
                )
            )

            overlap = len(
                top_ids(baseline) & top_ids(candidate)
            )

            print(
                f"  {scenario}: "
                f"eligible={len(candidate)}, "
                f"Top 10 overlap={overlap}/10, "
                f"chase threshold={thresholds['chase_threshold']:.5f}"
            )
            # -------------------------------------------------
        # Strict sample-size sensitivity
        # -------------------------------------------------
        strict = rank_strict_sample(r, role, names)

        strict.to_csv(
            C.PRESSURE_OUT_DIR /
            f"validation_{role}_strict_sample.csv"
        )

        overlap = len(
            top_ids(baseline) & top_ids(strict)
        )

        print("\nStrict sample requirement:")
        print("  Minimum pressure balls: 150")
        print("  Minimum pressure matches: 15")
        print("  Eligible players:", len(strict))
        print("  Top 10 overlap with baseline:", overlap, "/ 10")
        print("  Strict Top 10:")
        print(top_names(strict))

        rows.append(
            comparison_row(
                role,
                "sample_size",
                "150_balls_15_matches",
                baseline,
                strict
            )
        )

        if role == "bat":
            rashid = strict[
                strict["name"].astype(str).str.casefold()
                == "rashid khan"
            ]

            if rashid.empty:
                print("  Rashid Khan: NOT ELIGIBLE under strict criteria")
            else:
                x = rashid.iloc[0]
                print(
                    f"  Rashid Khan strict rank: {int(x['rank'])}"
                    f" | pressure balls: {int(x['n_p'])}"
                    f" | pressure matches: {int(x['matches_p'])}"
                    f" | score: {x['blend_lcb']:.4f}"
                )

    # -----------------------------------------------------
    # Save combined results
    # -----------------------------------------------------
    summary = pd.DataFrame(rows)

    output = C.PRESSURE_OUT_DIR / "validation_summary.csv"
    summary.to_csv(output, index=False)

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(
        summary[
            [
                "role", "test", "scenario",
                "eligible_players", "top10_overlap",
                "spearman_common"
            ]
        ].to_string(index=False)
    )

    print(f"\nFull validation summary saved to:\n{output}")
    print("\nValidation completed.")

    


if __name__ == "__main__":
    run_validation()
