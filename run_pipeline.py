"""End-to-end run:  python run_pipeline.py
Reads data/ + cores.csv, writes everything to outputs/.
"""
import json
import pandas as pd
from rr_scout import config as C
from rr_scout.data import load_raw, prepare_balls, player_names, resolve_cores
from rr_scout.features import batting_profiles, bowling_profiles
from rr_scout.players import build_meta
from rr_scout.similarity import zscore, candidate_pool, value_pct, rank_similar
from rr_scout.gaps import core_coverage, core_percentiles, score_gap_modules
from rr_scout.contention import rival_similarity, rival_niche_gaps

BAT_SHOW = ["sr_PP", "sr_MID", "sr_DEATH", "sr_Spin", "pos"]
BOWL_SHOW = ["btype", "eco_PP", "eco_MID", "eco_DEATH", "wkt100"]
NICHE_OF = {"spin_mid": "spin", "death_pace": "death", "pp_pace": "pp"}


def main():
    C.OUT_DIR.mkdir(exist_ok=True)
    raw = load_raw()
    b = prepare_balls(raw)
    names = player_names(b)
    cores = resolve_cores(C.CORES_CSV, names)
    meta = build_meta(b, raw, names)
    cores.to_csv(C.OUT_DIR / "cores_resolved.csv", index=False)
    print(f"balls={len(b):,}  cores resolved={len(cores)}  players with meta={len(meta):,}")

    bat, bowl = batting_profiles(b, 30), bowling_profiles(b, 30)
    Zb, Zw = zscore(bat, C.BAT_FEATURES), zscore(bowl, C.BOWL_FEATURES)
    vb, vw = value_pct(bat, "bat"), value_pct(bowl, "bowl")
    core_ids = set(cores.player_id)
    rr = cores[cores.team == C.MY_TEAM]
    rr_ids = list(rr.player_id)

    # ---- 1. what does the RR core cover / miss?
    core_coverage(b, rr_ids).to_csv(C.OUT_DIR / "rr_core_coverage.csv", header=["value"])
    core_percentiles(rr_ids, bat, bowl).assign(name=lambda d: d.player_id.map(names)).to_csv(C.OUT_DIR / "rr_core_percentiles.csv", index=False)

    # ---- 2. replacement candidates (established + local) for every RR core player
    pools = {(k, g): candidate_pool(prof, meta, k, g, core_ids)
             for k, prof in [("bat", bat), ("bowl", bowl)] for g in ["established", "local"]}
    repl_rows, best_est = [], []
    for _, r in rr.iterrows():
        i = r.player_id
        kind = "bowl" if bowl.n.get(i, 0) > bat.n.get(i, 0) else "bat"
        prof, Z, v, show = (bowl, Zw, vw, BOWL_SHOW) if kind == "bowl" else (bat, Zb, vb, BAT_SHOW)
        for g in ["established", "local"]:
            t = rank_similar(i, prof, Z, pools[(kind, g)], v).head(8)
            for rank, (pid, x) in enumerate(t.iterrows(), 1):
                repl_rows.append(dict(covers=r.matched, role=kind, group=g, rank=rank, player_id=pid, name=x["name"],
                                      segment=x.segment, squad_2026=x.squad_2026, n=int(x.n), conf=round(x.conf, 2),
                                      sim=round(x.sim, 3), value_pct=round(x.value_pct, 2), fit_score=round(x.fit_score, 3),
                                      raa100=round(x.raa100, 1), **{k: (round(x[k], 1) if k != "btype" else x[k]) for k in show}))
    repl = pd.DataFrame(repl_rows)
    repl.to_csv(C.OUT_DIR / "replacement_candidates.csv", index=False)

    # greedy distinct best established replacement per core; drop weakest to N_REPLACEMENTS
    est = repl[(repl.group == "established")]
    used, picks = set(), []
    for core in est.groupby("covers").fit_score.max().sort_values(ascending=False).index:
        for _, x in est[est.covers == core].sort_values("rank").iterrows():
            if x.player_id not in used:
                used.add(x.player_id); picks.append(x); break
    picks = sorted(picks, key=lambda x: -x.fit_score)[:C.N_REPLACEMENTS]

    # ---- 3. gap modules
    mods_est = score_gap_modules(pools[("bat", "established")], pools[("bowl", "established")], bat, bowl)
    mods_loc = score_gap_modules(pools[("bat", "local")], pools[("bowl", "local")], bat, bowl)
    gap_rows, gap_picks = [], []
    for mod, k in C.GAP_PLAN:
        got = 0
        for _, x in mods_est[mod].iterrows():
            if x.name in used:
                continue
            used.add(x.name); gap_picks.append((mod, x)); got += 1
            if got == k: break
    for grp, mods in [("established", mods_est), ("local", mods_loc)]:
        for mod, d in mods.items():
            for rank, (pid, x) in enumerate(d.head(8).iterrows(), 1):
                gap_rows.append(dict(module=mod, group=grp, rank=rank, player_id=pid, name=x["name"], segment=x.segment,
                                     squad_2026=x.squad_2026, n=int(x.n), conf=round(x.conf, 2), gap_score=round(x.gap_score, 3),
                                     raa100=round(x.raa100, 1)))
    pd.DataFrame(gap_rows).to_csv(C.OUT_DIR / "gap_candidates.csv", index=False)

    # ---- 4. contention proxy + shortlist
    niches = rival_niche_gaps(cores, bowl)
    json.dump(niches, open(C.OUT_DIR / "rival_niche_gaps.json", "w"), indent=1)
    rows = []
    def add(slot, covers, pid, x, score, vpct, kind, niche=None):
        Z, prof = (Zb, bat) if kind == "bat" else (Zw, bowl)
        hits = rival_similarity(pid, Z, prof, cores, names) if pid in Z.index else []
        lacking = niches.get(niche, []) if niche else []
        demand = len(hits) + len(lacking)
        label = "High competition" if (vpct >= 0.7 and demand >= 3) else ("Value" if demand <= 2 else "Medium")
        rows.append(dict(slot=slot, covers=covers, player_id=pid, name=x["name"], segment=x.segment, squad_2026=x.squad_2026,
                         n=int(x.n), conf=round(x.conf, 2), score=round(score, 3), value_pct=round(vpct, 2),
                         rivals_with_similar_core=len(hits), rivals_lacking_niche=len(lacking), label=label,
                         rival_detail="; ".join(hits)))
    for x in picks:
        add("replacement", x.covers, x.player_id, x, x.fit_score, x.value_pct, x.role)
    for mod, x in gap_picks:
        v = (vb if mod == "mid_bat_vs_spin" else vw).get(x.name, 0)
        add("gap", mod, x.name, x, x.gap_score, v, "bat" if mod == "mid_bat_vs_spin" else "bowl", NICHE_OF.get(mod))
    sl = pd.DataFrame(rows)
    sl.to_csv(C.OUT_DIR / "shortlist_10.csv", index=False)

    # ---- 5. local / uncapped gems: best local candidate per replacement core & per gap module
    loc = []
    for core, d in repl[repl.group == "local"].groupby("covers"):
        for _, x in d.sort_values("rank").head(3).iterrows():
            loc.append(dict(why=f"style-match for {core}", name=x["name"], segment=x.segment, squad_2026=x.squad_2026, n=x.n,
                            conf=x.conf, score=x.fit_score, raa100=x.raa100))
    for mod, d in mods_loc.items():
        for pid, x in d.head(3).iterrows():
            loc.append(dict(why=f"fills gap: {mod}", name=x["name"], segment=x.segment, squad_2026=x.squad_2026, n=int(x.n),
                            conf=round(x.conf, 2), score=round(x.gap_score, 3), raa100=round(x.raa100, 1)))
    pd.DataFrame(loc).to_csv(C.OUT_DIR / "local_gems.csv", index=False)

    # ---- 6. watchlist: profile named players regardless of pool filters
    wl_path = C.ROOT / "watchlist.csv"
    if wl_path.exists():
        out = []
        for nm in pd.read_csv(wl_path).player:
            ids = names[names.str.contains(nm, case=False, regex=False)].index
            for pid in ids:
                m = meta.loc[pid]
                for kind, prof in [("bat", bat), ("bowl", bowl)]:
                    if pid not in prof.index:
                        continue
                    one = prof.loc[[pid]].join(meta)
                    eb, ew = bat.iloc[0:0].join(meta), bowl.iloc[0:0].join(meta)
                    mods = score_gap_modules(one, ew, bat, bowl) if kind == "bat" else score_gap_modules(eb, one, bat, bowl)
                    for mod, d in mods.items():
                        if pid in d.index:
                            allm = (mods_loc if m.is_local else mods_est)[mod]
                            better = int((allm.gap_score > d.loc[pid, "gap_score"]).sum())
                            out.append(dict(player=nm, kind=kind, segment=m.segment, squad_2026=m.squad_2026, n=int(prof.loc[pid, "n"]),
                                            conf=round(prof.loc[pid, "conf"], 2), module=mod, gap_score=round(d.loc[pid, "gap_score"], 3),
                                            approx_rank_in_pool=better + 1, pool_size=len(allm)))
        pd.DataFrame(out).to_csv(C.OUT_DIR / "watchlist_report.csv", index=False)
        print("\nWATCHLIST\n", pd.DataFrame(out).to_string(index=False))

    print("\nRR core coverage:\n", core_coverage(b, rr_ids).round(2).to_string())
    print("\nSHORTLIST\n", sl[["slot", "covers", "name", "segment", "squad_2026", "n", "score", "value_pct",
                                 "rivals_with_similar_core", "rivals_lacking_niche", "label"]].to_string(index=False))
    print("\nRival niche gaps:", niches)
    print("\nLOCAL GEMS\n", pd.DataFrame(loc).to_string(index=False))


if __name__ == "__main__":
    main()
