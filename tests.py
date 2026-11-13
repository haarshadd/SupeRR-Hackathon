from rr_scout.data import load_raw, prepare_balls, add_match_state
raw = load_raw(); b = prepare_balls(raw); s = add_match_state(b, raw)
print(s.match_id.nunique(), len(s))