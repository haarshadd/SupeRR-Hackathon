
"""Create a clean, minimal Round 1 submission bundle.

Run from the repository root:
    python tools/make_zip.py

Output:
    rr_scout_code.zip
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rr_scout_code.zip"

# Explicit project files required to run and understand Round 1.
PROJECT_FILES = [
    "README.md",
    "requirements.txt",
    "run_pressure.py",
    "tools/make_zip.py",
]

# Only modules used by the Round 1 pressure workflow.
MODULE_FILES = [
    "rr_scout/__init__.py",
    "rr_scout/config.py",
    "rr_scout/data.py",
    "rr_scout/opposition.py",
    "rr_scout/pressure.py",
    "rr_scout/ratings.py",
    "rr_scout/validation.py",
    "rr_scout/win_prob.py",
]

# Approved Round 1 outputs.
OUTPUT_FILES = [
    "outputs/pressure/top10_bat_all_players.csv",
    "outputs/pressure/top10_bowl_all_players.csv",
    "outputs/pressure/ratings_bat.csv",
    "outputs/pressure/ratings_bowl.csv",
    "outputs/pressure/ratings_bat_unadjusted.csv",
    "outputs/pressure/ratings_bowl_unadjusted.csv",
    "outputs/pressure/validation_summary.csv",
    "outputs/pressure/wp_model_metrics.csv",
    "outputs/pressure/team_effects.csv",
    "outputs/pressure/validation_bat_baseline.csv",
    "outputs/pressure/validation_bowl_baseline.csv",
    "outputs/pressure/validation_bat_half_a.csv",
    "outputs/pressure/validation_bat_half_b.csv",
    "outputs/pressure/validation_bowl_half_a.csv",
    "outputs/pressure/validation_bowl_half_b.csv",
    "outputs/pressure/validation_bat_strict_sample.csv",
    "outputs/pressure/validation_bowl_strict_sample.csv",
]

INCLUDE_FILES = PROJECT_FILES + MODULE_FILES + OUTPUT_FILES

MAX_FILE_BYTES = 3_000_000

# Check all required files before creating or replacing the ZIP.
missing = [
    name for name in INCLUDE_FILES
    if not (ROOT / name).is_file()
]

if missing:
    print("ERROR: Required files are missing. No ZIP created.")
    for name in missing:
        print(f"  Missing: {name}")
    sys.exit(1)

large = [
    name for name in INCLUDE_FILES
    if (ROOT / name).stat().st_size > MAX_FILE_BYTES
]

if large:
    print("ERROR: Files exceed the size limit. No ZIP created.")
    for name in large:
        print(f"  Too large: {name}")
    sys.exit(1)

# Create the archive only after every check passes.
with zipfile.ZipFile(
    OUT,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as archive:
    for name in INCLUDE_FILES:
        archive.write(ROOT / name, name)

# Verify archive integrity.
with zipfile.ZipFile(OUT, "r") as archive:
    bad_file = archive.testzip()

if bad_file:
    OUT.unlink(missing_ok=True)
    sys.exit(f"ERROR: ZIP integrity check failed at {bad_file}")

print(f"\nCreated: {OUT}")
print(f"Files included: {len(INCLUDE_FILES)}")
print(f"Archive size: {OUT.stat().st_size / 1024:.1f} KB")

print("\nIncluded files:")
for name in INCLUDE_FILES:
    print(f"  {name}")

print("\nZIP integrity: PASSED")
print("Raw datasets and ball-level pressure data are excluded.")
