"""
Pre-build team JSON files from Google Drive (or local CSVs).

Run once locally before deploying:
    python prebuild.py

Outputs: prebuilt/{TEAM_TAG}.json  (one file per team, ~10-50 KB each)
These files are committed to git so Render serves them instantly.
"""

import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Make sure we can import app modules
sys.path.insert(0, os.path.dirname(__file__))

from app import get_team_index, load_team_data, PREBUILT_DIR

os.makedirs(PREBUILT_DIR, exist_ok=True)

def build_team(tag):
    out_path = os.path.join(PREBUILT_DIR, f'{tag}.json')
    if os.path.exists(out_path):
        return tag, 'skip'
    try:
        pitchers = load_team_data(tag)
        if not pitchers:
            return tag, 'empty'
        with open(out_path, 'w') as f:
            json.dump(pitchers, f, separators=(',', ':'))
        return tag, f'{len(pitchers)} pitchers'
    except Exception as e:
        return tag, f'ERROR: {e}'

if __name__ == '__main__':
    index = get_team_index()
    tags = sorted(index.keys())
    print(f"Building {len(tags)} teams into {PREBUILT_DIR}/")
    print("(skipping teams that already have a .json file)\n")

    t0 = time.time()
    workers = int(os.environ.get('PREBUILD_WORKERS', '4'))
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(build_team, tag): tag for tag in tags}
        for fut in as_completed(futures):
            tag, result = fut.result()
            done += 1
            print(f"[{done}/{len(tags)}] {tag}: {result}")

    elapsed = time.time() - t0
    files = [f for f in os.listdir(PREBUILT_DIR) if f.endswith('.json')]
    total_mb = sum(os.path.getsize(os.path.join(PREBUILT_DIR, f)) for f in files) / 1e6
    print(f"\nDone in {elapsed:.0f}s — {len(files)} files, {total_mb:.1f} MB total")
