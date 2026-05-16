#!/usr/bin/env python3
"""
Build damage_zones.json from all TrackMan CSVs.
Run this locally whenever you want to refresh the heat map, then commit the result.

Usage:
  python3 build_damage_zones.py                          # uses default data paths
  python3 build_damage_zones.py --data-dirs /path/2025 /path/2026
"""

import os
import csv
import json
import math
import argparse
import sys


def sf(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def compute_damage_zones(rows):
    NX, NZ = 12, 12
    X_MIN, X_MAX = -1.5, 1.5
    Z_MIN, Z_MAX = 0.5, 5.0

    bucket_sum = [[0.0] * NX for _ in range(NZ)]
    bucket_cnt = [[0]   * NX for _ in range(NZ)]

    for r in rows:
        ev = sf(r.get('ExitSpeed'))
        px = sf(r.get('PlateLocSide'))
        pz = sf(r.get('PlateLocHeight'))
        if ev is None or px is None or pz is None:
            continue
        if ev < 50:
            continue
        col   = int((px - X_MIN) / (X_MAX - X_MIN) * NX)
        row_i = int((Z_MAX - pz) / (Z_MAX - Z_MIN) * NZ)
        if 0 <= col < NX and 0 <= row_i < NZ:
            bucket_sum[row_i][col] += ev
            bucket_cnt[row_i][col] += 1

    all_vals = [
        bucket_sum[r][c] / bucket_cnt[r][c]
        for r in range(NZ) for c in range(NX)
        if bucket_cnt[r][c] > 0
    ]
    global_mean = sum(all_vals) / len(all_vals) if all_vals else 80.0

    grid_raw = []
    for r in range(NZ):
        row_vals = []
        for c in range(NX):
            if bucket_cnt[r][c] >= 3:
                row_vals.append(bucket_sum[r][c] / bucket_cnt[r][c])
            else:
                row_vals.append(None)
        grid_raw.append(row_vals)

    def fill_none(grid):
        for r in range(NZ):
            for c in range(NX):
                if grid[r][c] is None:
                    near = []
                    for dr in range(-3, 4):
                        for dc in range(-3, 4):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < NZ and 0 <= nc < NX and grid[nr][nc] is not None:
                                d = math.sqrt(dr * dr + dc * dc)
                                near.append((d, grid[nr][nc]))
                    if near:
                        near.sort()
                        grid[r][c] = near[0][1]
                    else:
                        grid[r][c] = global_mean
        return grid

    grid_filled = fill_none(grid_raw)
    ev_min = min(grid_filled[r][c] for r in range(NZ) for c in range(NX))
    ev_max = max(grid_filled[r][c] for r in range(NZ) for c in range(NX))
    ev_range = ev_max - ev_min if ev_max > ev_min else 1.0

    grid_norm = [
        [round((grid_filled[r][c] - ev_min) / ev_range, 4) for c in range(NX)]
        for r in range(NZ)
    ]

    return {
        'grid': grid_norm,
        'nx': NX, 'nz': NZ,
        'x_min': X_MIN, 'x_max': X_MAX,
        'z_min': Z_MIN, 'z_max': Z_MAX,
        'ev_min': round(ev_min, 1),
        'ev_max': round(ev_max, 1),
        'n_contacts': sum(bucket_cnt[r][c] for r in range(NZ) for c in range(NX)),
    }


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dirs = [
        os.path.join(base, 'data', '2025'),
        os.path.join(base, 'data', '2026'),
    ]

    parser = argparse.ArgumentParser(description='Build damage_zones.json from TrackMan CSVs')
    parser.add_argument(
        '--data-dirs', nargs='+', default=None,
        help='One or more directories containing TrackMan CSVs (default: ../data/2025 and ../data/2026)',
    )
    parser.add_argument(
        '--out',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'damage_zones.json'),
        help='Output path (default: damage_zones.json next to this script)',
    )
    args = parser.parse_args()

    data_dirs = args.data_dirs or [d for d in default_dirs if os.path.isdir(d)]
    if not data_dirs:
        sys.exit("No data directories found. Pass --data-dirs to specify them.")

    print(f"Reading from: {data_dirs}")

    all_rows = []
    for d in data_dirs:
        files = [f for f in os.listdir(d) if f.endswith('.csv')]
        print(f"  {d}: {len(files)} files")
        for i, fname in enumerate(files, 1):
            if i % 1000 == 0:
                print(f"    {i}/{len(files)}...")
            fpath = os.path.join(d, fname)
            try:
                with open(fpath, encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ev = row.get('ExitSpeed', '')
                        px = row.get('PlateLocSide', '')
                        pz = row.get('PlateLocHeight', '')
                        if ev and px and pz:
                            all_rows.append(row)
            except Exception:
                continue

    print(f"\nTotal contact rows collected: {len(all_rows):,}")

    if not all_rows:
        sys.exit("No rows with ExitSpeed/PlateLocSide/PlateLocHeight found.")

    print("Computing damage zones...")
    result = compute_damage_zones(all_rows)

    with open(args.out, 'w') as f:
        json.dump(result, f)

    print(f"\n✓ Written to {args.out}")
    print(f"  {result['n_contacts']:,} contact events")
    print(f"  EV range: {result['ev_min']}–{result['ev_max']} mph")
    print("\nNext steps:")
    print("  git add damage_zones.json build_damage_zones.py")
    print("  git commit -m 'Add full damage zones from 2025+2026 data'")
    print("  git push")


if __name__ == '__main__':
    main()
