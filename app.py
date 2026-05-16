import os
import io
import math
import csv
import json
import secrets
import pickle
import time
from collections import defaultdict
from flask import Flask, render_template, abort, jsonify
import pandas as pd
import numpy as np
import requests as _requests
from pitch_classifier import run_pitch_classification

app = Flask(__name__)

# ── Token: env var takes priority (for deployment), falls back to local file ──
SECRET_TOKEN = os.environ.get('SECRET_TOKEN')
if not SECRET_TOKEN:
    TOKEN_FILE = os.path.join(os.path.dirname(__file__), '.secret_token')
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            SECRET_TOKEN = f.read().strip()
    else:
        SECRET_TOKEN = secrets.token_urlsafe(20)
        with open(TOKEN_FILE, 'w') as f:
            f.write(SECRET_TOKEN)

# ── Data directory: env var takes priority (for deployment) ──────────────────
DATA_DIR = os.environ.get(
    'DATA_DIR',
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '2026')
)
TEAM_INDEX_PKL = os.path.join(os.path.dirname(__file__), 'team_index.pkl')
TEAM_INDEX_MAX_AGE = 86400  # 1 day in seconds

# ── Google Drive mode ─────────────────────────────────────────────────────────
# If gdrive_manifest.json exists, use Google Drive instead of local DATA_DIR.
# Build the manifest once with build_gdrive_manifest.py then commit it to GitHub.
GDRIVE_MANIFEST_FILE = os.path.join(os.path.dirname(__file__), 'gdrive_manifest.json')
_gdrive_manifest = None  # loaded lazily

def _load_gdrive_manifest():
    global _gdrive_manifest
    if _gdrive_manifest is None and os.path.exists(GDRIVE_MANIFEST_FILE):
        with open(GDRIVE_MANIFEST_FILE) as f:
            _gdrive_manifest = json.load(f)
    return _gdrive_manifest

def _fetch_drive_csv(file_id):
    """Download a publicly-shared CSV from Google Drive and return a StringIO."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = _requests.get(url, timeout=60)
    r.raise_for_status()
    # Google sometimes redirects large files to a warning page; detect it
    if 'text/html' in r.headers.get('Content-Type', '') and len(r.content) < 50_000:
        # Try the confirmed-download URL
        url2 = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        r = _requests.get(url2, timeout=60)
        r.raise_for_status()
    return io.StringIO(r.content.decode('utf-8-sig'))

PITCH_COLORS = {
    'Fastball':    '#FF2222',
    'Slider':      '#FFEE00',
    'ChangeUp':    '#32CD32',
    'Cutter':      '#8B4513',
    'Sinker':      '#FF8800',
    'Curveball':   '#3366FF',
    'Splitter':    '#00CCCC',
    'Knuckleball': '#9922CC',
    'Sweeper':     '#FFD700',
}

# In-memory cache: team_tag -> pitcher dict
TEAM_CACHE = {}


def sf(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def solve_t_at_y(y0, vy0, ay0, target_y):
    """Find the smaller positive t where y(t) = target_y using constant-acceleration model."""
    a = 0.5 * ay0
    b = vy0
    c = y0 - target_y
    if abs(a) < 1e-9:
        return -c / b if abs(b) > 1e-9 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    pos = [t for t in (t1, t2) if t > 0]
    return min(pos) if pos else None


def compute_trajectory(x0, y0, z0, vx0, vy0, vz0, ax0, ay0, az0, n=30):
    """
    Compute pitch path from release (~54 ft) through the plate and 1.5 ft past it
    so the tube visually passes through the zone box rather than stopping at its face.
    Returns list of [x, y_dist, z, frac] where frac=0 is release, frac=1 is plate, >1 is past plate.
    y_dist is signed distance from plate (negative = past plate / behind).
    """
    t_plate = solve_t_at_y(y0, vy0, ay0, 0.0)
    t_catch = solve_t_at_y(y0, vy0, ay0, -1.5)
    if t_plate is None or t_catch is None or t_plate <= 0:
        return []

    release_y = 54.0
    a = 0.5 * ay0
    b = vy0
    c_rel = y0 - release_y
    disc2 = b * b - 4 * a * c_rel
    t_release = -0.035
    if disc2 >= 0:
        sqrt_disc2 = math.sqrt(disc2)
        tr1 = (-b + sqrt_disc2) / (2 * a)
        tr2 = (-b - sqrt_disc2) / (2 * a)
        neg_roots = [t for t in (tr1, tr2) if t < 0]
        if neg_roots:
            t_release = max(neg_roots)

    pts = []
    for i in range(n):
        frac = i / (n - 1)
        t = t_release + (t_catch - t_release) * frac
        x = x0 + vx0 * t + 0.5 * ax0 * t * t
        y_dist = y0 + vy0 * t + 0.5 * ay0 * t * t
        z = z0 + vz0 * t + 0.5 * az0 * t * t
        frac_plate = (t - t_release) / (t_plate - t_release) if t_plate != t_release else frac
        pts.append([round(x, 4), round(y_dist, 4), round(z, 4), round(frac_plate, 4)])
    return pts


def avg_field(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    return sum(vals) / len(vals) if vals else None


def compute_damage_zones(rows):
    """
    Build a 12×12 exit-velocity heat map over the plate area using actual contact data.
    """
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
        col = int((px - X_MIN) / (X_MAX - X_MIN) * NX)
        row_i = int((Z_MAX - pz) / (Z_MAX - Z_MIN) * NZ)
        if 0 <= col < NX and 0 <= row_i < NZ:
            bucket_sum[row_i][col] += ev
            bucket_cnt[row_i][col] += 1

    all_vals = [bucket_sum[r][c] / bucket_cnt[r][c]
                for r in range(NZ) for c in range(NX) if bucket_cnt[r][c] > 0]
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


def load_damage_zones():
    """Load pre-built damage zones JSON, or fall back to Illinois CSV."""
    json_file = os.path.join(os.path.dirname(__file__), 'damage_zones.json')
    if os.path.exists(json_file):
        with open(json_file) as f:
            return json.load(f)
    # Fallback: compute from bundled Illinois CSV
    data_file = os.path.join(os.path.dirname(__file__), 'Illinois - Sheet1.csv')
    if not os.path.exists(data_file):
        # No data at all — return a flat/neutral grid
        NX, NZ = 12, 12
        return {
            'grid': [[0.5] * NX for _ in range(NZ)],
            'nx': NX, 'nz': NZ,
            'x_min': -1.5, 'x_max': 1.5,
            'z_min': 0.5,  'z_max': 5.0,
            'ev_min': 80.0, 'ev_max': 100.0,
            'n_contacts': 0,
        }
    with open(data_file, encoding='utf-8-sig') as f:
        all_rows = list(csv.DictReader(f))
    return compute_damage_zones(all_rows)


def build_team_index():
    """Scan all CSVs in DATA_DIR, index each file under every team that appears in it."""
    print("Building team index from CSVs...")
    index = defaultdict(set)
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    for i, fname in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}...")
        fpath = os.path.join(DATA_DIR, fname)
        try:
            with open(fpath, encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    team = row.get('PitcherTeam', '').strip()
                    if team:
                        index[team].add(fpath)
        except Exception:
            continue
    result = {k: list(v) for k, v in index.items()}
    with open(TEAM_INDEX_PKL, 'wb') as f:
        pickle.dump(result, f)
    print(f"Team index built: {len(result)} teams, {sum(len(v) for v in result.values())} file-team mappings")
    return result


def get_team_index():
    """Return team index — from gdrive manifest if available, else local scan."""
    manifest = _load_gdrive_manifest()
    if manifest is not None:
        # Drive mode: index maps team_tag → list of {name, id} dicts
        return manifest
    # Local mode
    if not os.path.isdir(DATA_DIR):
        return {}
    if os.path.exists(TEAM_INDEX_PKL):
        age = time.time() - os.path.getmtime(TEAM_INDEX_PKL)
        if age < TEAM_INDEX_MAX_AGE:
            with open(TEAM_INDEX_PKL, 'rb') as f:
                return pickle.load(f)
    return build_team_index()


def load_team_data(team_tag):
    """Load all CSVs for a team, run classification, return pitcher dict."""
    if team_tag in TEAM_CACHE:
        return TEAM_CACHE[team_tag]

    index = get_team_index()
    files = index.get(team_tag, [])
    if not files:
        return {}

    # Required columns
    needed = [
        'Pitcher', 'PitcherId', 'PitcherThrows', 'PitcherTeam',
        'TaggedPitchType', 'HorzBreak', 'InducedVertBreak', 'RelSpeed', 'SpinRate',
        'PlateLocSide', 'PlateLocHeight',
        'x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0', 'ax0', 'ay0', 'az0',
        'ExitSpeed', 'Extension', 'RelHeight', 'RelSide', 'BatterSide',
    ]

    drive_mode = _load_gdrive_manifest() is not None

    dfs = []
    for entry in files:
        try:
            if drive_mode:
                # entry is {'name': ..., 'id': ...}
                csv_src = _fetch_drive_csv(entry['id'])
                df_chunk = pd.read_csv(csv_src, low_memory=False)
            else:
                # entry is a local file path string
                df_chunk = pd.read_csv(entry, encoding='utf-8-sig', low_memory=False)
            # Filter to this team only (file may have multiple teams)
            if 'PitcherTeam' in df_chunk.columns:
                df_chunk = df_chunk[df_chunk['PitcherTeam'] == team_tag]
            # Keep only needed columns that exist
            cols = [c for c in needed if c in df_chunk.columns]
            dfs.append(df_chunk[cols])
        except Exception as e:
            print(f"  Error loading {'Drive:'+entry['id'] if drive_mode else entry}: {e}")
            continue

    if not dfs:
        return {}

    df = pd.concat(dfs, ignore_index=True)

    # Drop rows without key physics columns
    for col in ['x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0', 'ax0', 'ay0', 'az0']:
        if col in df.columns:
            df = df.dropna(subset=[col])

    # Filter out short-distance releases
    if 'y0' in df.columns:
        df = df[pd.to_numeric(df['y0'], errors='coerce') >= 10]

    if df.empty:
        return {}

    # Run pitch classification
    try:
        df = run_pitch_classification(df)
    except Exception as e:
        print(f"Classification error for {team_tag}: {e}")

    # Build pitcher profiles
    pitchers = {}
    groups = defaultdict(list)

    for _, row in df.iterrows():
        pitcher = str(row.get('Pitcher', '')).strip()
        pitcher_id = str(row.get('PitcherId', '')).strip()
        pitch_type = str(row.get('TaggedPitchType', '')).strip()
        throws = str(row.get('PitcherThrows', 'Right')).strip()
        team = str(row.get('PitcherTeam', team_tag)).strip()

        if not pitcher or pitch_type in ('Unknown', '', 'nan'):
            continue

        x0 = sf(row.get('x0')); y0 = sf(row.get('y0')); z0 = sf(row.get('z0'))
        vx0 = sf(row.get('vx0')); vy0 = sf(row.get('vy0')); vz0 = sf(row.get('vz0'))
        ax0 = sf(row.get('ax0')); ay0 = sf(row.get('ay0')); az0 = sf(row.get('az0'))

        if None in (x0, y0, z0, vx0, vy0, vz0, ax0, ay0, az0):
            continue

        record = {
            'pitcher': pitcher,
            'pitcher_id': pitcher_id,
            'throws': throws,
            'team': team,
            'pitch_type': pitch_type,
            'x0': x0, 'y0': y0, 'z0': z0,
            'vx0': vx0, 'vy0': vy0, 'vz0': vz0,
            'ax0': ax0, 'ay0': ay0, 'az0': az0,
            'plate_x': sf(row.get('PlateLocSide')),
            'plate_z': sf(row.get('PlateLocHeight')),
            'rel_height': sf(row.get('RelHeight')),
            'rel_side': sf(row.get('RelSide')),
            'extension': sf(row.get('Extension')),
            'velo': sf(row.get('RelSpeed')),
            'ivb': sf(row.get('InducedVertBreak')),
            'hb': sf(row.get('HorzBreak')),
            'spin': sf(row.get('SpinRate')),
            'batter_side': str(row.get('BatterSide', 'Right')),
            'ExitSpeed': sf(row.get('ExitSpeed')),
        }
        key = (pitcher, pitcher_id, pitch_type, throws, team)
        groups[key].append(record)

    for (pitcher, pitcher_id, pitch_type, throws, team), grp in groups.items():
        if len(grp) < 5:
            continue

        a_x0 = avg_field(grp, 'x0'); a_y0 = avg_field(grp, 'y0'); a_z0 = avg_field(grp, 'z0')
        a_vx0 = avg_field(grp, 'vx0'); a_vy0 = avg_field(grp, 'vy0'); a_vz0 = avg_field(grp, 'vz0')
        a_ax0 = avg_field(grp, 'ax0'); a_ay0 = avg_field(grp, 'ay0'); a_az0 = avg_field(grp, 'az0')

        if None in (a_x0, a_y0, a_z0, a_vx0, a_vy0, a_vz0, a_ax0, a_ay0, a_az0):
            continue

        traj = compute_trajectory(a_x0, a_y0, a_z0, a_vx0, a_vy0, a_vz0, a_ax0, a_ay0, a_az0)
        if not traj:
            continue

        pid = pitcher_id if pitcher_id and pitcher_id != 'nan' else pitcher.replace(', ', '_').replace(' ', '_')

        profile = {
            'pitch_type': pitch_type,
            'throws': throws,
            'count': len(grp),
            'velo': round(avg_field(grp, 'velo') or 0, 1),
            'ivb': round(avg_field(grp, 'ivb') or 0, 1),
            'hb': round(avg_field(grp, 'hb') or 0, 1),
            'spin': round(avg_field(grp, 'spin') or 0, 0),
            'plate_x': round(avg_field(grp, 'plate_x') or 0, 3),
            'plate_z': round(avg_field(grp, 'plate_z') or 0, 3),
            'rel_height': round(avg_field(grp, 'rel_height') or 0, 3),
            'rel_side': round(avg_field(grp, 'rel_side') or 0, 3),
            'extension': round(avg_field(grp, 'extension') or 0, 2),
            'trajectory': traj,
            'color': PITCH_COLORS.get(pitch_type, '#AAAAAA'),
        }

        if pid not in pitchers:
            pitchers[pid] = {
                'name': pitcher,
                'id': pid,
                'team': team,
                'throws': throws,
                'pitches': [],
            }
        pitchers[pid]['pitches'].append(profile)

    for p in pitchers.values():
        p['pitches'].sort(key=lambda x: -x['count'])

    TEAM_CACHE[team_tag] = pitchers
    return pitchers


# ─── STARTUP ──────────────────────────────────────────────────────────────────
print("Loading damage zones from Illinois CSV...")
DAMAGE_ZONES = load_damage_zones()
print(f"Damage zones loaded: {DAMAGE_ZONES['n_contacts']} contact events, "
      f"EV range {DAMAGE_ZONES['ev_min']}–{DAMAGE_ZONES['ev_max']} mph")

print("Loading team index...")
TEAM_INDEX = get_team_index()
mode = "Google Drive" if _load_gdrive_manifest() is not None else f"local ({DATA_DIR})"
print(f"Team index ready: {len(TEAM_INDEX)} teams [{mode}]")


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def root():
    from flask import redirect
    return redirect(f'/view/{SECRET_TOKEN}')


@app.route('/view/<token>')
def view(token):
    if token != SECRET_TOKEN:
        abort(404)
    return render_template(
        'index.html',
        damage_zones=json.dumps(DAMAGE_ZONES),
        token=SECRET_TOKEN,
    )


@app.route('/api/<token>/teams')
def api_teams(token):
    if token != SECRET_TOKEN:
        abort(404)
    result = []
    for tag, files in TEAM_INDEX.items():
        result.append({'tag': tag, 'file_count': len(files)})
    result.sort(key=lambda x: x['tag'])
    return jsonify(result)


@app.route('/api/<token>/team/<tag>')
def api_team(token, tag):
    if token != SECRET_TOKEN:
        abort(404)
    pitchers = load_team_data(tag)
    return jsonify({'pitchers': pitchers})


if __name__ == '__main__':
    print(f"\n{'=' * 60}")
    print("  Pitch Visualizer — Private Link")
    print(f"{'=' * 60}")
    print(f"\n  Local:   http://localhost:5001/view/{SECRET_TOKEN}")
    print(f"\n  For remote access: ngrok http 5001")
    print(f"{'=' * 60}\n")
    app.run(host='0.0.0.0', port=5001, debug=False)
