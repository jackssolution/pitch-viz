#!/usr/bin/env python3
"""
Build gdrive_manifest.json — run this ONCE after uploading your data to Google Drive.

What it does:
  1. Lists all CSV files in your Google Drive folder via the Drive API
  2. Reads team names from your LOCAL copies of those same CSVs
  3. Writes gdrive_manifest.json  (commit this file to GitHub)

After that, the app reads from Google Drive instead of local files.

─── Setup ────────────────────────────────────────────────────────────────────
1. Go to https://console.cloud.google.com/
2. Create a project (or pick an existing one)
3. Enable "Google Drive API"  (APIs & Services → Enable APIs → search Drive)
4. Create an API key  (APIs & Services → Credentials → + Create Credentials → API key)
   • Restrict it to "Google Drive API" only
5. Share your Google Drive data folder: right-click → Share → "Anyone with the link" → Viewer
6. Copy the folder ID from the URL:
     https://drive.google.com/drive/folders/THIS_PART_IS_THE_FOLDER_ID

─── Run ──────────────────────────────────────────────────────────────────────
  pip install requests
  python3 build_gdrive_manifest.py \\
      --folder-id YOUR_FOLDER_ID \\
      --api-key   YOUR_API_KEY

  # If your local data is not at the default path, add:
      --data-dir  /path/to/your/data/2026
"""

import os
import csv
import json
import argparse
import sys
from collections import defaultdict

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests")


def list_drive_files(folder_id, api_key):
    """List all files in a Google Drive folder (handles pagination)."""
    files = []
    page_token = None
    print(f"Listing files in Drive folder {folder_id!r}...")
    while True:
        params = {
            'q': f"'{folder_id}' in parents and trashed=false",
            'fields': 'nextPageToken, files(id, name)',
            'pageSize': 1000,
            'key': api_key,
        }
        if page_token:
            params['pageToken'] = page_token
        r = requests.get(
            'https://www.googleapis.com/drive/v3/files',
            params=params,
            timeout=30,
        )
        if not r.ok:
            print(f"\nDrive API error {r.status_code}: {r.text}")
            print("\nCommon causes:")
            print("  • API key is wrong or not enabled for Drive API")
            print("  • Folder is not shared publicly")
            sys.exit(1)
        data = r.json()
        batch = data.get('files', [])
        files.extend(batch)
        print(f"  ...found {len(files)} files so far")
        page_token = data.get('nextPageToken')
        if not page_token:
            break
    return files


def main():
    parser = argparse.ArgumentParser(description='Build Google Drive manifest for pitch_viz')
    parser.add_argument('--folder-id', required=True, help='Google Drive folder ID')
    parser.add_argument('--api-key',   required=True, help='Google API key')
    parser.add_argument(
        '--data-dir',
        default=None,
        help='Local path to your CSV folder (default: ../data/2026 relative to this script)',
    )
    parser.add_argument(
        '--out',
        default=os.path.join(os.path.dirname(__file__), 'gdrive_manifest.json'),
        help='Output path for the manifest JSON',
    )
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', '2026',
    )

    if not os.path.isdir(data_dir):
        sys.exit(f"Data directory not found: {data_dir}\nPass --data-dir to specify it.")

    # Step 1: list Drive files
    drive_files = list_drive_files(args.folder_id, args.api_key)
    drive_map = {f['name']: f['id'] for f in drive_files}
    print(f"Total Drive files: {len(drive_files)}")

    # Step 2: read team names from local CSVs
    local_csvs = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"\nReading team names from {len(local_csvs)} local CSVs...")

    manifest = defaultdict(list)
    missing_in_drive = []

    for i, fname in enumerate(local_csvs, 1):
        if i % 500 == 0:
            print(f"  {i}/{len(local_csvs)}...")
        fpath = os.path.join(data_dir, fname)
        try:
            with open(fpath, encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
                if row is None:
                    continue
                team = row.get('PitcherTeam', '').strip()
                if not team:
                    continue
        except Exception as e:
            print(f"  Warning: could not read {fname}: {e}")
            continue

        file_id = drive_map.get(fname)
        if file_id:
            manifest[team].append({'name': fname, 'id': file_id})
        else:
            missing_in_drive.append(fname)

    # Step 3: write manifest
    out = dict(manifest)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    total_files = sum(len(v) for v in out.values())
    print(f"\n✓ Manifest written to {args.out}")
    print(f"  {len(out)} teams, {total_files} files mapped")

    if missing_in_drive:
        print(f"\n⚠  {len(missing_in_drive)} local files were NOT found in Drive "
              f"(not uploaded yet):")
        for name in missing_in_drive[:10]:
            print(f"    {name}")
        if len(missing_in_drive) > 10:
            print(f"    ... and {len(missing_in_drive) - 10} more")

    print("\nNext steps:")
    print("  1. git add gdrive_manifest.json && git commit -m 'Add Drive manifest'")
    print("  2. git push")
    print("  3. The app will now read data from Google Drive automatically.")


if __name__ == '__main__':
    main()
