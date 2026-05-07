"""把 phase{N}_verified_v4.json 的 verified_seed 合进 phase456_seed_manifest.json.

确保 evaluate_bc / build_dataset 等通过 load_seed_manifest 读到的 seeds 是
v4 的 verified_seed (保证 box-target 配对友好).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase-jsons", nargs="+", required=True,
                   help="phase{N}_verified_v4.json paths")
    p.add_argument("--manifest", default=str(ROOT / "assets" / "maps" / "phase456_seed_manifest.json"))
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.manifest):
        try:
            with open(args.manifest, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception:
            manifest = {"phases": {}}
    else:
        manifest = {"phases": {}}
    if "phases" not in manifest:
        manifest["phases"] = {}

    if not args.no_backup and os.path.exists(args.manifest):
        backup = args.manifest + ".bak"
        if not os.path.exists(backup):
            shutil.copy(args.manifest, backup)
            print(f"[backup] → {backup}")

    total_added = 0
    total_updated = 0
    for jpath in args.phase_jsons:
        with open(jpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        ph = data["phase"]
        ph_key = f"phase{ph}"
        ph_dict = manifest["phases"].setdefault(ph_key, {})
        for r in data["results"]:
            if r.get("status") != "ok":
                continue
            seed = r.get("verified_seed")
            if seed is None:
                continue
            map_name = r["map"]
            existing = ph_dict.get(map_name, {})
            old_seeds = existing.get("verified_seeds", [])
            new_seeds = [seed] + [s for s in old_seeds if s != seed]
            if existing:
                total_updated += 1
            else:
                total_added += 1
            ph_dict[map_name] = {
                **existing,
                "verified_seeds": new_seeds,
                "_source": os.path.basename(jpath),
            }

    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"[merged] added={total_added} updated={total_updated} → {args.manifest}")


if __name__ == "__main__":
    main()
