"""Rebuild Extracted_dataset_gee/metadata.json (lost with the scripts).

Scans every chip npz for crop ids / fields, writes the 13-class map
(AgriFieldNet crop ids) plus the 8-class consolidated scheme used by the
final classifier.
"""
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AGRI_DATA_DIR", os.path.join(ROOT, "Extracted_dataset_gee"))
CHIP_DIR = os.path.join(DATA_DIR, "chips")

# AgriFieldNet raw crop ids
CROP_NAMES = {
    1: "Wheat", 2: "Mustard", 3: "Lentil", 4: "No crop/Fallow", 5: "Green pea",
    6: "Sugarcane", 8: "Garlic", 9: "Maize", 13: "Gram", 14: "Coriander",
    15: "Potato", 16: "Bersem", 36: "Rice",
}

# 8-class consolidated scheme (the deliverable): everything not named -> Other
SCHEME_8 = ["Wheat", "Mustard", "Lentil", "No crop/Fallow", "Sugarcane", "Maize", "Rice", "Other"]
CROP_TO_8 = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 9: 5, 36: 6}  # rest -> 7 (Other)

REGION_NAMES = {0: "UP", 1: "BIHAR", 2: "ODISHA", 3: "RAJASTHAN"}


def scan():
    chips = sorted(glob.glob(os.path.join(CHIP_DIR, "*.npz")))
    print(f"{len(chips)} chips")
    field_counts = Counter()          # crop_id -> n fields (global)
    px_counts = Counter()
    region_chips = Counter()
    chip_index = {}                   # chip name -> {region, crops present, n_fields}
    for i, p in enumerate(chips):
        name = os.path.splitext(os.path.basename(p))[0]
        d = np.load(p)
        crop = d["crop_id"]
        fid = d["field_id"]
        lm = d["label_mask"].astype(bool)
        region = int(d["region_id"])
        region_chips[region] += 1
        crops_here = []
        for f in np.unique(fid[lm & (fid > 0)]):
            sel = lm & (fid == f)
            cids, cnt = np.unique(crop[sel], return_counts=True)
            cid = int(cids[np.argmax(cnt)])
            if cid == 0:
                continue
            field_counts[cid] += 1
            px_counts[cid] += int(sel.sum())
            crops_here.append(cid)
        chip_index[name] = {
            "region_id": region,
            "crops": sorted(set(crops_here)),
            "n_fields": len(crops_here),
        }
        if (i + 1) % 200 == 0:
            print(f"  scanned {i+1}")

    meta = {
        "crop_names": {str(k): v for k, v in CROP_NAMES.items()},
        "scheme_8": SCHEME_8,
        "crop_to_8": {str(k): v for k, v in CROP_TO_8.items()},
        "other_index": 7,
        "region_names": {str(k): v for k, v in REGION_NAMES.items()},
        "n_chips": len(chips),
        "region_chips": {str(k): v for k, v in region_chips.items()},
        "field_counts": {str(k): v for k, v in field_counts.items()},
        "pixel_counts": {str(k): v for k, v in px_counts.items()},
        "chip_index": chip_index,
    }
    out = os.path.join(DATA_DIR, "metadata.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out}")
    print("\nfields per crop:")
    for cid, n in field_counts.most_common():
        print(f"  {CROP_NAMES.get(cid, cid):>16}: {n}")
    print("\nchips per region:", {REGION_NAMES[k]: v for k, v in sorted(region_chips.items())})


if __name__ == "__main__":
    scan()
