"""Rebuild splits.json (lost): chip-level 80/10/10 train/val/test.

Protocol (same as the original data_loader split): each chip is assigned a
stratum = (region_id, globally-rarest crop present on the chip), then a
stratified shuffle split at CHIP level — no field/pixel leakage across
splits. Retries seeds until every 8-scheme class appears in all three
splits.
"""
import json
import os
import random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AGRI_DATA_DIR", os.path.join(ROOT, "Extracted_dataset_gee"))

FRACS = (0.80, 0.10, 0.10)
BASE_SEED = 42


def build(meta):
    field_counts = {int(k): v for k, v in meta["field_counts"].items()}
    crop_to_8 = {int(k): v for k, v in meta["crop_to_8"].items()}
    rarity = sorted(field_counts, key=lambda c: field_counts[c])  # rarest first

    def rarest(crops):
        for c in rarity:
            if c in crops:
                return c
        return -1

    strata = defaultdict(list)
    for name, info in meta["chip_index"].items():
        if info["n_fields"] == 0:
            strata[(info["region_id"], -1)].append(name)
        else:
            strata[(info["region_id"], rarest(info["crops"]))].append(name)

    def classes_of(split, idx):
        out = set()
        for ch in split:
            for c in idx[ch]["crops"]:
                out.add(crop_to_8.get(c, 7))
        return out

    idx = meta["chip_index"]
    for attempt in range(200):
        rng = random.Random(BASE_SEED + attempt)
        splits = {"train": [], "val": [], "test": []}
        for key in sorted(strata):
            chips = sorted(strata[key])
            rng.shuffle(chips)
            n = len(chips)
            n_val = round(n * FRACS[1])
            n_test = round(n * FRACS[2])
            # tiny strata (1-2 chips) go to train so rare classes keep volume
            if n <= 2:
                splits["train"] += chips
                continue
            splits["val"] += chips[:n_val]
            splits["test"] += chips[n_val:n_val + n_test]
            splits["train"] += chips[n_val + n_test:]
        want = set(range(8))
        if all(classes_of(splits[s], idx) >= want for s in splits):
            print(f"seed {BASE_SEED + attempt}: all 8 classes in every split")
            break
    else:
        raise RuntimeError("no seed satisfied class coverage")

    for s in splits:
        splits[s] = sorted(splits[s])
        nf = sum(idx[ch]["n_fields"] for ch in splits[s])
        print(f"{s}: {len(splits[s])} chips / {nf} fields")
    out = os.path.join(DATA_DIR, "splits.json")
    with open(out, "w") as f:
        json.dump({"seed": BASE_SEED + attempt, "fracs": FRACS, **splits}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        meta = json.load(f)
    build(meta)
