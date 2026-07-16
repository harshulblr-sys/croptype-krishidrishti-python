"""
PyTorch Data Loader for the TempCNN + U-Net Hybrid
=================================================================
Feeds the combined UP+Bihar dataset (Extracted_dataset_combined/) to the hybrid:

  Branch 1  awifs   [T_a, C_a+1, H, W]   NDVI, NDMI (+ validity mask channel)
  Branch 2  s1_asc  [T_s, C_s+1, H, W]   VV, VH, VV/VH (+ validity mask channel)
  Static    s2      [4, H, W]            B02, B03, B04, B08  -> U-Net encoder
  region    scalar  (0=UP, 1=BIHAR)      region feature (embed / broadcast in model)
  label     [H, W]  long                 crop index (ignore_index=-100 for unlabelled)
  label_mask[H, W]  float                1 = labelled pixel (for masked loss)

Design points
-------------
* Leakage-free split at the CHIP level (one .npz == one chip); grouped so all
  pixels of a field stay together. Stratified by each chip's dominant class so
  rare classes (Rice, Others) appear in every split. Split saved to splits.json.
* Per-channel normalisation stats are FIT ON THE TRAIN SPLIT ONLY (valid pixels
  only) and cached to norm_stats.json -> no leakage, reproducible.
* Missing temporal months are handled: after normalisation the masked positions
  are zeroed AND the validity mask is appended as an extra channel so the TempCNN
  can tell "missing" from a real zero.
* Labels are remapped through metadata's class_to_index (tail already merged into
  "Others"); crop_id==0 -> ignore_index.
* D4 augmentation (flips + 90 deg rotations) on the train split — cheap 8x boost,
  valid for nadir imagery, critical for this small dataset.
"""

import os
import re
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

BASE = r"C:\Users\harsh\OneDrive\Desktop\Harshul\ISRO_Hackathon"
# Override with AGRI_DATA_DIR to train on a different extraction (e.g. the GEE
# season-matched dataset in Extracted_dataset_gee) without code changes.
DATA_DIR = os.environ.get("AGRI_DATA_DIR",
                          os.path.join(BASE, "Extracted_dataset_combined"))
CHIPS_DIR = os.path.join(DATA_DIR, "chips")
IGNORE_INDEX = -100


# ============================================================
# Metadata + label remapping
# ============================================================
def load_metadata():
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        return json.load(f)


def build_label_lut(meta):
    """crop_id -> training index LUT (0..num_classes-1); unmapped -> IGNORE_INDEX."""
    c2i = {int(k): v for k, v in meta["class_to_index"].items()}
    lut = np.full(max(c2i) + 1, IGNORE_INDEX, dtype=np.int64)
    for c, i in c2i.items():
        lut[c] = i
    return lut


# ============================================================
# Chip-level stratified split (grouped, leakage-free)
# ============================================================
def _index_frequency(meta):
    """Global labelled-field frequency per training index (for split rarity key)."""
    c2i = {int(k): v for k, v in meta["class_to_index"].items()}
    fc = {int(k): v for k, v in meta["field_counts"].items()}
    freq = np.zeros(meta["num_classes"])
    for cid, n in fc.items():
        if cid in c2i:
            freq[c2i[cid]] += n
    return freq


def _rarest_present_index(npz_path, lut, idx_freq):
    """Stratify key = the RAREST class present in the chip. Grouping by this
    (rather than the dominant class) spreads scarce classes across all splits."""
    d = np.load(npz_path)
    crop = d["crop_id"]
    lab = crop[crop > 0]
    if lab.size == 0:
        return -1
    idx = np.unique(lut[lab])
    idx = idx[idx != IGNORE_INDEX]
    return int(idx[np.argmin(idx_freq[idx])]) if idx.size else -1


def _andhra_chips():
    """Region-4 (Andhra) supplementary rice chips — pinned to TRAIN only. They are
    rice-positive-only, and region_id=4 would be a giveaway shortcut, so letting them
    into val/test would inflate metrics without measuring real discrimination."""
    p = os.path.join(DATA_DIR, "andhra_chips.txt")
    return set(l.strip() for l in open(p) if l.strip()) if os.path.exists(p) else set()


def _pin_andhra_train(splits):
    """Force the Andhra chips into train; scrub them from val/test if ever present.
    Applied on every make_splits call so a splits.json cached before Andhra existed
    still gets them in train (and the original eval set stays byte-identical)."""
    ap = _andhra_chips()
    if not ap:
        return splits
    splits["train"] = sorted(set(splits["train"]) | ap)
    splits["val"] = sorted(f for f in splits["val"] if f not in ap)
    splits["test"] = sorted(f for f in splits["test"] if f not in ap)
    return splits


def make_splits(meta, ratios=(0.8, 0.1, 0.1), seed=42, force=False):
    """Stratified chip-level split, cached to splits.json. Andhra (region 4) chips
    are excluded from the stratified eval split and pinned to TRAIN."""
    out = os.path.join(DATA_DIR, "splits.json")
    if os.path.exists(out) and not force:
        with open(out) as f:
            return _pin_andhra_train(json.load(f))

    ap = _andhra_chips()
    lut = build_label_lut(meta)
    idx_freq = _index_frequency(meta)
    files = sorted(f for f in os.listdir(CHIPS_DIR) if f.endswith(".npz") and f not in ap)
    by_class = {}
    for f in files:
        k = _rarest_present_index(os.path.join(CHIPS_DIR, f), lut, idx_freq)
        by_class.setdefault(k, []).append(f)

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for k, group in sorted(by_class.items()):
        g = list(group)
        rng.shuffle(g)
        n = len(g)
        n_tr = max(1, round(n * ratios[0]))
        n_va = round(n * ratios[1])
        # guarantee test gets a chip when the group is big enough
        if n >= 3 and n_tr + n_va >= n:
            n_va = max(0, n - n_tr - 1)
        train += g[:n_tr]
        val += g[n_tr:n_tr + n_va]
        test += g[n_tr + n_va:]

    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test),
              "seed": seed, "ratios": ratios}
    with open(out, "w") as f:                         # persist original-only split
        json.dump(splits, f, indent=1)
    return _pin_andhra_train(splits)


# ============================================================
# Per-channel normalisation (fit on train, valid pixels only)
# ============================================================
def fit_normalization(train_files, meta, force=False):
    out = os.path.join(DATA_DIR, "norm_stats.json")
    if os.path.exists(out) and not force:
        with open(out) as f:
            return json.load(f)

    def accum(key, mask_key, n_ch):
        s = np.zeros(n_ch); ss = np.zeros(n_ch); cnt = np.zeros(n_ch)
        for fn in train_files:
            d = np.load(os.path.join(CHIPS_DIR, fn))
            x = d[key].reshape(-1, n_ch).astype(np.float64)   # [.., C]
            if mask_key:
                m = d[mask_key].reshape(-1).astype(bool)
                x = x[m]
            else:
                x = x[(x != 0).any(1)]
            s += x.sum(0); ss += (x * x).sum(0); cnt += x.shape[0]
        mean = s / np.maximum(cnt, 1)
        std = np.sqrt(np.maximum(ss / np.maximum(cnt, 1) - mean ** 2, 1e-12))
        return mean.tolist(), std.tolist()

    stats = {}
    stats["awifs"] = dict(zip(("mean", "std"), accum("awifs", "awifs_mask", 2)))
    stats["s1_asc"] = dict(zip(("mean", "std"), accum("s1_asc", "s1_asc_mask", 3)))
    stats["s2"] = dict(zip(("mean", "std"), accum("s2", None, 4)))
    with open(out, "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def compute_class_weights(train_files, meta):
    """Square-root inverse-frequency class weights (by labelled-pixel count).

    Plain 1/count gave a ~113:1 spread (Others vs Wheat) and made the model
    over-predict rare classes (e.g. Potato recall .66 / precision .12 in
    hybrid_v1). 1/sqrt(count) compresses that to ~11:1 — rare classes are still
    boosted, but not at the cost of the common ones. Normalised to mean 1 so
    the loss scale stays comparable across runs."""
    lut = build_label_lut(meta)
    counts = np.zeros(meta["num_classes"])
    for fn in train_files:
        d = np.load(os.path.join(CHIPS_DIR, fn))
        crop = d["crop_id"]; lab = crop[crop > 0]
        idx = lut[lab]; idx = idx[idx != IGNORE_INDEX]
        counts += np.bincount(idx, minlength=meta["num_classes"])
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), counts.astype(int)


# ============================================================
# Dataset
# ============================================================
class AgriChips(Dataset):
    def __init__(self, files, meta, norm, augment=False, seed=0, crop_size=None):
        """crop_size: if set (train only), return a crop_size x crop_size window
        CENTERED ON A RANDOM LABELLED PIXEL (labels cover ~0.5% of pixels, so an
        unbiased random crop would usually contain no supervision at all)."""
        self.files = files
        self.meta = meta
        self.lut = build_label_lut(meta)
        self.augment = augment
        self.crop_size = crop_size
        self.gen = torch.Generator().manual_seed(seed)
        self.norm = {k: (torch.tensor(v["mean"]).float(), torch.tensor(v["std"]).float())
                     for k, v in norm.items()}

    def __len__(self):
        return len(self.files)

    def _norm_temporal(self, arr, mask, key):
        # arr [T,H,W,C] -> [T,C,H,W]; normalise, zero masked, append mask channel
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
        m = torch.from_numpy(mask).float().unsqueeze(1)          # [T,1,H,W]
        mean, std = self.norm[key]
        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        x = x * m                                                # masked -> 0
        return torch.cat([x, m], dim=1)                          # [T,C+1,H,W]

    def _augment(self, tensors):
        if torch.rand((), generator=self.gen) < 0.5:
            tensors = [torch.flip(t, dims=[-1]) for t in tensors]
        if torch.rand((), generator=self.gen) < 0.5:
            tensors = [torch.flip(t, dims=[-2]) for t in tensors]
        k = int(torch.randint(0, 4, (1,), generator=self.gen))
        if k:
            tensors = [torch.rot90(t, k, dims=[-2, -1]) for t in tensors]
        return tensors

    def __getitem__(self, i):
        d = np.load(os.path.join(CHIPS_DIR, self.files[i]))
        awifs = self._norm_temporal(d["awifs"], d["awifs_mask"], "awifs")
        s1 = self._norm_temporal(d["s1_asc"], d["s1_asc_mask"], "s1_asc")
        s2 = torch.from_numpy(d["s2"]).permute(2, 0, 1).float()
        mean, std = self.norm["s2"]
        s2 = (s2 - mean[:, None, None]) / std[:, None, None]

        crop = d["crop_id"].astype(np.int64)
        label = torch.from_numpy(self.lut[crop])                 # [H,W], ignore=-100
        label_mask = torch.from_numpy(d["label_mask"]).float()
        field_id = torch.from_numpy(d["field_id"].astype(np.int64))   # [H,W], 0=bg
        region = torch.tensor(int(d["region_id"]), dtype=torch.long)

        if self.augment:
            awifs, s1, s2, label, label_mask, field_id = self._augment(
                [awifs, s1, s2, label, label_mask, field_id])

        if self.crop_size:
            cs = self.crop_size
            H_, W_ = label.shape[-2], label.shape[-1]
            ys, xs = torch.nonzero(label_mask, as_tuple=True)
            if len(ys):
                j = int(torch.randint(0, len(ys), (1,), generator=self.gen))
                cy, cx = int(ys[j]), int(xs[j])
            else:                                   # no labels (shouldn't happen)
                cy, cx = H_ // 2, W_ // 2
            y0 = min(max(cy - cs // 2, 0), H_ - cs)
            x0 = min(max(cx - cs // 2, 0), W_ - cs)
            sl = (..., slice(y0, y0 + cs), slice(x0, x0 + cs))
            awifs, s1, s2 = awifs[sl], s1[sl], s2[sl]
            label, label_mask, field_id = label[sl], label_mask[sl], field_id[sl]

        return {"awifs": awifs, "s1_asc": s1, "s2": s2, "region": region,
                "label": label.long(), "label_mask": label_mask,
                "field_id": field_id.long(), "region_id": int(region),
                "chip": self.files[i]}


# ============================================================
# Convenience builder
# ============================================================
def build_dataloaders(batch_size=8, num_workers=0, augment_train=True, seed=42,
                      train_crop=None, eval_batch_size=None):
    meta = load_metadata()
    splits = make_splits(meta, seed=seed)
    norm = fit_normalization(splits["train"], meta)
    class_w, counts = compute_class_weights(splits["train"], meta)

    ds = {s: AgriChips(splits[s], meta, norm,
                       augment=(s == "train" and augment_train), seed=seed,
                       crop_size=(train_crop if s == "train" else None))
          for s in ("train", "val", "test")}
    ebs = eval_batch_size or batch_size
    loaders = {
        s: DataLoader(ds[s], batch_size=(batch_size if s == "train" else ebs),
                      shuffle=(s == "train"),
                      num_workers=num_workers, drop_last=(s == "train"))
        for s in ("train", "val", "test")}
    return loaders, meta, norm, class_w, counts, splits


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    loaders, meta, norm, class_w, counts, splits = build_dataloaders(batch_size=8, seed=42)
    lut = build_label_lut(meta)

    print("Split sizes:", {s: len(splits[s]) for s in ("train", "val", "test")})
    for s in ("train", "val", "test"):
        regs = [0, 0]
        for f in splits[s]:
            regs[int(np.load(os.path.join(CHIPS_DIR, f))["region_id"])] += 1
        print(f"  {s}: UP={regs[0]} BIHAR={regs[1]}")

    print("\nPer-class labelled pixels in TRAIN and class weights:")
    for i, name in sorted(meta["index_to_name"].items(), key=lambda kv: int(kv[0])):
        i = int(i)
        print(f"  {i:2d} {name:38s} px={counts[i]:7d}  w={class_w[i]:.3f}")

    # class coverage per split (are rare classes present everywhere?)
    print("\nClasses present per split:")
    for s in ("train", "val", "test"):
        present = set()
        for f in splits[s]:
            crop = np.load(os.path.join(CHIPS_DIR, f))["crop_id"]
            idx = lut[crop[crop > 0]]; present |= set(idx[idx != IGNORE_INDEX].tolist())
        print(f"  {s}: {sorted(present)}")

    print("\nNormalisation (fit on train):")
    for k, v in norm.items():
        print(f"  {k}: mean={[round(x,2) for x in v['mean']]} std={[round(x,2) for x in v['std']]}")

    batch = next(iter(loaders["train"]))
    print("\nOne train batch tensor shapes:")
    for k in ("awifs", "s1_asc", "s2", "region", "label", "label_mask"):
        print(f"  {k:11s} {tuple(batch[k].shape)}  {batch[k].dtype}")
    print("  label unique:", torch.unique(batch["label"]).tolist())
