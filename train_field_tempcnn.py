"""Field-level 1D TempCNN (rebuild) — the rice-rescue specialist.

Architecture reconstructed from the surviving runs/tempcnn_v1/best_model.pt
state dict: 3x [Conv1d(k=3) + BN + ReLU] (7->32->32->64), masked mean-pool
over the 11 months, head Linear(64+8 -> 64) + ReLU + Dropout + Linear(64 -> 13).
Trained on the 13-class raw scheme; the finalize step only consumes
P(rice) = softmax[..., 12].
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AGRI_DATA_DIR", os.path.join(ROOT, "Extracted_dataset_gee"))

N_CLASSES = 13
RICE_IDX = 12  # crop 36 in CLASSES_13
SEED = 42


class FieldTempCNN(nn.Module):
    def __init__(self, in_ch=7, static_dim=8, n_classes=N_CLASSES):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, 3, padding=1), nn.BatchNorm1d(co), nn.ReLU())
        self.temporal = nn.Sequential(block(in_ch, 32), block(32, 32), block(32, 64))
        self.head = nn.Sequential(nn.Linear(64 + static_dim, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, seq, static):
        # seq [B, T, C] -> conv over time
        h = self.temporal(seq.permute(0, 2, 1))  # [B, 64, T]
        h = h.mean(-1)
        return self.head(torch.cat([h, static], 1))


def load_seq_data():
    d = np.load(os.path.join(DATA_DIR, "field_seq.npz"), allow_pickle=True)
    with open(os.path.join(DATA_DIR, "splits.json")) as f:
        splits = json.load(f)
    chip = d["chip"]
    idx = {s: np.isin(chip, splits[s]) for s in ("train", "val", "test")}
    return d, idx


def normalize(seq, static, norm=None):
    """Standardize the 5 signal channels + 4 static-S2 dims with train stats."""
    seq = seq.copy()
    static = static.copy()
    if norm is None:
        mu = seq[..., :5].reshape(-1, 5).mean(0)
        sd = seq[..., :5].reshape(-1, 5).std(0) + 1e-6
        smu = static[:, 4:].mean(0)
        ssd = static[:, 4:].std(0) + 1e-6
        norm = dict(mu=mu, sd=sd, smu=smu, ssd=ssd)
    seq[..., :5] = (seq[..., :5] - norm["mu"]) / norm["sd"]
    static[:, 4:] = (static[:, 4:] - norm["smu"]) / norm["ssd"]
    return seq, static, norm


def train(out_dir=os.path.join(ROOT, "runs", "tempcnn_rebuilt"), epochs=250, verbose=True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    d, idx = load_seq_data()
    seq_tr, st_tr, norm = normalize(d["seq"][idx["train"]], d["static"][idx["train"]])
    seq_va, st_va, _ = normalize(d["seq"][idx["val"]], d["static"][idx["val"]], norm)
    y_tr = d["y13"][idx["train"]].astype(np.int64)
    y_va = d["y13"][idx["val"]].astype(np.int64)

    cnt = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
    w = np.sqrt(cnt.sum() / np.maximum(cnt, 1))
    w /= w.mean()

    model = FieldTempCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w))
    Xtr = torch.tensor(seq_tr)
    Str = torch.tensor(st_tr)
    Ytr = torch.tensor(y_tr)
    Xva = torch.tensor(seq_va)
    Sva = torch.tensor(st_va)

    from sklearn.metrics import f1_score
    best, best_state, patience = -1, None, 0
    B = 256
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), B):
            j = perm[i:i + B]
            opt.zero_grad()
            loss = lossf(model(Xtr[j], Str[j]), Ytr[j])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva, Sva).argmax(1).numpy()
        f1 = f1_score(y_va, pv, average="macro", zero_division=0)
        if f1 > best:
            best, best_state, patience = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if verbose and (ep % 10 == 0 or patience == 0):
            print(f"ep {ep:3d} loss {loss.item():.3f} val macroF1 {f1:.3f} (best {best:.3f})")
        if patience >= 40:
            break
    model.load_state_dict(best_state)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": best_state,
                "norm": {k: v.tolist() for k, v in norm.items()},
                "val_macroF1": best}, os.path.join(out_dir, "best_model.pt"))
    print(f"best val macroF1 {best:.3f} -> {out_dir}")
    return model, norm


def rice_probs(model, norm, seq, static):
    seq, static, _ = normalize(seq, static, {k: np.asarray(v, np.float32) for k, v in norm.items()})
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(seq), torch.tensor(static)), 1).numpy()
    return p[:, RICE_IDX]


def load_trained(out_dir=os.path.join(ROOT, "runs", "tempcnn_rebuilt")):
    ck = torch.load(os.path.join(out_dir, "best_model.pt"), map_location="cpu")
    model = FieldTempCNN()
    model.load_state_dict(ck["state_dict"])
    return model, ck["norm"]


if __name__ == "__main__":
    train()
