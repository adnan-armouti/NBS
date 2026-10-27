"""Train a per-modality XRF55 HAR classifier and report top-1 = the train-on-real
upper bound for the NBS cross-band payoff (Stage 6).

  python -m eval.downstream.train_har --modality mmwave --scenes Scene1 --epochs 12
Eval a trained model on synthesized inputs later via --input-dir (Stage 6 swap).
"""
import os, sys, json, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.downstream.dataset import XRFHar
from eval.downstream import models

DATA = "/home/adnan/Desktop/NBS/data/XRF55"


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); correct = tot = 0
    for x, y in loader:
        p = model(x.to(dev)).argmax(1).cpu()
        correct += (p == y).sum().item(); tot += len(y)
    return correct / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="mmwave", choices=["mmwave", "wifi", "rfid"])
    ap.add_argument("--scenes", nargs="+", default=["Scene1"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-trials", nargs="+", default=None,
                    help="subsample training trials e.g. 01 02 03 (speed)")
    ap.add_argument("--split-mode", default="subject", choices=["subject", "random"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # one manifest per scene; concatenate datasets
    def make(split, trials):
        dss = []
        for sc in a.scenes:
            mani = os.path.join(DATA, f"manifest_{sc}.csv")
            dss.append(XRFHar(mani, a.modality, split, scenes={sc}, trials=trials,
                              split_mode=a.split_mode))
        return torch.utils.data.ConcatDataset(dss)

    tr = make("train", a.train_trials)
    te = make("test", None)
    print(f"[data] {a.modality} {a.scenes} train={len(tr)} test={len(te)} dev={dev}")
    tl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(te, a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    model = models.build(a.modality).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = torch.nn.CrossEntropyLoss()

    best = 0.0
    out = os.path.join("output", f"har_{a.modality}_{'_'.join(a.scenes)}{a.tag}")
    os.makedirs(out, exist_ok=True)
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
            run += loss.item()
        sched.step()
        acc = evaluate(model, vl, dev)
        best = max(best, acc)
        print(f"  ep{ep+1:02d}/{a.epochs} loss={run/len(tl):.3f} test_top1={acc:.4f} "
              f"best={best:.4f} ({time.time()-t0:.0f}s)")
        if acc >= best:
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        json.dump({"modality": a.modality, "scenes": a.scenes, "train_n": len(tr),
                   "test_n": len(te), "epoch": ep + 1, "test_top1": acc, "best_top1": best,
                   "train_trials": a.train_trials, "n_classes": 55, "chance": 1 / 55},
                  open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"[done] {a.modality} {a.scenes} BEST test top-1 = {best:.4f} (chance {1/55:.3f}) -> {out}")


if __name__ == "__main__":
    main()
