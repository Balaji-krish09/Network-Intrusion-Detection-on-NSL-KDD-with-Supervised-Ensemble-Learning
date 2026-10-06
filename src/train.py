"""Train and evaluate the final binary and 5-class detectors.

Run from the project root:  python -m src.train
Outputs: models/*.joblib, reports/predictions.npz, reports/metrics.json
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from .data import CLASSES, load_all
from .features import make_preprocessor
from .model import (EnsembleDetector, HierarchicalDetector, base_models, hierarchical_decision,
                    leave_attack_out_folds, oof_predict, tune_threshold)

ROOT = Path(__file__).resolve().parents[1]
MULTICLASS_WEIGHT_POWER = 0.5  # sqrt-balanced sample weights: a middle ground between none and fully balanced


def binary_metrics(y, proba, thr):
    pred = proba >= thr
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba),
            "false_alarm_rate": float(pred[y == 0].mean())}


def multiclass_metrics(y, pred):
    rec = recall_score(y, pred, average=None, labels=range(len(CLASSES)))
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "weighted_f1": f1_score(y, pred, average="weighted"),
            **{f"recall_{c}": float(r) for c, r in zip(CLASSES, rec)}}


def main():
    t0 = time.time()
    train, test, test21 = load_all()
    out, metrics = {}, {}

    # ---------------- Binary: normal vs attack ----------------
    y = train["is_attack"].to_numpy()
    print("Binary: random 3-fold CV with LightGBM (for comparison only)")
    rand = np.zeros(len(train))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(train, y):
        pre = make_preprocessor(scale=False)
        m = base_models("binary")["lgbm"].fit(pre.fit_transform(train.iloc[a]), y[a])
        rand[b] = m.predict_proba(pre.transform(train.iloc[b]))[:, 1]
    out["bin_oof_random"] = rand

    # Leave-attack-out OOF is deterministic (fixed folds and seeds), so it is cached between runs.
    cache = ROOT / "reports" / "cache_binary_oof.npz"
    if cache.exists():
        print(f"Binary: leave-attack-type-out CV (loaded from {cache.name})")
        oof = dict(np.load(cache))
    else:
        print("Binary: leave-attack-type-out CV")
        oof = oof_predict(train, y, leave_attack_out_folds(train, y))
        np.savez_compressed(cache, **oof)
    thr, curve = tune_threshold(y, oof["ensemble"])
    print(f"  tuned threshold = {thr:.4f}")

    det = EnsembleDetector("binary").fit(train, y)
    det.threshold = thr
    joblib.dump(det, ROOT / "models" / "binary_detector.joblib")
    for name, df in [("test", test), ("test21", test21)]:
        avg, members = det.predict_proba(df, return_members=True)
        out[f"bin_{name}"] = avg
        for m, p in members.items():
            out[f"bin_{name}_{m}"] = p
    for m, p in oof.items():
        out[f"bin_oof_{m}"] = p
    out["bin_thr_grid"], out["bin_thr_f1"] = curve.index.to_numpy(), curve.to_numpy()

    metrics["binary"] = {
        "threshold": thr,
        "cv_random@0.5": binary_metrics(y, out["bin_oof_random"], 0.5),
        "cv_leave_attack_out@0.5": binary_metrics(y, oof["ensemble"], 0.5),
        "cv_leave_attack_out@tuned": binary_metrics(y, oof["ensemble"], thr),
        "test+@0.5": binary_metrics(test.is_attack.to_numpy(), out["bin_test"], 0.5),
        "test+@tuned": binary_metrics(test.is_attack.to_numpy(), out["bin_test"], thr),
        "test-21@0.5": binary_metrics(test21.is_attack.to_numpy(), out["bin_test21"], 0.5),
        "test-21@tuned": binary_metrics(test21.is_attack.to_numpy(), out["bin_test21"], thr),
    }
    print(json.dumps({k: v for k, v in metrics["binary"].items() if "test" in k}, indent=1))

    # ---------------- Multiclass: Normal / DoS / Probe / R2L / U2R ----------------
    cmap = {c: i for i, c in enumerate(CLASSES)}
    yc = train["category"].map(cmap).to_numpy()
    print("Multiclass: leave-attack-type-out CV")
    oofc = oof_predict(train, yc, leave_attack_out_folds(train, yc, n_splits=3), "multiclass",
                       MULTICLASS_WEIGHT_POWER)

    detc = EnsembleDetector("multiclass", MULTICLASS_WEIGHT_POWER).fit(train, yc)
    hier = HierarchicalDetector(det, detc)
    joblib.dump(hier, ROOT / "models" / "multiclass_detector.joblib")
    out["mc_oof"] = oofc["ensemble"]
    for name, df in [("test", test), ("test21", test21)]:
        out[f"mc_{name}"] = detc.predict_proba(df)
    yt, y21 = test.category.map(cmap).to_numpy(), test21.category.map(cmap).to_numpy()
    metrics["multiclass"] = {
        "cv_leave_attack_out@argmax": multiclass_metrics(yc, oofc["ensemble"].argmax(1)),
        "cv_leave_attack_out@hierarchical": multiclass_metrics(
            yc, hierarchical_decision(oof["ensemble"], oofc["ensemble"], thr)),
        "test+@argmax": multiclass_metrics(yt, out["mc_test"].argmax(1)),
        "test+@hierarchical": multiclass_metrics(yt, hier.decide(out["bin_test"], out["mc_test"])),
        "test-21@argmax": multiclass_metrics(y21, out["mc_test21"].argmax(1)),
        "test-21@hierarchical": multiclass_metrics(y21, hier.decide(out["bin_test21"], out["mc_test21"])),
    }
    print(json.dumps({k: v for k, v in metrics["multiclass"].items() if "test" in k}, indent=1))

    np.savez_compressed(ROOT / "reports" / "predictions.npz", **out)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
