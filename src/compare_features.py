"""Raw 41 features vs engineered features, across 5 model families (binary task, threshold 0.5).

Run from the project root:  python -m src.compare_features   (~30 min on 4 cores)
Output: reports/feature_comparison.csv
"""
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .data import load_all
from .features import make_preprocessor, make_raw_preprocessor
from .model import leave_attack_out_folds

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000),
    "RandomForest": lambda: RandomForestClassifier(300, n_jobs=-1, random_state=0),
    "ExtraTrees": lambda: ExtraTreesClassifier(300, n_jobs=-1, random_state=0),
    "XGBoost": lambda: XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=8, n_jobs=-1,
                                     random_state=0, verbosity=0),
    "LightGBM": lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                       random_state=0, verbose=-1),
}


def main():
    train, test, test21 = load_all()
    y = train.is_attack.to_numpy()
    folds = leave_attack_out_folds(train, y)
    rows = []
    for fe_name, pre in [("raw", make_raw_preprocessor), ("engineered", make_preprocessor)]:
        for name, make in MODELS.items():
            t = time.time()
            pipe = Pipeline([("pre", pre()), ("clf", make())])
            oof = cross_val_predict(pipe, train, y, cv=folds, method="predict_proba")[:, 1]
            pipe.fit(train, y)
            pt, p21 = pipe.predict_proba(test)[:, 1], pipe.predict_proba(test21)[:, 1]
            rows.append({"features": fe_name, "model": name,
                         "lao_cv_acc": accuracy_score(y, oof > .5), "lao_cv_f1": f1_score(y, oof > .5),
                         "test_acc": accuracy_score(test.is_attack, pt > .5),
                         "test_f1": f1_score(test.is_attack, pt > .5),
                         "test_recall": recall_score(test.is_attack, pt > .5),
                         "test_auc": roc_auc_score(test.is_attack, pt),
                         "test21_acc": accuracy_score(test21.is_attack, p21 > .5)})
            print(rows[-1], f"{time.time() - t:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "feature_comparison.csv", index=False)


if __name__ == "__main__":
    main()
