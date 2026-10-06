"""Models, leave-attack-type-out CV, and the final NSL-KDD detector."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .data import CLASSES
from .features import make_preprocessor

SEED = 0


def leave_attack_out_folds(df: pd.DataFrame, y, n_splits: int = 5):
    """CV folds where every attack *type* sits entirely in one fold.

    The validation fold therefore contains attack types the model never saw,
    mimicking KDDTest+, where ~17% of rows are novel attack types. Normal rows
    are spread randomly over 50 pseudo-groups so they still split evenly.
    """
    rng = np.random.RandomState(SEED)
    groups = np.where(df["label"] == "normal",
                      "normal_" + pd.Series(rng.randint(0, 50, len(df))).astype(str).values,
                      df["label"])
    cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED)
    return list(cv.split(df, y, groups))


def base_models(task: str = "binary") -> dict:
    """Diverse, regularised learners that make up the ensemble.

    Binary: XGBoost + LightGBM + ExtraTrees. Multiclass: LightGBM + ExtraTrees (multiclass
    XGBoost was left out to keep training time reasonable on a 4-core machine).
    """
    if task == "multiclass":
        return {
            "lgbm": LGBMClassifier(n_estimators=300, learning_rate=0.06, num_leaves=31,
                                   min_child_samples=20, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.5, reg_lambda=5, random_state=SEED, verbose=-1),
            "et": ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=SEED,
                                       class_weight="balanced_subsample"),
        }
    return {
        "xgb": XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                             colsample_bytree=0.5, min_child_weight=5, reg_lambda=5, n_jobs=-1,
                             random_state=SEED, verbosity=0),
        "lgbm": LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                               min_child_samples=50, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, reg_lambda=5, random_state=SEED, verbose=-1),
        "et": ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    }


def sample_weights(y, power: float):
    """power 0 = unweighted, 0.5 = sqrt-balanced, 1 = fully balanced."""
    return compute_sample_weight("balanced", y) ** power if power else None


class EnsembleDetector:
    """Feature engineering + soft-voting ensemble.

    task="binary": predicts 1 = attack when mean P(attack) >= threshold (tuned on leave-attack-out OOF).
    task="multiclass": predicts the argmax class.
    """

    def __init__(self, task: str = "binary", weight_power: float = 0.0):
        self.task, self.weight_power = task, weight_power
        self.threshold = 0.5

    def fit(self, df: pd.DataFrame, y):
        self.pre_ = make_preprocessor(scale=False)
        X = self.pre_.fit_transform(df)
        w = sample_weights(y, self.weight_power)
        self.models_ = {}
        for name, m in base_models(self.task).items():
            m.fit(X, y) if name == "et" else m.fit(X, y, sample_weight=w)
            self.models_[name] = m
        return self

    def predict_proba(self, df: pd.DataFrame, return_members: bool = False):
        X = self.pre_.transform(df)
        members = {n: m.predict_proba(X) for n, m in self.models_.items()}
        if self.task == "binary":
            members = {n: p[:, 1] for n, p in members.items()}
        avg = np.mean(list(members.values()), axis=0)
        return (avg, members) if return_members else avg

    def decide(self, proba):
        if self.task == "binary":
            return (proba >= self.threshold).astype(int)
        return np.argmax(proba, axis=1)

    def predict(self, df: pd.DataFrame):
        return self.decide(self.predict_proba(df))

    @property
    def feature_names_(self):
        return list(self.pre_.named_steps["ct"].get_feature_names_out())


def oof_predict(df, y, folds, task="binary", weight_power=0.0, verbose=True):
    """Out-of-fold ensemble probabilities (and per-member ones) on leave-attack-out folds."""
    n_out = 1 if task == "binary" else len(CLASSES)
    shape = (len(df),) if n_out == 1 else (len(df), n_out)
    oof = {n: np.zeros(shape) for n in ["ensemble", *base_models(task)]}
    for i, (a, b) in enumerate(folds):
        det = EnsembleDetector(task, weight_power).fit(df.iloc[a], y[a])
        avg, members = det.predict_proba(df.iloc[b], return_members=True)
        oof["ensemble"][b] = avg
        for n, p in members.items():
            oof[n][b] = p
        if verbose:
            print(f"  fold {i + 1}/{len(folds)} done", flush=True)
    return oof


def tune_threshold(y, proba, grid=None):
    """Threshold that maximises F1 on out-of-fold probabilities."""
    grid = np.unique(np.r_[np.logspace(-4, -1, 61), np.linspace(0.1, 0.9, 17)]) if grid is None else grid
    scores = np.array([f1_score(y, proba >= t) for t in grid])
    return float(grid[scores.argmax()]), pd.Series(scores, index=grid)


class HierarchicalDetector:
    """Two-stage 5-class detector.

    Stage 1: the binary detector (with its tuned threshold) decides normal vs attack.
    Stage 2: for flagged rows, the multiclass model picks the most likely *attack* category.
    This carries stage 1's strength on novel attacks into the category prediction.
    """

    def __init__(self, binary: EnsembleDetector, multiclass: EnsembleDetector):
        self.binary, self.multiclass = binary, multiclass

    def decide(self, p_attack, class_proba):
        return hierarchical_decision(p_attack, class_proba, self.binary.threshold)

    def predict(self, df: pd.DataFrame):
        return self.decide(self.binary.predict_proba(df), self.multiclass.predict_proba(df))


def hierarchical_decision(p_attack, class_proba, threshold):
    """0 (Normal) below the threshold, otherwise the best of the 4 attack classes (1..4)."""
    return np.where(p_attack >= threshold, class_proba[:, 1:].argmax(axis=1) + 1, 0)
