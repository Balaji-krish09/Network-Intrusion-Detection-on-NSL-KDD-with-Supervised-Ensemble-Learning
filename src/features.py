"""Feature engineering and preprocessing pipeline for NSL-KDD."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES

# Constant in train (always 0) -> carries no information.
DROP = ["num_outbound_cmds"]

# Heavy-tailed counts/volumes (skew > 10) that get a log1p transform.
LOG_COLS = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "num_compromised", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count",
]


class NSLFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain-driven features. Stateless, so it is safe inside CV."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[[c for c in FEATURES if c not in DROP]].copy()
        eps = 1.0
        # Traffic volume and direction
        X["total_bytes"] = X["src_bytes"] + X["dst_bytes"]
        X["src_dst_ratio"] = np.log1p(X["src_bytes"]) - np.log1p(X["dst_bytes"])
        X["no_payload"] = (X["total_bytes"] == 0).astype(int)
        X["bytes_per_sec"] = X["total_bytes"] / (X["duration"] + eps)
        # Connection-error aggregates (SYN errors -> DoS/scans, REJ -> probes)
        X["serror_total"] = X[["serror_rate", "srv_serror_rate",
                               "dst_host_serror_rate", "dst_host_srv_serror_rate"]].mean(axis=1)
        X["rerror_total"] = X[["rerror_rate", "srv_rerror_rate",
                               "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]].mean(axis=1)
        # Service concentration of the recent connections
        X["srv_count_ratio"] = X["srv_count"] / (X["count"] + eps)
        X["dst_host_srv_ratio"] = X["dst_host_srv_count"] / (X["dst_host_count"] + eps)
        # Host-intrusion indicators (R2L / U2R are payload-content attacks)
        X["intrusion_signals"] = (
            X["hot"] + X["num_failed_logins"] + X["num_compromised"] + X["root_shell"]
            + X["su_attempted"] + X["num_root"] + X["num_file_creations"]
            + X["num_shells"] + X["num_access_files"]
        )
        X["root_activity"] = ((X["root_shell"] > 0) | (X["num_root"] > 0) | (X["su_attempted"] > 0)).astype(int)
        # Common flag shortcuts
        X["flag_is_sf"] = (X["flag"] == "SF").astype(int)
        X["flag_half_open"] = X["flag"].isin(["S0", "S1", "S2", "S3", "SH"]).astype(int)
        X["flag_rejected"] = X["flag"].isin(["REJ", "RSTO", "RSTOS0", "RSTR"]).astype(int)

        for c in LOG_COLS + ["total_bytes", "bytes_per_sec", "intrusion_signals"]:
            X[c] = np.log1p(X[c])
        return X


def make_preprocessor(scale: bool = True, min_frequency: int = 20) -> Pipeline:
    """Feature engineering -> one-hot categoricals (rare services grouped) -> optional scaling."""
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                        sparse_output=False)
    ct = ColumnTransformer(
        [("cat", ohe, CATEGORICAL)],
        remainder=StandardScaler() if scale else "passthrough",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return Pipeline([("fe", NSLFeatureEngineer()), ("ct", ct)])


def make_raw_preprocessor() -> ColumnTransformer:
    """Baseline: raw 41 features, one-hot + scaling, no engineering."""
    num = [c for c in FEATURES if c not in CATEGORICAL]
    ct = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
         ("num", StandardScaler(), num)],
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


def drop_correlated(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Return columns to drop so no remaining pair has |corr| above threshold."""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    return [c for c in upper.columns if (upper[c] > threshold).any()]
