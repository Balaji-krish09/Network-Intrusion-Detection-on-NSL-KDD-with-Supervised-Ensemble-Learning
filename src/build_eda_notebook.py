"""Builds notebooks/01_eda.ipynb. Run: python -m src.build_eda_notebook"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# NSL-KDD: Exploratory Data Analysis

**Goal:** understand the NSL-KDD intrusion-detection data before modelling. That covers its structure and quality, class balance,
how the features behave, and how the **test distribution differs from training**. That last point matters most for this dataset.

Splits: `KDDTrain+` (125,973 rows), `KDDTest+` (22,544 rows), `KDDTest-21` (11,850 harder rows).
""")
code("""
import sys, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from src.data import load_all, FEATURES, CATEGORICAL, NUMERIC, CLASSES

sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams["figure.dpi"] = 110
FIG = "../reports/figures/"
pd.set_option("display.max_columns", 50)

train, test, test21 = load_all()
print("train", train.shape, "| test", test.shape, "| test-21", test21.shape)
train.head()
""")

md("## 1. Structure and data quality")
code("""
info = pd.DataFrame({
    "dtype": train[FEATURES].dtypes.astype(str),
    "n_unique": train[FEATURES].nunique(),
    "n_missing": train[FEATURES].isna().sum(),
    "pct_zero": (train[FEATURES] == 0).mean().round(3),
})
info
""")
code("""
dup = train.duplicated(subset=FEATURES).sum()
conflict = train.groupby(FEATURES).category.nunique().gt(1).sum()
constant = [c for c in NUMERIC if train[c].nunique() <= 1]
print(f"Missing values (all splits): {train.isna().sum().sum() + test.isna().sum().sum()}")
print(f"Duplicate feature rows in train: {dup}  ({conflict} with conflicting labels)")
print(f"Constant columns: {constant}")
mostly_zero = info[info.pct_zero > 0.99].index.tolist()
print(f"Columns that are >99% zero: {mostly_zero}")
""")

md("## 2. Target distribution")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (name, df) in zip(axes, [("Train", train), ("Test+", test)]):
    vc = df.category.value_counts().reindex(CLASSES)
    ax.bar(vc.index, vc.values, color=sns.color_palette("tab10", 5))
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}\\n({v/len(df):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{name}: attack category"); ax.set_ylim(0, vc.max() * 1.25)
plt.tight_layout(); plt.savefig(FIG + "class_distribution.png", bbox_inches="tight"); plt.show()

dist = pd.concat([train.category.value_counts(normalize=True).rename("train"),
                  test.category.value_counts(normalize=True).rename("test+"),
                  test21.category.value_counts(normalize=True).rename("test-21")], axis=1).reindex(CLASSES)
(dist * 100).round(2)
""")
code("""
# Attack-level counts and the attack types that appear only in test
lab = pd.concat([train.label.value_counts().rename("train"), test.label.value_counts().rename("test+")], axis=1).fillna(0).astype(int)
lab["category"] = lab.index.map(dict(zip(pd.concat([train, test]).label, pd.concat([train, test]).category)))
lab["unseen_in_train"] = lab.train.eq(0)
unseen = lab[lab.unseen_in_train]
print(f"{len(unseen)} attack types appear only in test, covering {unseen['test+'].sum():,} rows "
      f"({unseen['test+'].sum()/len(test):.1%} of Test+)")
lab.sort_values(["category", "train"], ascending=[True, False])
""")
code("""
fig, ax = plt.subplots(figsize=(8, 4))
novel = test.assign(novel=~test.label.isin(train.label.unique()))
pd.crosstab(novel.category, novel.novel).reindex(CLASSES).plot.bar(stacked=True, ax=ax, color=["#4C72B0", "#DD8452"])
ax.legend(["seen in train", "novel attack type"]); ax.set_title("Test+: share of each category that is a novel attack type")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIG + "novel_attacks.png", bbox_inches="tight"); plt.show()
""")

md("### Difficulty level\nEach row has a difficulty score: the number of 21 reference classifiers that labelled it correctly. Lower means harder.")
code("""
fig, ax = plt.subplots(figsize=(10, 3.5))
for name, df in [("train", train), ("test+", test)]:
    df.difficulty.value_counts(normalize=True).sort_index().plot(ax=ax, marker="o", label=name)
ax.set_xlabel("difficulty level (21 = easiest)"); ax.set_ylabel("share of rows"); ax.legend()
plt.tight_layout(); plt.show()
print("Rows with difficulty <= 15 -> train: {:.1%}  test+: {:.1%}".format((train.difficulty <= 15).mean(), (test.difficulty <= 15).mean()))
""")

md("## 3. Categorical features")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
pd.crosstab(train.protocol_type, train.category, normalize="index")[CLASSES].plot.bar(stacked=True, ax=axes[0])
axes[0].set_title("protocol_type -> category share"); axes[0].tick_params(axis="x", rotation=0)
pd.crosstab(train.flag, train.category, normalize="index")[CLASSES].plot.bar(stacked=True, ax=axes[1], legend=False)
axes[1].set_title("flag -> category share"); axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig(FIG + "protocol_flag.png", bbox_inches="tight"); plt.show()
pd.crosstab(train.flag, train.category)[CLASSES]
""")
code("""
svc = train.groupby("service").agg(n=("is_attack", "size"), attack_rate=("is_attack", "mean")).sort_values("n", ascending=False)
fig, ax = plt.subplots(figsize=(13, 4))
top = svc.head(30)
ax.bar(top.index, top.n, color=plt.cm.RdYlGn_r(top.attack_rate))
ax.set_yscale("log"); ax.set_title("Top 30 services by count (colour = attack rate: green=benign, red=attack)")
plt.xticks(rotation=70); plt.tight_layout(); plt.savefig(FIG + "services.png", bbox_inches="tight"); plt.show()
print(f"{train.service.nunique()} services; {(svc.n < 20).sum()} have fewer than 20 rows; "
      f"{(svc.attack_rate.isin([0, 1])).sum()} are 100% normal or 100% attack")
svc.head(15).T
""")

md("## 4. Numeric features: distributions and skew")
code("""
desc = train[NUMERIC].describe().T
desc["skew"] = train[NUMERIC].skew()
desc.sort_values("skew", ascending=False).round(3)
""")
code("""
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.flat, ["src_bytes", "dst_bytes", "count", "srv_count", "dst_host_count", "dst_host_srv_count", "same_srv_rate", "dst_host_serror_rate"]):
    for cat in CLASSES:
        v = train.loc[train.category == cat, c]
        v = v if "rate" in c else np.log1p(v)
        sns.kdeplot(v, ax=ax, label=cat, warn_singular=False, cut=0, common_norm=False)
    ax.set_title(f"log1p({c})" if "rate" not in c else c); ax.set_xlabel("")
axes[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG + "numeric_kde.png", bbox_inches="tight"); plt.show()
""")
code("""
rate_cols = [c for c in NUMERIC if "rate" in c]
fig, ax = plt.subplots(figsize=(14, 4))
train.groupby("category")[rate_cols].mean().reindex(CLASSES).T.plot.bar(ax=ax)
ax.set_title("Mean of the rate features by category"); plt.xticks(rotation=60, ha="right")
plt.tight_layout(); plt.savefig(FIG + "rate_features.png", bbox_inches="tight"); plt.show()
""")

md("## 5. Feature relevance: mutual information with the target")
code("""
X = train[FEATURES].copy()
for c in CATEGORICAL:
    X[c] = X[c].astype("category").cat.codes
disc = [c in CATEGORICAL or c in ("land", "logged_in", "root_shell", "is_host_login", "is_guest_login") for c in FEATURES]
mi_bin = mutual_info_classif(X, train.is_attack, discrete_features=disc, random_state=0)
mi_cat = mutual_info_classif(X, train.category, discrete_features=disc, random_state=0)
mi = pd.DataFrame({"MI_binary": mi_bin, "MI_5class": mi_cat}, index=FEATURES).sort_values("MI_5class", ascending=False)
fig, ax = plt.subplots(figsize=(14, 4))
mi.plot.bar(ax=ax); ax.set_title("Mutual information with the target")
plt.tight_layout(); plt.savefig(FIG + "mutual_information.png", bbox_inches="tight"); plt.show()
print("Near-zero MI (<0.001):", mi.index[mi.MI_5class < 1e-3].tolist())
mi.head(15).round(4)
""")
code("""
top = mi.index[:8]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.flat, top):
    if c in CATEGORICAL:
        train[c].value_counts().head(8).plot.bar(ax=ax)
    else:
        sns.boxplot(data=train, x="category", y=c, order=CLASSES, ax=ax, showfliers=False)
    ax.set_title(c); ax.set_xlabel("")
plt.tight_layout(); plt.savefig(FIG + "top_features_by_class.png", bbox_inches="tight"); plt.show()
""")

md("## 6. Correlation and redundancy")
code("""
num = [c for c in NUMERIC if train[c].nunique() > 1]
corr = train[num].corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .6}, xticklabels=True, yticklabels=True)
plt.title("Pearson correlation of numeric features"); plt.tight_layout()
plt.savefig(FIG + "correlation.png", bbox_inches="tight"); plt.show()

pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().rename("r").reset_index()
pairs = pairs.loc[pairs.r.abs() > 0.9].sort_values("r", ascending=False)
print(f"{len(pairs)} feature pairs with |r| > 0.9")
pairs
""")

md("## 7. Train vs test drift\nThe classic NSL-KDD trap: a model can score 99%+ on a random train split and much lower on `KDDTest+`. Two checks:")
code("""
ks = pd.DataFrame([(c, *ks_2samp(train[c], test[c])) for c in num], columns=["feature", "KS", "p"]).set_index("feature").sort_values("KS", ascending=False)
fig, ax = plt.subplots(figsize=(14, 3.5))
ks.KS.plot.bar(ax=ax); ax.set_title("Kolmogorov-Smirnov statistic train vs Test+ (higher = more drift)")
plt.tight_layout(); plt.savefig(FIG + "drift_ks.png", bbox_inches="tight"); plt.show()
ks.head(10).round(4)
""")
code("""
# Adversarial validation: can a model tell train rows from test rows?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
both = pd.concat([train[FEATURES].assign(is_test=0), test[FEATURES].assign(is_test=1)], ignore_index=True)
for c in CATEGORICAL:
    both[c] = both[c].astype("category").cat.codes
rf = RandomForestClassifier(200, n_jobs=-1, random_state=0, class_weight="balanced", min_samples_leaf=5)
auc = cross_val_score(rf, both[FEATURES], both.is_test, cv=3, scoring="roc_auc").mean()
print(f"Adversarial validation AUC: {auc:.3f}  (0.5 = identical distributions, 1.0 = fully separable)")
rf.fit(both[FEATURES], both.is_test)
pd.Series(rf.feature_importances_, FEATURES).sort_values(ascending=False).head(10).round(4)
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, c in zip(axes, ["src_bytes", "dst_host_srv_count", "same_srv_rate"]):
    f = np.log1p if "bytes" in c else (lambda s: s)
    for name, df in [("train", train), ("test+", test)]:
        sns.kdeplot(f(df.loc[df.is_attack == 1, c]), ax=ax, label=f"{name} attacks", warn_singular=False)
    ax.set_title(c); ax.legend(fontsize=8)
plt.suptitle("Attack traffic looks different in test than in train"); plt.tight_layout(); plt.show()
""")

md("## 8. Low-dimensional view (PCA)")
code("""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
s = train.sample(20000, random_state=0)
Z = np.log1p(s[num].clip(lower=0))
pc = PCA(2, random_state=0).fit(StandardScaler().fit_transform(Z))
P = pc.transform(StandardScaler().fit_transform(Z))
plt.figure(figsize=(8, 6))
for cat in CLASSES:
    m = (s.category == cat).values
    plt.scatter(P[m, 0], P[m, 1], s=4 if cat in ("Normal", "DoS") else 14, alpha=.5, label=cat)
plt.legend(markerscale=3); plt.title(f"PCA of log-scaled numeric features ({pc.explained_variance_ratio_.sum():.0%} variance)")
plt.tight_layout(); plt.savefig(FIG + "pca.png", bbox_inches="tight"); plt.show()
""")

md("""
## 9. Key findings → decisions for feature engineering and modelling

| Finding | Evidence | Decision |
|---|---|---|
| Clean data: no missing values, only 16 duplicate rows | §1 | No imputation. Keep duplicates (too few to matter). |
| `num_outbound_cmds` is constant | §1 | Drop it. |
| Severe imbalance: U2R = 52 rows (0.04%), R2L = 995 (0.8%) | §2 | Class weights, macro-F1 / per-class recall as metrics, try oversampling. |
| **Test has 17 attack types not in train (~17% of Test+)**, and R2L jumps from 0.8% to 13% | §2 | Plain random CV is misleading. Validate with **leave-attack-type-out CV**. Tune the decision threshold for recall on unseen attacks. |
| Volume and count features are extremely skewed (skew up to 350) | §4 | `log1p` transform. Ratio features. |
| `flag`, `service`, error rates, `same_srv_rate` and byte counts carry the most signal | §3, §5 | Engineer flag groups, error-rate aggregates, byte ratios and service concentration. |
| Many rare services | §3 | One-hot with rare categories grouped (`min_frequency`). |
| Groups of near-duplicate rate features (r > 0.9) | §6 | Optional correlation pruning. Trees handle it; linear models benefit. |
| Strong train→test drift (adversarial AUC well above 0.5) | §7 | Prefer models that generalise: regularised boosting / ensembles. Report on Test+ and Test-21 separately. |
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "nslkdd-py311", "display_name": "Python 3.11 (NSL-KDD)", "language": "python"}})
out = ROOT / "notebooks" / "01_eda.ipynb"
nbf.write(nb, out)
print("wrote", out)
