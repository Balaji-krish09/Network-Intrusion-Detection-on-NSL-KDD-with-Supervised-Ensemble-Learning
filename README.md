# Network Intrusion Detection: NSL-KDD

Detects malicious network connections in the [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd) dataset:
- **binary**: normal vs attack
- **5-class**: Normal / DoS / Probe / R2L / U2R

## Results

All modelling decisions were made with cross-validation on `KDDTrain+`. `KDDTest+` and `KDDTest-21` were never used for tuning.

| Task | Model | KDDTest+ accuracy | KDDTest-21 accuracy |
|---|---|---|---|
| Binary | Baseline: best single model, raw features, 0.5 cutoff | 79.1% | 60.3% |
| Binary | **Final: ensemble + engineered features + tuned threshold** | **93.6%** | **88.6%** |
| 5-class | Plain multi-class argmax | 75.8% | 54.0% |
| 5-class | **Final: two-stage (binary detector → attack category)** | **82.5%** | **67.6%** |

Final binary model on KDDTest+: precision 92.0%, recall (detection rate) 97.2%, F1 0.945, ROC-AUC 0.967, false-alarm rate 11.2%.
Final 5-class model on KDDTest+: macro-F1 0.67. Recall: Normal 89%, DoS 87%, Probe 92%, R2L 42%, U2R 72%.

Exact numbers are in `reports/metrics.json`.

## Key ideas

1. **Leave-attack-type-out cross-validation.** 17% of KDDTest+ consists of 17 attack types never seen in training.
   Random CV scores about 99.9% and badly overestimates test performance. Holding out whole attack types per fold gives
   a validation score that behaves like the test set.
2. **Threshold tuning on that CV.** The models rank attacks well (AUC about 0.97), but at the 0.5 cutoff they miss a third of
   test attacks, mostly the novel ones. A threshold chosen on leave-attack-out predictions raises detection from 66% to 97%.
3. **Two-stage 5-class model.** The binary detector flags attacks, then a class model assigns the category. This lifts R2L
   recall from 8% to 42% and U2R from 22% to 72%.
4. **Feature engineering.** Log transforms, byte ratios, error-rate aggregates, service concentration, host-intrusion signals,
   flag groups and rare-service grouping. Improves leave-attack-out CV for every tree model.
5. **Diverse, regularised ensemble.** XGBoost + LightGBM + ExtraTrees (binary), soft voting.

**Trade-off:** the tuned threshold raises false alarms on normal traffic (2.8% → 11.2% on KDDTest+). If that is too high,
choose a higher threshold from the curve in notebook 02, §4.2.

## Project layout

```
data/                     NSL-KDD files (KDDTrain+.txt, KDDTest+.txt, KDDTest-21.txt, ...)
notebooks/
  01_eda.ipynb                                 exploratory data analysis
  02_feature_engineering_and_modeling.ipynb    validation, features, models, results
src/
  data.py                 loading, column names, attack → category mapping
  features.py             feature engineering + preprocessing pipeline
  model.py                ensemble, leave-attack-out CV, threshold tuning, two-stage detector
  train.py                trains and evaluates the final models
  compare_features.py     raw vs engineered features across 5 model families
  build_*_notebook.py     generate the notebooks
models/                   saved detectors (joblib)
reports/                  metrics.json, predictions.npz, feature_comparison.csv, figures/
```

## Reproduce

```bash
pip install -r requirements.txt
python -m src.train                 # ~10 min on 4 cores; delete reports/cache_binary_oof.npz to recompute CV from scratch
python -m src.compare_features      # optional, ~30 min
python -m src.build_eda_notebook
python -m src.build_model_notebook
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Using the models

```python
import joblib
from src.data import load, CLASSES

binary = joblib.load("models/binary_detector.joblib")
five_class = joblib.load("models/multiclass_detector.joblib")

df = load("KDDTest+")                      # any DataFrame with the 41 NSL-KDD feature columns
p_attack = binary.predict_proba(df)        # probability of attack
is_attack = binary.predict(df)             # 1 = attack, using the tuned threshold
category = [CLASSES[i] for i in five_class.predict(df)]
```

## Limitations

- NSL-KDD is synthetic 1998 traffic. Results may not carry over to modern networks. Validate on CIC-IDS2017 or UNSW-NB15.
- U2R has only 52 training rows, so its scores are unstable.
- Some novel test attacks are barely separable from normal traffic with these 41 features.
