"""
Trains and evaluates a Random Forest classifier predicting whether a South Sudanese
county will experience a flood in a given month.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, RocCurveDisplay
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import matplotlib.pyplot as plt

PANEL_PATH = Path("./processed_data_rf/flood_panel.parquet")
OUT_DIR = Path("./processed_data_rf")

# "flooded" = any flood (recurring + unusual).
# "unusual_flood" is the harder prediction task
TARGET_COL = "flooded"
TRAIN_END_YEAR = 2021  # train on <= this year, test on later years

# If True, drop static exposure/geography features (cattle, land cover, population,
# health facilities, area) and keep only weather/hydrology + season - isolates how much
# predictive power the meteorological data has on its own.
HAZARD_ONLY = False

FEATURE_VERSION = "roll6" # Change to roll3 or roll6 for filepath

STATIC_FEATURES = {
    "area_sqkm", "crop_pct", "rangeland_pct", "cattle_count",
    "population", "health_facility_count",
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["adm2_name", "month"]).copy()
    df["month_num"] = df["month"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    # Lag everything that varies month-to-month, so the model only ever sees the past.
    group = df.groupby("adm2_name")
    for col in ["precip_mm", "runoff_mm", "discharge_m3s"]:
        df[f"{col}_lag1"] = group[col].shift(1)
        df[f"{col}_roll3"] = group[col].transform(lambda s: s.shift(1).rolling(3).sum())
        df[f"{col}_roll6"] = group[col].transform(lambda s: s.shift(1).rolling(6).sum())
    for col in [c for c in df.columns if c.startswith("lake_") and not c.endswith("level_lag1")]:
        df[f"{col}_lag1"] = group[col].shift(1)
        df[f"{col}_roll6"] = group[col].transform(lambda s: s.shift(1).rolling(6).mean())

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    non_lagged_dynamic = {"precip_mm", "runoff_mm", "discharge_m3s"}
    non_lagged_dynamic |= {c for c in df.columns if c.startswith("lake_") and c.endswith("level")}
    drop_cols = {
        "adm2_name", "month", "year", "discharge_station_id",
        "flood_pixel_count", "unusual_pixel_count", "flooded", "unusual_flood",
    } | non_lagged_dynamic
    return [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]


def main():
    panel = pd.read_parquet(PANEL_PATH)
    panel["month"] = panel["month"].astype("period[M]")
    panel = add_time_features(panel)

    feature_cols = get_feature_columns(panel)

    # Drop features that are missing for most/all rows
    na_frac = panel[feature_cols].isna().mean()
    usable = na_frac[na_frac < 0.5].index.tolist()
    if len(usable) < len(feature_cols):
        print(f"Dropping features with too much missing data: {sorted(set(feature_cols) - set(usable))}")
    feature_cols = usable

    if HAZARD_ONLY:
        feature_cols = [c for c in feature_cols if c not in STATIC_FEATURES]
        print(f"HAZARD_ONLY mode: keeping {len(feature_cols)} weather/hydrology features only")

    if FEATURE_VERSION == "roll6":
        feature_cols = [c for c in feature_cols if not c.endswith("_roll3")]
    elif FEATURE_VERSION == "roll3":
        feature_cols = [c for c in feature_cols if not c.endswith("_roll6")]

    print(f"Using {len(feature_cols)} features:\n  " + "\n  ".join(feature_cols))
    print(panel[feature_cols].isna().sum())

    data = panel.dropna(subset=feature_cols + [TARGET_COL])
    train = data[data["year"] <= TRAIN_END_YEAR]
    test = data[data["year"] > TRAIN_END_YEAR]
    print(f"\nTrain: {len(train)} rows ({train['year'].min()}-{train['year'].max()})"
          f" | Test: {len(test)} rows ({test['year'].min()}-{test['year'].max()})")
    print(f"Positive rate ({TARGET_COL}) — train: {train[TARGET_COL].mean():.3f}, "
          f"test: {test[TARGET_COL].mean():.3f}")

    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_test, y_test = test[feature_cols], test[TARGET_COL]

    clf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        class_weight="balanced",  # floods are a rare event per county-month
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)

    print("\n--- Classification report (threshold=0.5) ---")
    print(classification_report(y_test, pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"PR-AUC (average precision): {average_precision_score(y_test, proba):.3f} trust this one more than ROC-AUC because of the class imbalance")

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- Feature importances ---")
    print(importances.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    importances.head(15).iloc[::-1].plot.barh(ax=axes[0])
    axes[0].set_title("Top 15 feature importances")
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1])
    axes[1].set_title("ROC curve (test set)")
    plt.tight_layout()

    suffix = "_hazard_only" if HAZARD_ONLY else ""
    out_path = OUT_DIR / f"rf_evaluation_{TARGET_COL}{suffix}_{FEATURE_VERSION}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved evaluation plot to {out_path}")

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print("\n--- Confusion-matrix-based metrics (threshold=0.5) ---")
    print(f"Accuracy:              {accuracy_score(y_test, pred):.3f}")
    print(f"Precision:             {precision_score(y_test, pred):.3f}")
    print(f"False positive rate:   {fpr:.3f}  (predicted flood, no flood occurred)")
    print(f"False negative rate:   {fnr:.3f}  (missed an actual flood)")


if __name__ == "__main__":
    main()