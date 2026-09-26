import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut


def make_synthetic_panel(n_stations=20, n_days=365, n_features=4, seed=42):
    """Create a synthetic panel dataset with station IDs and timestamps.

    This keeps the geometry of the methodological setup described in the text:
    a temporally ordered record indexed by date and a station/group identifier.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2022-01-01", periods=n_stations * n_days, freq="D")
    station_ids = np.repeat(np.arange(n_stations), n_days)

    X = rng.normal(size=(len(station_ids), n_features))
    signal = (
        0.5 * X[:, 0]
        + 0.3 * X[:, 1]
        + 0.2 * np.sin(np.arange(len(station_ids)) / 30.0)
        + station_ids / max(1, n_stations)
    )
    y = (signal + rng.normal(0, 0.25, size=len(station_ids)) > 0).astype(int)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station_ids,
        "target": y,
        **{f"feature_{i}": X[:, i] for i in range(n_features)},
    })
    return df


def chronological_split(df, train_fraction=0.70, val_fraction=0.15, test_fraction=0.15):
    """Split a sorted time series chronologically without shuffling."""
    n_rows = len(df)
    train_end = int(n_rows * train_fraction)
    val_end = train_end + int(n_rows * val_fraction)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n_rows)

    return train_idx, val_idx, test_idx


def rolling_origin_evaluation(df, model, horizon_months=3, step_months=3, n_origins=4):
    """Evaluate temporal generalization by advancing the split origin.

    The paper describes training on all data up to an origin and evaluating on the
    following three-month block, then moving the origin forward and repeating.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["month"] = df["timestamp"].dt.to_period("M")
    month_sequence = df["month"].unique()

    if len(month_sequence) < (n_origins * step_months + horizon_months):
        raise ValueError("Not enough months for the requested rolling-origin evaluation.")

    fold_metrics = []
    for fold in range(n_origins):
        origin_month = month_sequence[fold * step_months + 6]
        train_mask = df["month"] < origin_month
        test_mask = (df["month"] >= origin_month) & (df["month"] < origin_month + horizon_months)

        if not train_mask.any() or not test_mask.any():
            continue

        X_train = df.loc[train_mask, [col for col in df.columns if col.startswith("feature_")]].to_numpy()
        y_train = df.loc[train_mask, "target"].to_numpy()
        X_test = df.loc[test_mask, [col for col in df.columns if col.startswith("feature_")]].to_numpy()
        y_test = df.loc[test_mask, "target"].to_numpy()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        fold_metrics.append({
            "fold": fold + 1,
            "origin": str(origin_month),
            "accuracy": acc,
        })

    return fold_metrics


def leave_one_station_out_evaluation(df, model):
    """Spatial generalization: hold out one station at a time and train on all others."""
    logo = LeaveOneGroupOut()
    station_ids = df["station_id"].to_numpy()
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    X = df[feature_cols].to_numpy()
    y = df["target"].to_numpy()

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, station_ids), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        held_out_station = station_ids[test_idx][0]
        fold_metrics.append({
            "fold": fold,
            "held_out_station": int(held_out_station),
            "accuracy": acc,
        })

    return fold_metrics


if __name__ == "__main__":
    df = make_synthetic_panel(n_stations=20, n_days=365, n_features=4, seed=42)
    df = df.sort_values("timestamp").reset_index(drop=True)

    train_idx, val_idx, test_idx = chronological_split(df)
    print("Chronological split sizes:", len(train_idx), len(val_idx), len(test_idx))

    model = RandomForestClassifier(random_state=42)
    rolling_folds = rolling_origin_evaluation(df, model)
    print("\nRolling-origin results:")
    for fold in rolling_folds:
        print(f"Fold {fold['fold']}: origin={fold['origin']}, accuracy={fold['accuracy']:.4f}")

    print("\nLeave-one-station-out results:")
    loso_folds = leave_one_station_out_evaluation(df, model)
    for fold in loso_folds:
        print(f"Fold {fold['fold']}: held-out station={fold['held_out_station']}, accuracy={fold['accuracy']:.4f}")

    mean_accuracy = np.mean([fold["accuracy"] for fold in loso_folds])
    print(f"\nMean leave-one-station-out accuracy: {mean_accuracy:.4f}")
