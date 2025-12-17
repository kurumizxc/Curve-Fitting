from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.graphics.gofplots import qqplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mpl_toolkits.basemap import Basemap


DATA_PATH = "202412_CombinedData.csv"
OUTDIR = Path("results_gam")

RANDOM_STATE = 42
TEST_SIZE = 0.2

RESPONSE = "main_aqi"

SMOOTH_COLS = [
    "components_co",
    "components_no",
    "components_no2",
    "components_o3",
    "components_so2",
    "components_nh3",
    "components_pm2_5",
    "components_pm10",
    "coord_lon",
    "coord_lat",
    "hour",
    "month",
    "dayofweek",
]

PARAMETRIC_COLS: list[str] = []

SPLINE_DF = 6
SPLINE_DEGREE = 3


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace(".", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns exist: {candidates}")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt_col = None
    for c in ["datetime", "dt", "date", "timestamp"]:
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None:
        return df

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    df = df.dropna(subset=[dt_col])
    df["hour"] = df[dt_col].dt.hour.astype(int)
    df["month"] = df[dt_col].dt.month.astype(int)
    df["dayofweek"] = df[dt_col].dt.dayofweek.astype(int)
    return df


def ensure_outdir() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def _to_1d(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr.ravel()


def fig_hist_response(y: pd.Series) -> None:
    plt.figure()
    plt.hist(y, bins=20)
    plt.title("Histogram of response (AQI)")
    plt.xlabel("AQI")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_hist_response.png", dpi=200)
    plt.close()


def fig_obs_vs_fitted(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([lo, hi], [lo, hi])
    plt.title(title)
    plt.xlabel("Observed AQI")
    plt.ylabel("Fitted AQI")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_obs_vs_fitted.png", dpi=200)
    plt.close()


def fig_resid_vs_fitted(y_pred: np.ndarray, resid: np.ndarray) -> None:
    plt.figure()
    plt.scatter(y_pred, resid, s=10)
    plt.axhline(0)
    plt.title("Residuals vs Fitted")
    plt.xlabel("Fitted AQI")
    plt.ylabel("Residuals (Observed - Fitted)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_resid_vs_fitted.png", dpi=200)
    plt.close()


def fig_qq_residuals(resid: np.ndarray) -> None:
    plt.figure()
    qqplot(resid, line="45", ax=plt.gca())
    plt.title("QQ plot of residuals")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_qq_residuals.png", dpi=200)
    plt.close()


def fig_ph_map_scatter(lon, lat, values, fname: str, title: str) -> None:
    lon = _to_1d(lon)
    lat = _to_1d(lat)
    values = _to_1d(values)

    n = min(len(lon), len(lat), len(values))
    lon, lat, values = lon[:n], lat[:n], values[:n]

    plt.figure(figsize=(8, 8))
    m = Basemap(
        projection="merc",
        llcrnrlon=115, llcrnrlat=4,
        urcrnrlon=128, urcrnrlat=21,
        resolution="l",
    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary()
    x, y = m(lon, lat)
    x = _to_1d(x)
    y = _to_1d(y)

    sc = m.scatter(x, y, c=values, s=12, marker="o")
    plt.title(title)
    plt.colorbar(sc, shrink=0.7)
    plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=200)
    plt.close()


def minmax_fit_transform(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Fit MinMax using TRAIN only, then transform TRAIN and TEST.
    Also clip TEST to TRAIN min/max before scaling (prevents out-of-knot issues).
    """
    train_min = train_df.min(axis=0)
    train_max = train_df.max(axis=0)

    # avoid divide-by-zero if a column is constant
    denom = (train_max - train_min).replace(0, np.nan)

    train_scaled = (train_df - train_min) / denom
    # clip test into training range before scaling
    test_clipped = test_df.clip(lower=train_min, upper=train_max, axis=1)
    test_scaled = (test_clipped - train_min) / denom

    # fill NaN (from constant columns) with 0.0 (all values same anyway)
    train_scaled = train_scaled.fillna(0.0)
    test_scaled = test_scaled.fillna(0.0)

    return train_scaled, test_scaled, train_min, train_max


def main() -> None:
    ensure_outdir()

    df = pd.read_csv(DATA_PATH)
    df = sanitize_columns(df)
    df = add_time_features(df)

    lon_col = pick_col(df, ["coord_lon", "lon", "longitude"])
    lat_col = pick_col(df, ["coord_lat", "lat", "latitude"])

    if RESPONSE not in df.columns:
        raise ValueError(f"Response column '{RESPONSE}' not found. Available: {list(df.columns)}")

    smooth_cols = [c for c in SMOOTH_COLS if c in df.columns]
    if lon_col not in smooth_cols:
        smooth_cols.append(lon_col)
    if lat_col not in smooth_cols:
        smooth_cols.append(lat_col)

    param_cols = [c for c in PARAMETRIC_COLS if c in df.columns]

    needed = [RESPONSE] + smooth_cols + param_cols + [lon_col, lat_col]
    model_df = df[needed].replace([np.inf, -np.inf], np.nan).dropna()

    y = model_df[RESPONSE].astype(float)
    fig_hist_response(y)

    X_smooth_all = model_df[smooth_cols].astype(float)

    if param_cols:
        X_param_all = sm.add_constant(model_df[param_cols].astype(float), has_constant="add")
    else:
        X_param_all = sm.add_constant(pd.DataFrame(index=model_df.index), has_constant="add")

    idx = model_df.index.to_numpy()
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    Xs_train_raw = X_smooth_all.loc[train_idx]
    Xs_test_raw = X_smooth_all.loc[test_idx]

    # KEY FIX: scale + clip to prevent out-of-knot errors
    Xs_train, Xs_test, train_min, train_max = minmax_fit_transform(Xs_train_raw, Xs_test_raw)

    Xp_train = X_param_all.loc[train_idx]
    Xp_test = X_param_all.loc[test_idx]

    k = Xs_train.shape[1]
    df_list = [SPLINE_DF] * k
    deg_list = [SPLINE_DEGREE] * k
    bs = BSplines(Xs_train, df=df_list, degree=deg_list)

    gam = GLMGam(y_train, exog=Xp_train, smoother=bs, family=sm.families.Gaussian())
    res = gam.fit()

    (OUTDIR / "model_summary.txt").write_text(res.summary().as_text(), encoding="utf-8")

    y_pred = res.predict(exog=Xp_test, exog_smooth=Xs_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test.values, y_pred))),
        "mae": float(mean_absolute_error(y_test.values, y_pred)),
        "r2": float(r2_score(y_test.values, y_pred)),
        "aic": float(res.aic) if hasattr(res, "aic") else None,
        "bic": float(res.bic) if hasattr(res, "bic") else None,
        "spline_df": SPLINE_DF,
        "spline_degree": SPLINE_DEGREE,
        "n_smooth_terms": int(k),
    }
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig_obs_vs_fitted(y_test.values, y_pred, "Observed vs Fitted (GAM)")
    resid = y_test.values - y_pred
    fig_resid_vs_fitted(y_pred, resid)
    fig_qq_residuals(resid)

    # NOTE: maps use ORIGINAL lon/lat, not scaled
    lon = model_df.loc[test_idx, lon_col]
    lat = model_df.loc[test_idx, lat_col]

    fig_ph_map_scatter(lon, lat, y_test.values, "fig_map_observed_aqi.png", "Philippines map: Observed AQI (test set)")
    fig_ph_map_scatter(lon, lat, y_pred, "fig_map_fitted_aqi.png", "Philippines map: Fitted AQI (GAM, test set)")

    print("GAM done. Outputs in:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
