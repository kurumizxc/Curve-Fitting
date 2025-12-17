from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mpl_toolkits.basemap import Basemap


DATA_PATH = "202412_CombinedData.csv"
OUTDIR = Path("results_glm")

RANDOM_STATE = 42
TEST_SIZE = 0.2

RESPONSE = "main_aqi"

PREDICTORS = [
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


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace(".", "_") for c in df.columns]
    # KEY FIX: drop duplicate column names created by sanitization
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


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.replace([np.inf, -np.inf], np.nan).dropna()
    Xv = sm.add_constant(Xc, has_constant="add")
    rows = []
    for i, col in enumerate(Xv.columns):
        if col == "const":
            continue
        try:
            vif_val = float(variance_inflation_factor(Xv.values, i))
        except Exception:
            vif_val = float("inf")
        rows.append({"variable": col, "VIF": vif_val})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def _to_1d(a) -> np.ndarray:
    """Force a to 1D numpy array; if it is 2D (duplicate columns), take the first column."""
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

    # KEY FIX: enforce equal length
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


def main() -> None:
    ensure_outdir()

    df = pd.read_csv(DATA_PATH)
    df = sanitize_columns(df)
    df = add_time_features(df)

    lon_col = pick_col(df, ["coord_lon", "lon", "longitude"])
    lat_col = pick_col(df, ["coord_lat", "lat", "latitude"])

    if RESPONSE not in df.columns:
        raise ValueError(f"Response column '{RESPONSE}' not found. Available: {list(df.columns)}")

    available_predictors = [c for c in PREDICTORS if c in df.columns]
    if lon_col not in available_predictors:
        available_predictors.append(lon_col)
    if lat_col not in available_predictors:
        available_predictors.append(lat_col)

    needed = [RESPONSE] + available_predictors + [lon_col, lat_col]
    model_df = df[needed].replace([np.inf, -np.inf], np.nan).dropna()

    y = model_df[RESPONSE].astype(float)
    X = model_df[available_predictors].astype(float)

    fig_hist_response(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train_c = sm.add_constant(X_train, has_constant="add")
    X_test_c = sm.add_constant(X_test, has_constant="add")

    glm = sm.GLM(y_train, X_train_c, family=sm.families.Gaussian())
    res = glm.fit()

    (OUTDIR / "model_summary.txt").write_text(res.summary().as_text(), encoding="utf-8")

    vif_df = compute_vif(X_train)
    vif_df.to_csv(OUTDIR / "vif.csv", index=False)

    y_pred = res.predict(X_test_c)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test.values, y_pred.values))),
        "mae": float(mean_absolute_error(y_test.values, y_pred.values)),
        "r2": float(r2_score(y_test.values, y_pred.values)),
        "aic": float(res.aic) if hasattr(res, "aic") else None,
        "bic": float(res.bic) if hasattr(res, "bic") else None,
    }
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig_obs_vs_fitted(y_test.values, y_pred.values, "Observed vs Fitted (GLM)")

    resid = y_test.values - y_pred.values
    fig_resid_vs_fitted(y_pred.values, resid)
    fig_qq_residuals(resid)

    lon = model_df.loc[X_test.index, lon_col]
    lat = model_df.loc[X_test.index, lat_col]

    fig_ph_map_scatter(lon, lat, y_test.values, "fig_map_observed_aqi.png", "Philippines map: Observed AQI (test set)")
    fig_ph_map_scatter(lon, lat, y_pred.values, "fig_map_fitted_aqi.png", "Philippines map: Fitted AQI (GLM, test set)")

    print("GLM done. Outputs in:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
