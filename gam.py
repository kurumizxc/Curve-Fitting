# gam_airquality.py
# GAM implementation for Philippine AQI data with visualizations and PH map
# All plots are automatically saved in the "gam" folder.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
import geopandas as gpd
from statsmodels.graphics.gofplots import qqplot


def main():
    # ==========================================
    # 0. OUTPUT FOLDER
    # ==========================================
    OUT_DIR = "gam"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ==========================================
    # 1. LOAD DATA (uses 202412_CombinedData.csv)
    # ==========================================
    DATA_PATH = "202412_CombinedData.csv"  # same dataset as GLM

    df = pd.read_csv(DATA_PATH)

    cols_needed = [
        "main.aqi",
        "components.pm2_5",
        "components.pm10",
        "components.no2",
        "components.o3",
        "coord.lon",
        "coord.lat",
        "city_name",
    ]

    df = df[cols_needed].dropna().copy()

    y = df["main.aqi"].values
    X = df[["components.pm2_5", "components.pm10", "components.no2", "components.o3"]].values

    # ==========================================
    # 2. SPECIFY AND FIT GAM
    #    (Gaussian identity, smooths for all 4 pollutants)
    # ==========================================

    # term indices in X:
    # 0: pm2_5, 1: pm10, 2: no2, 3: o3
    gam = LinearGAM(
        s(0) +  # s(pm2_5)
        s(1) +  # s(pm10)
        s(2) +  # s(no2)
        s(3)    # s(o3)
    )

    # gridsearch chooses smoothing parameters via internal CV
    gam.gridsearch(X, y)

    print(gam.summary())

    # ==========================================
    # 3. MODEL METRICS
    # ==========================================
    y_pred = gam.predict(X)      # fitted values (like GLM's fitted)
    resid = y - y_pred

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(resid ** 2))
    mae = np.mean(np.abs(resid))

    print("\n=== GAM (LinearGAM) metrics ===")
    print(f"R^2   : {r2:.3f}")
    print(f"RMSE  : {rmse:.3f}")
    print(f"MAE   : {mae:.3f}")

    # ==========================================
    # 4. VISUALISATIONS (MATCH GLM STYLE)
    # ==========================================

    # (a) Residual histogram + Q–Q plot (same style as GLM)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(resid, bins=30, density=True)
    plt.title("GAM residuals – histogram")
    plt.xlabel("Residuals")

    plt.subplot(1, 2, 2)
    qqplot(resid, line="45", fit=True, ax=plt.gca())
    plt.title("GAM residuals – Q–Q plot")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gam_residual_hist_qq.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (b) Residuals vs fitted (predicted), like GLM
    plt.figure()
    plt.scatter(y_pred, resid, alpha=0.3)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted values (GAM)")
    plt.ylabel("Residuals")
    plt.title("GAM: Residuals vs fitted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gam_residuals_vs_fitted.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (c) Observed vs predicted (same type as GLM)
    plt.figure()
    plt.scatter(y, y_pred, alpha=0.3)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Observed main.aqi")
    plt.ylabel("Predicted main.aqi")
    plt.title("GAM: Observed vs predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gam_observed_vs_predicted.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (d) Relationship plots for each pollutant:
    #     main.aqi vs predictor with GAM fitted overlay (same idea as GLM)
    predictor_names = [
        "components.pm2_5",
        "components.pm10",
        "components.no2",
        "components.o3",
    ]

    for j, name in enumerate(predictor_names):
        safe_name = name.replace(".", "_")

        # Sort by predictor for a smoother line
        order = np.argsort(df[name].values)
        x_sorted = df[name].values[order]
        y_pred_sorted = y_pred[order]

        plt.figure()
        plt.scatter(df[name], y, alpha=0.2, label="Observed")
        plt.plot(x_sorted, y_pred_sorted, color="red", linewidth=2, label="Fitted (GAM)")
        plt.xlabel(name)
        plt.ylabel("main.aqi")
        plt.title(f"GAM: main.aqi vs {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"gam_main_aqi_vs_{safe_name}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # ==========================================
    # 5. OPTIONAL: Partial effect plots (extra, for interpretation)
    # ==========================================

    for term_idx, label in enumerate(predictor_names):
        XX = gam.generate_X_grid(term=term_idx)

        # partial_dependence with CI returns (pdep, confi)
        pdep, confi = gam.partial_dependence(term=term_idx, X=XX, width=0.95)
        lower = confi[:, 0]
        upper = confi[:, 1]

        safe_label = label.replace(".", "_")

        plt.figure()
        plt.plot(XX[:, term_idx], pdep)
        plt.fill_between(
            XX[:, term_idx],
            lower,
            upper,
            alpha=0.3,
        )
        plt.xlabel(label)
        plt.ylabel(f"f({label})")
        plt.title(f"GAM smooth for {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"gam_smooth_{safe_label}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # ==========================================
    # 6. PHILIPPINES-STYLE HEATMAP (CITY-LEVEL, WITH MAP)
    # ==========================================

    # Attach GAM predictions to the dataframe
    df["gam_pred"] = y_pred

    # Aggregate at city level
    city_gam = df.groupby("city_name").agg(
        lon=("coord.lon", "mean"),
        lat=("coord.lat", "mean"),
        aqi_obs=("main.aqi", "mean"),
        aqi_pred=("gam_pred", "mean"),
    ).reset_index()

    # Load world countries from Natural Earth (Admin 0 – countries)
    ne_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(ne_url)

    # Try to isolate the Philippines depending on column names
    if "ADMIN" in world.columns:
        ph = world[world["ADMIN"] == "Philippines"]
    elif "SOVEREIGNT" in world.columns:
        ph = world[world["SOVEREIGNT"] == "Philippines"]
    elif "NAME" in world.columns:
        ph = world[world["NAME"] == "Philippines"]
    elif "name" in world.columns:
        ph = world[world["name"] == "Philippines"]
    else:
        ph = world

    # Create GeoDataFrame for city points
    gdf_cities_gam = gpd.GeoDataFrame(
        city_gam,
        geometry=gpd.points_from_xy(city_gam["lon"], city_gam["lat"]),
        crs="EPSG:4326",
    )

    # Plot Philippines map + city points colored by predicted AQI (GAM)
    fig, ax = plt.subplots(figsize=(6, 8))

    # Base PH map
    ph.plot(ax=ax, color="white", edgecolor="black")

    # Cities
    gdf_cities_gam.plot(
        ax=ax,
        column="aqi_pred",
        markersize=60,
        cmap="plasma",
        legend=True,
        legend_kwds={"label": "Predicted AQI (GAM, city mean)"},
        alpha=0.9,
        edgecolor="black",
    )

    ax.set_title("Philippines AQI heatmap (GAM, city-level)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gam_philippines_map.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
