# glm_airquality.py
# GLM implementation for Philippine AQI data with visualizations and PH map
# All plots are automatically saved in the "glm" folder.

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import geopandas as gpd


def main():
    # ==========================================
    # 0. OUTPUT FOLDER
    # ==========================================
    OUT_DIR = "glm"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ==========================================
    # 1. LOAD DATA (uses 202412_CombinedData.csv)
    # ==========================================
    DATA_PATH = "202412_CombinedData.csv"  # make sure this file is in the same folder

    df = pd.read_csv(DATA_PATH)

    # Keep the columns we need (include coords + city for the map)
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

    # Drop rows with missing value(s) in required columns
    df = df[cols_needed].dropna().copy()

    # ==========================================
    # 2. PREPARE RESPONSE AND PREDICTORS
    # ==========================================
    y = df["main.aqi"].values
    X = df[["components.pm2_5", "components.pm10", "components.no2", "components.o3"]]

    # Add intercept
    X = sm.add_constant(X)

    # ==========================================
    # 3. FIT GLM (Gaussian with identity link)
    # ==========================================
    glm_model = sm.GLM(
        y,
        X,
        family=sm.families.Gaussian(sm.families.links.identity())
    )
    glm_result = glm_model.fit()

    print(glm_result.summary())

    # ==========================================
    # 4. MODEL METRICS
    # ==========================================
    fitted = glm_result.fittedvalues
    resid = y - fitted

    deviance = glm_result.deviance
    null_deviance = glm_result.null_deviance
    pseudo_r2 = 1 - deviance / null_deviance
    rmse = np.sqrt(np.mean(resid ** 2))
    mae = np.mean(np.abs(resid))

    print("\n=== GLM (Gaussian identity) metrics ===")
    print(f"AIC        : {glm_result.aic:.3f}")
    print(f"Pseudo R^2 : {pseudo_r2:.3f}")
    print(f"RMSE       : {rmse:.3f}")
    print(f"MAE        : {mae:.3f}")

    # ==========================================
    # 5. STANDARD VISUALISATIONS (SAVED TO FOLDER)
    # ==========================================

    # (a) Histogram + Q–Q plot of residuals
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(resid, bins=30, density=True)
    plt.title("GLM residuals – histogram")
    plt.xlabel("Residuals")

    plt.subplot(1, 2, 2)
    qqplot(resid, line="45", fit=True, ax=plt.gca())
    plt.title("GLM residuals – Q–Q plot")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "glm_residual_hist_qq.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (b) Residuals vs fitted
    plt.figure()
    plt.scatter(fitted, resid, alpha=0.3)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted values (GLM)")
    plt.ylabel("Residuals")
    plt.title("GLM: Residuals vs fitted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "glm_residuals_vs_fitted.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (c) Observed vs predicted
    plt.figure()
    plt.scatter(y, fitted, alpha=0.3)
    min_val = min(y.min(), fitted.min())
    max_val = max(y.max(), fitted.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Observed main.aqi")
    plt.ylabel("Predicted main.aqi")
    plt.title("GLM: Observed vs predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "glm_observed_vs_predicted.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (d) Relationship plots for each pollutant
    predictor_names = [
        "components.pm2_5",
        "components.pm10",
        "components.no2",
        "components.o3",
    ]

    for name in predictor_names:
        safe_name = name.replace(".", "_")
        plt.figure()
        plt.scatter(df[name], y, alpha=0.2, label="Observed")
        plt.scatter(df[name], fitted, alpha=0.2, label="Fitted (GLM)", s=10)
        plt.xlabel(name)
        plt.ylabel("main.aqi")
        plt.title(f"GLM: main.aqi vs {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"glm_main_aqi_vs_{safe_name}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # ==========================================
    # 6. PHILIPPINES-STYLE HEATMAP (CITY-LEVEL, WITH MAP)
    # ==========================================

    # Attach GLM predictions to the dataframe
    df["glm_pred"] = fitted

    # Aggregate at city level to avoid overplotting:
    city_glm = df.groupby("city_name").agg(
        lon=("coord.lon", "mean"),
        lat=("coord.lat", "mean"),
        aqi_obs=("main.aqi", "mean"),
        aqi_pred=("glm_pred", "mean"),
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
    gdf_cities_glm = gpd.GeoDataFrame(
        city_glm,
        geometry=gpd.points_from_xy(city_glm["lon"], city_glm["lat"]),
        crs="EPSG:4326",
    )

    # Plot Philippines map + city points colored by predicted AQI
    fig, ax = plt.subplots(figsize=(6, 8))

    # Base PH map
    ph.plot(ax=ax, color="white", edgecolor="black")

    # Cities
    gdf_cities_glm.plot(
        ax=ax,
        column="aqi_pred",
        markersize=60,
        cmap="viridis",
        legend=True,
        legend_kwds={"label": "Predicted AQI (GLM, city mean)"},
        alpha=0.9,
        edgecolor="black",
    )

    ax.set_title("Philippines AQI heatmap (GLM, city-level)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "glm_philippines_map.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
