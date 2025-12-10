"""
Preprocessing pipeline for CrimeHotspotSim (Cleveland only)
Authors: Emmanuel Bautista, Julian Fennema

This script:

1. Calls the Cleveland Crime_Incidents ArcGIS FeatureService API.
   - URL: https://services3.arcgis.com/dty2kHktVXHrqO8i/ArcGIS/rest/services/Crime_Incidents/FeatureServer/0/query
   - Uses fields: OffenseDate (date), LAT, LON, CENSUS_TRACT_GEOID.
2. Renames OffenseDate → DATE OCC so it plugs into the grid/aggregation pipeline.
3. Converts raw latitude/longitude into a regular grid (0.5 km size by default).
4. Aggregates into WEEKLY crime counts per grid cell.
5. Creates lag features (lag1, lag2, lag3).
6. Optionally fetches tract-level demographics (population, age, income, employment)
   from the Census ACS API and joins them to grid cells via CENSUS_TRACT_GEOID.
7. Adds intervention/context fields (patrol_cars, ped_activity, etc.).
8. Saves:
   - frames.csv (full weekly dataset for model training)
   - inference_frame.csv (latest week's frame for hotspot mapping)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import rasterio

# ---------------------------------------------------------------
# 1. Cleveland Crime Incidents FeatureServer URL
# ---------------------------------------------------------------
CLEVELAND_FEATURE_URL = (
    "https://services3.arcgis.com/dty2kHktVXHrqO8i/ArcGIS/rest/services/"
    "Crime_Incidents/FeatureServer/0/query"
)


# ---------------------------------------------------------------
# Helper: Cleveland API loader → DataFrame with DATE OCC, LAT, LON, CENSUS_TRACT_GEOID
# ---------------------------------------------------------------
def load_cleveland_crime_from_api(
    api_url: str = CLEVELAND_FEATURE_URL,
    grid_fields=("OffenseDate", "LAT", "LON"),
    batch_size: int = 2000,
) -> pd.DataFrame:
    """
    Fetches Cleveland crime incidents from the ArcGIS FeatureService API.
    Returns a DataFrame with columns:
      - DATE OCC (datetime-like)
      - LAT, LON (floats)
      - CENSUS_TRACT_GEOID (string; used to join demographics)

    Notes:
    - MaxRecordCount for this service is ~2000; we page using resultOffset.
    - OffenseDate is an esriFieldTypeDate (ms since epoch) on this service.
    """
    offense_date_field, lat_field, lon_field = grid_fields

    # We also request the tract GEOID so we can join ACS demographics later
    extra_fields = ["CENSUS_TRACT_GEOID", "DISTRICT"]

    records = []
    offset = 0
    total = 0

    while True:
        field_list = [offense_date_field, lat_field, lon_field] + extra_fields
        params = {
            "where": "1=1",  # no filter – pull everything; filter later if needed
            "outFields": ",".join(field_list),
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": batch_size,
        }

        resp = requests.get(api_url, params=params)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        n = len(features)
        if n == 0:
            # No more records
            break

        for feat in features:
            attrs = feat.get("attributes", {})
            records.append(attrs)

        total += n
        offset += n  # move by number of records actually returned

        # Progress print
        print(f"Fetched {n} records (total so far: {total})")

    if not records:
        raise RuntimeError("No records returned from Cleveland Crime_Incidents API.")

    df = pd.DataFrame.from_records(records)

    # Standardize column names so the core pipeline can stay the same
    rename_map = {
        offense_date_field: "DATE OCC",
        lat_field: "LAT",
        lon_field: "LON",
        # CENSUS_TRACT_GEOID already has a good name for joining
    }
    df = df.rename(columns=rename_map)



    # Drop District 0 records if the DISTRICT field is present
    if "DISTRICT" in df.columns:
        # Make sure it's numeric, then filter
        df["DISTRICT"] = pd.to_numeric(df["DISTRICT"], errors="coerce")
        before = len(df)
        df = df[df["DISTRICT"] != 0]
        after = len(df)
        print(f"Dropped {before - after} rows with DISTRICT == 0")




    # Convert date column; ArcGIS often returns ms since epoch, but it may also be
    # ISO strings. We handle both cases.
    date_series = df["DATE OCC"]
    if np.issubdtype(date_series.dtype, np.number):
        df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], unit="ms", errors="coerce")
    else:
        df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")

    # Drop rows with missing core fields
    df = df.dropna(subset=["DATE OCC", "LAT", "LON"])

    # Drop all records from 2015 and earlier
    df = df[df["DATE OCC"].dt.year >= 2016]

    # Ensure tract IDs are strings for safe joining
    if "CENSUS_TRACT_GEOID" in df.columns:
        df["CENSUS_TRACT_GEOID"] = df["CENSUS_TRACT_GEOID"].astype(str)

    # Simple sanity check: how many unique weeks and min/max dates?
    print(f"Total incidents after cleaning: {len(df)}")
    print(f"DATE OCC range: {df['DATE OCC'].min()} → {df['DATE OCC'].max()}")

    # Keep tract GEOID so we can attach ACS demographics by tract later
    return df[["DATE OCC", "LAT", "LON", "CENSUS_TRACT_GEOID"]]


# ---------------------------------------------------------------
# Helper: Load tract-level demographics from the Census ACS API
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# GEOID-based ACS loader — fetches only tracts present in data
# ---------------------------------------------------------------
def load_acs_tracts_by_geoid(target_geoids, api_key, year=2023):
    """
    Pull ACS tract-level demographics ONLY for tracts appearing in Cleveland data.
    Handles tracts in ANY county/state — no more missing values due to county mismatch.

    Parameters:
        target_geoids: iterable of full 11-digit GEOIDs (state+county+tract)
        api_key: Census API key
        year: ACS 5-year dataset year

    Returns:
        DataFrame with:
           CENSUS_TRACT_GEOID, pop_total, median_age, median_income,
           labor_force, unemployed, unemployment_rate
    """
    import requests  # already imported above, but safe to reuse
    from collections import defaultdict

    # Convert all GEOIDs to clean strings
    target_geoids = [str(g).zfill(11) for g in target_geoids]

    # Parse GEOIDs → (state, county, tract)
    groups = defaultdict(list)
    for geoid in target_geoids:
        state = geoid[0:2]
        county = geoid[2:5]
        tract = geoid[5:11]
        groups[(state, county)].append(tract)

    print("\nACS QUERY GROUPS (state, county) → tracts:")
    for key, tracts in groups.items():
        print(f"  {key} → {len(tracts)} tract(s)")

    base_url = f"https://api.census.gov/data/{year}/acs/acs5"

    # Variables we want
    acs_vars = {
        "B01001_001E": "pop_total",
        "B01002_001E": "median_age",
        "B17001_002E": "pop_below_poverty",
        "B19013_001E": "median_income",
        "B23025_003E": "labor_force",
        "B23025_005E": "unemployed",
    }

    results = []

    # Query ACS once per (state, county)
    for (state, county), tracts in groups.items():
        tract_list = ",".join(tracts)

        params = {
            "get": ",".join(acs_vars.keys()),
            "for": f"tract:{tract_list}",
            "in": f"state:{state} county:{county}",
            "key": api_key,
        }

        resp = requests.get(base_url, params=params)
        if resp.status_code != 200:
            print(f"ACS request failed for ({state}, {county})")
            continue

        data = resp.json()
        header = data[0]
        rows = data[1:]

        for row in rows:
            row_dict = dict(zip(header, row))
            full_geoid = (
                row_dict["state"]
                + row_dict["county"]
                + row_dict["tract"].zfill(6)
            )
            row_dict["CENSUS_TRACT_GEOID"] = full_geoid
            results.append(row_dict)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    if df.empty:
        print("WARNING: No ACS data returned.")
        return None

    # Convert ACS columns to numeric
    for code, name in acs_vars.items():
        df[name] = pd.to_numeric(df[code], errors="coerce")

    # Cleanup and compute unemployment_rate
    df["unemployment_rate"] = df["unemployed"] / df["labor_force"]
    df["unemployment_rate"] = df["unemployment_rate"].fillna(0)

    # Compute poverty_rate = pop_below_poverty / pop_total
    df["poverty_rate"] = df["pop_below_poverty"] / df["pop_total"]
    df["poverty_rate"] = df["poverty_rate"].fillna(0)

    # Output cleaned ACS dataset
    keep_cols = [
        "CENSUS_TRACT_GEOID",
        "pop_total",
        "median_age",
        "median_income",
        "labor_force",
        "unemployed",
        "unemployment_rate",
        "poverty_rate",
    ]
    return df[keep_cols]



# ---------------------------------------------------------------------------
# Core panel-building pipeline
# ---------------------------------------------------------------------------
def build_panel(
    df: pd.DataFrame,
    outdir: str,
    grid_step: float = 0.005,
    tracts_df: pd.DataFrame | None = None,
) -> None:
    """
    Core pipeline that assumes df has columns:
    - DATE OCC (datetime-like)
    - LAT (float), LON (float)
    - CENSUS_TRACT_GEOID (optional but used for demographics)

    grid_step = 0.005 degrees ≈ 0.5 km
    This gives us meaningful spatial resolution while keeping compute light.

    If tracts_df is provided, it is expected to hold tract-level demographics
    keyed by CENSUS_TRACT_GEOID, which will be joined to each grid cell via
    the dominant tract observed in that grid.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Ensure datetime
    df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")
    df = df.dropna(subset=["DATE OCC", "LAT", "LON"])

    # ---------------------------------------------------------------
    # 2. Build a spatial grid by rounding lat/lon into bins
    # ---------------------------------------------------------------
    df["lat_bin"] = (df["LAT"] / grid_step).round().astype(int)
    df["lon_bin"] = (df["LON"] / grid_step).round().astype(int)

    # Unique grid cell identifier
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

    # ---------------------------------------------------------------
    # 3. Convert raw timestamps into WEEK START dates (Mondays)
    # ---------------------------------------------------------------
    df["week_start"] = df["DATE OCC"] - pd.to_timedelta(
        df["DATE OCC"].dt.weekday, unit="D"
    )
    df["week_start"] = df["week_start"].dt.normalize()

    # ---------------------------------------------------------------
    # 4. Weekly aggregation: crime_count per (grid, week)
    # ---------------------------------------------------------------
    agg = (
        df.groupby(["grid_id", "lat_bin", "lon_bin", "week_start"])
        .size()
        .reset_index(name="crime_count")
    )

    # ---------------------------------------------------------------
    # 5. Create a FULL PANEL (every grid × every week)
    # This guarantees clean time series → needed for lag features.
    # ---------------------------------------------------------------
    all_weeks = pd.date_range(
        start=agg["week_start"].min(),
        end=agg["week_start"].max(),
        freq="W-MON",
    )

    grids = agg[["grid_id", "lat_bin", "lon_bin"]].drop_duplicates()

    panel = (
        grids.assign(key=1)
        .merge(pd.DataFrame({"week_start": all_weeks, "key": 1}), on="key")
        .drop(columns="key")
        .merge(agg, on=["grid_id", "lat_bin", "lon_bin", "week_start"], how="left")
        .fillna({"crime_count": 0})
        .sort_values(["grid_id", "week_start"])
    )




    # ---------------------------------------------------------------
    # Attach weekly average temperature from data/raw/temperature.csv
    # ---------------------------------------------------------------
    temp_candidates = [
        Path("data/raw/temperature.csv"),
    ]
    temp_path = next((p for p in temp_candidates if p.exists()), None)

    if temp_path is not None:
        print(f"Loading temperature data from {temp_path}")
        temp_df = pd.read_csv(temp_path)

        if "Date" in temp_df.columns:
            # Parse dates and keep 2016+ to match the crime data window
            temp_df["Date"] = pd.to_datetime(temp_df["Date"], errors="coerce")
            temp_df = temp_df.dropna(subset=["Date"])
            temp_df = temp_df[temp_df["Date"].dt.year >= 2016]

            # Prefer TAVG, otherwise fall back to (TMAX + TMIN)/2 if needed
            if "TAVG (Degrees Fahrenheit)" in temp_df.columns:
                temp_df["weather_temp"] = pd.to_numeric(
                    temp_df["TAVG (Degrees Fahrenheit)"], errors="coerce"
                )
            else:
                tmax_col = "TMAX (Degrees Fahrenheit)"
                tmin_col = "TMIN (Degrees Fahrenheit)"
                if tmax_col in temp_df.columns and tmin_col in temp_df.columns:
                    temp_df["weather_temp"] = (
                        pd.to_numeric(temp_df[tmax_col], errors="coerce")
                        + pd.to_numeric(temp_df[tmin_col], errors="coerce")
                    ) / 2.0
                else:
                    print(
                        "WARNING: No TAVG/TMAX/TMIN columns found in temperature file; "
                        "setting weather_temp = 0."
                    )
                    panel["weather_temp"] = 0.0
                    temp_df = None  # to skip weekly calculation

            if temp_df is not None:
                # Compute Monday week_start to match the crime panel
                temp_df["week_start"] = temp_df["Date"] - pd.to_timedelta(
                    temp_df["Date"].dt.weekday, unit="D"
                )
                temp_df["week_start"] = temp_df["week_start"].dt.normalize()

                weekly_temp = (
                    temp_df.groupby("week_start")["weather_temp"]
                    .mean()
                    .reset_index()
                )

                # Join weekly temperature onto every grid × week
                panel = panel.merge(weekly_temp, on="week_start", how="left")

                # For weeks outside the temperature range, fill with 0.0
                panel["weather_temp"] = panel["weather_temp"].fillna(0.0)
        else:
            print(
                "WARNING: temperature file is missing 'Date' column; "
                "setting weather_temp = 0."
            )
            panel["weather_temp"] = 0.0
    else:
        print(
            "WARNING: temperature.csv not found under data/raw; "
            "setting weather_temp = 0."
        )
        panel["weather_temp"] = 0.0





    # ---------------------------------------------------------------
    # Join tract-level demographics to each grid cell
    # ---------------------------------------------------------------
    if "CENSUS_TRACT_GEOID" in df.columns:
        # For each grid, pick the most common tract among incidents
        grid_tract = (
            df.dropna(subset=["CENSUS_TRACT_GEOID"])
            .groupby("grid_id")["CENSUS_TRACT_GEOID"]
            .agg(lambda x: x.value_counts().idxmax())
            .reset_index()
        )
        panel = panel.merge(grid_tract, on="grid_id", how="left")

        # -----------------------------------------------------------
        # DEBUG 1: Check GEOID alignment between Cleveland + ACS
        # -----------------------------------------------------------
        if tracts_df is not None:
            print("\n--- DEBUG: Checking GEOID alignment ---")
            print("Sample GEOIDs from Cleveland df:")
            print(df["CENSUS_TRACT_GEOID"].dropna().astype(str).head())

            print("\nSample GEOIDs from ACS tracts_df:")
            print(tracts_df["CENSUS_TRACT_GEOID"].dropna().astype(str).head())

            cle_ids = set(df["CENSUS_TRACT_GEOID"].dropna().astype(str))
            acs_ids = set(tracts_df["CENSUS_TRACT_GEOID"].dropna().astype(str))
            intersection = cle_ids.intersection(acs_ids)

            print("\nNumber of shared GEOIDs between Cleveland + ACS:",
                  len(intersection))
            print("------------------------------------------\n")

        # -----------------------------------------------------------
        # Attach ACS demographics via CENSUS_TRACT_GEOID
        # -----------------------------------------------------------
        if tracts_df is not None:
            tracts_df = tracts_df.copy()
            tracts_df["CENSUS_TRACT_GEOID"] = tracts_df["CENSUS_TRACT_GEOID"].astype(str)
            panel = panel.merge(
                tracts_df,
                on="CENSUS_TRACT_GEOID",
                how="left",
            )

            # Normalize ACS columns: collapse *_x / *_y into single fields
            for base in [
                "pop_total",
                "median_age",
                "median_income",
                "labor_force",
                "unemployed",
                "unemployment_rate",
            ]:
                col_x = f"{base}_x"
                col_y = f"{base}_y"
                if col_x in panel.columns or col_y in panel.columns:
                    panel[base] = panel.get(col_x).combine_first(panel.get(col_y))
                    # Drop raw suffixed columns to avoid confusion
                    drop_cols = []
                    if col_x in panel.columns:
                        drop_cols.append(col_x)
                    if col_y in panel.columns:
                        drop_cols.append(col_y)
                    if drop_cols:
                        panel = panel.drop(columns=drop_cols)


            # -----------------------------------------------------------
            # Clean ACS columns: fill any remaining NaNs with 0
            # -----------------------------------------------------------
            acs_columns = [
                "pop_total",
                "median_age",
                "median_income",
                "labor_force",
                "unemployed",
                "unemployment_rate",
                "poverty_rate",
            ]

            for col in acs_columns:
                if col in panel.columns:
                    panel[col] = panel[col].fillna(0)


            # Debug to confirm merged rows
            print("\n--- DEBUG: Sample merged panel rows with ACS ---")
            print(
                panel[
                    [
                        "grid_id",
                        "CENSUS_TRACT_GEOID",
                        "pop_total",
                        "median_income",
                        "unemployment_rate",
                    ]
                ]
                .dropna(
                    how="all",
                    subset=["pop_total", "median_income", "unemployment_rate"],
                )
                .head(10)
            )
            print("-----------------------------------------------\n")

    # ---------------------------------------------------------------
    # 6. Add lag features (1–3 weeks)
    # ---------------------------------------------------------------
    for k in [1, 2, 3]:
        panel[f"lag{k}"] = panel.groupby("grid_id")["crime_count"].shift(k).fillna(0)



    # ---------------------------------------------------------------
    # Spatial Neighbor Crime Features
    # ---------------------------------------------------------------
    # This builds neighbor-based features:
    #  - neighbor_crime_1wk : average crime_count of all adjacent bins last week
    #  - neighbor_crime_4wk : average crime_count for the last 4 weeks
    #  - neighbor_crime_8wk : average crime_count for the last 8 weeks

    print("Computing spatial neighbor crime features...")

    # Step 1: Create neighbor coordinate shifts (8 directions)
    neighbor_shifts = [
        (1, 0), (-1, 0),    # North, South
        (0, 1), (0, -1),    # East, West
        (1, 1), (1, -1),    # NE, SE
        (-1, 1), (-1, -1)   # NW, SW
    ]

    # Base columns for merging
    base = panel[["grid_id", "lat_bin", "lon_bin", "week_start"]].copy()

    # Step 2: Create a mapping for neighbor coordinates
    all_neighbors = []

    for dlat, dlon in neighbor_shifts:
        temp = base.copy()
        temp["n_lat_bin"] = temp["lat_bin"] + dlat
        temp["n_lon_bin"] = temp["lon_bin"] + dlon
        all_neighbors.append(temp)

    neighbor_map = pd.concat(all_neighbors, ignore_index=True)

    # Step 3: Merge neighbors with crime data
    merged_neighbors = neighbor_map.merge(
        panel[["lat_bin", "lon_bin", "week_start", "crime_count"]],
        left_on=["n_lat_bin", "n_lon_bin", "week_start"],
        right_on=["lat_bin", "lon_bin", "week_start"],
        how="left"
    )

    # Step 4: Aggregate to mean neighbor crime for each (grid_id, week)
    neighbor_weekly = (
        merged_neighbors.groupby(["grid_id", "week_start"])["crime_count"]
        .mean()
        .reset_index()
        .rename(columns={"crime_count": "neighbor_crime_raw"})
    )

    # Step 5: Merge back into panel
    panel = panel.merge(neighbor_weekly, on=["grid_id", "week_start"], how="left")

    panel["neighbor_crime_raw"] = panel["neighbor_crime_raw"].fillna(0)

    # Step 6: Create temporal neighbor features
    panel["neighbor_crime_1wk"] = (
        panel.groupby("grid_id")["neighbor_crime_raw"].shift(1).fillna(0)
    )
    panel["neighbor_crime_4wk"] = (
        panel.groupby("grid_id")["neighbor_crime_raw"]
        .rolling(4, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    panel["neighbor_crime_8wk"] = (
        panel.groupby("grid_id")["neighbor_crime_raw"]
        .rolling(8, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    print("Spatial neighbor crime features added.")


    # ---------------------------------------------------------------
    # 7. Add geographic centroids (inverse of grid bin)
    # ---------------------------------------------------------------
    panel["lat"] = panel["lat_bin"] * grid_step
    panel["lon"] = panel["lon_bin"] * grid_step



    # ---------------------------------------------------------------
    # Attach VIIRS nightlight values from night_light.tif
    # ---------------------------------------------------------------
    # The raster should be located at: data/external/night_light.tif
    nightlight_path = Path("data/external/night_light.tif")

    if nightlight_path.exists():
        print(f"Sampling nightlight from {nightlight_path}")

        # Sample only once per grid_id to avoid wasteful repeated sampling
        grid_centroids = (
            panel[["grid_id", "lat", "lon"]]
            .drop_duplicates(subset=["grid_id"])
            .reset_index(drop=True)
        )

        # Coordinates must be (lon, lat)
        coords = list(zip(grid_centroids["lon"], grid_centroids["lat"]))

        with rasterio.open(nightlight_path) as src:
            # sample() returns one array per coordinate (typically shape (1,))
            samples = list(src.sample(coords))

        night_vals = []
        for s in samples:
            if s is None:
                night_vals.append(float("nan"))
            else:
                try:
                    night_vals.append(float(s[0]))
                except Exception:
                    night_vals.append(float("nan"))

        grid_centroids["nightlight"] = night_vals

        # Merge single nightlight value per grid back into all weeks
        panel = panel.merge(
            grid_centroids[["grid_id", "nightlight"]],
            on="grid_id",
            how="left",
        )

        panel["nightlight"] = panel["nightlight"].fillna(0.0)

    else:
        print("WARNING: data/external/night_light.tif not found; setting nightlight=0.")
        panel["nightlight"] = 0.0



    # ---------------------------------------------------------------
    # 8. Add intervention/context features
    # If ACS demographics are present, we back-fill income/pop_density from them.
    # Otherwise, we keep the original placeholder values.
    # ---------------------------------------------------------------
    panel["poi_density"] = 0.0
    # panel["nightlight"] = 0.0

    # If population is available, we can at least use it as a crude population scale.
    if "pop_total" in panel.columns:
        panel["pop_density"] = panel["pop_total"]
    else:
        panel["pop_density"] = 0.0

    # Ensure poverty_rate exists even if ACS failed / was skipped
    if "poverty_rate" not in panel.columns:
        panel["poverty_rate"] = 0.0

    # weather_temp should already be set from temperature.csv;
    # only default to 0.0 if something went wrong earlier.
    if "weather_temp" not in panel.columns:
        panel["weather_temp"] = 0.0

    panel["events_count"] = 0.0



    # -----------------------------------------------------------
    # Pedestrian activity index (relative, around 1.0)
    #   - spatial component: pop_density + nightlight
    #   - temporal component: weekly temperature
    # -----------------------------------------------------------

    # Avoid division by zero
    eps = 1e-6

    # Spatial scaling: per grid (use one value per grid_id)
    # We min-max scale across grids so high pop / high nightlight → values near 1.
    grid_stats = panel.groupby("grid_id")[["pop_density", "nightlight"]].first().copy()

    # Min-max scale each static variable
    for col in ["pop_density", "nightlight"]:
        col_min = grid_stats[col].min()
        col_max = grid_stats[col].max()
        if col_max - col_min < eps:
            grid_stats[col + "_scaled"] = 0.0
        else:
            grid_stats[col + "_scaled"] = (grid_stats[col] - col_min) / (
                    col_max - col_min + eps
            )

    # Combine into a single spatial score in [0, 1]
    grid_stats["ped_spatial"] = 0.5 * grid_stats["pop_density_scaled"] + 0.5 * grid_stats["nightlight_scaled"]

    # Merge spatial component back to panel
    panel = panel.merge(
        grid_stats[["ped_spatial"]], left_on="grid_id", right_index=True, how="left"
    )

    # Temporal scaling: per week, based on weather_temp
    week_temp = panel.groupby("week_start")["weather_temp"].first().copy()
    t_min = week_temp.min()
    t_max = week_temp.max()
    if t_max - t_min < eps:
        week_temp_scaled = pd.Series(0.5, index=week_temp.index)
    else:
        week_temp_scaled = (week_temp - t_min) / (t_max - t_min + eps)

    # Map back to panel and form a factor roughly in [0.7, 1.3]
    temp_factor = 0.7 + 0.6 * week_temp_scaled
    panel = panel.merge(
        temp_factor.rename("ped_temporal"),
        left_on="week_start",
        right_index=True,
        how="left",
    )

    # Raw ped index = spatial × temporal
    panel["ped_raw"] = panel["ped_spatial"] * panel["ped_temporal"]

    # Normalize so median is 1.0 (keeps your intervention semantics)
    median_val = panel["ped_raw"].median()
    if median_val < eps:
        panel["ped_activity"] = 1.0
    else:
        panel["ped_activity"] = panel["ped_raw"] / median_val



    # ---------------------------------------------------------------
    # POLICE PATROLS: allocate officers to grid cells by year
    # ---------------------------------------------------------------

    # Approx. number of officers by year (interpolated using your numbers)
    officer_by_year = {
        2016: 1479,
        2017: 1513,
        2018: 1546,
        2019: 1580,
        2020: 1613,   # corrected peak
        2021: 1518,
        2022: 1423,
        2023: 1327,
        2024: 1232,
        2025: 1137,
    }

    # Map each weekly frame to a year and assign a total force size
    panel["year"] = panel["week_start"].dt.year
    panel["officers_citywide"] = panel["year"].map(officer_by_year).fillna(1137)

    # If pop info is present, use it as proxy for spatial allocation.
    # If it's missing or zero, fall back to equal allocation.
    pop_by_week = panel.groupby("week_start")["pop_density"].transform("sum")
    pop_by_week = pop_by_week.replace(0, np.nan)

    patrol_share = panel["pop_density"] / pop_by_week
    patrol_share = patrol_share.fillna(
        1.0 / panel.groupby("week_start")["grid_id"].transform("count")
    )

    # Raw patrols = share of citywide officers that week
    panel["police_patrols"] = panel["officers_citywide"] * patrol_share

    # Normalize so median grid gets 1.0 → matches how other intervention features work
    raw_med = panel["police_patrols"].median()
    if raw_med > 0:
        panel["police_patrols"] = panel["police_patrols"] / raw_med
    else:
        panel["police_patrols"] = 1.0

    # Drop helper columns
    panel = panel.drop(columns=["year", "officers_citywide"], errors="ignore")



    # ---------------------------------------------------------------
    # 9. Save outputs
    # ---------------------------------------------------------------
    frames_csv = out / "frames.csv"
    panel.to_csv(frames_csv, index=False)

    latest_week = panel["week_start"].max()
    inference_frame = panel[panel["week_start"] == latest_week]
    inference_frame.to_csv(out / "inference_frame.csv", index=False)

    print(f"Saved:\n- {frames_csv}\n- {out / 'inference_frame.csv'}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=CLEVELAND_FEATURE_URL,
        help="Cleveland Crime_Incidents FeatureService query URL ",
    )
    parser.add_argument(
        "--outdir",
        default="data/processed",
        help="Output directory for frames.csv and inference_frame.csv",
    )
    parser.add_argument(
        "--grid_step",
        type=float,
        default=0.005,
        help="Grid step in degrees (~0.005 ≈ 0.5km).",
    )
    # Census API configuration (optional, Cleveland-only)
    parser.add_argument(
        "--census_api_key",
        help="Census API key for pulling ACS tract demographics (optional).",
    )
    parser.add_argument(
        "--census_year",
        type=int,
        default=2023,
        help="ACS 5-year year (e.g., 2023) for tract demographics.",
    )
    parser.add_argument(
        "--census_state_fips",
        default="39",  # Ohio
        help="State FIPS code for ACS (default=39 for Ohio).",
    )
    parser.add_argument(
        "--census_county_fips",
        default="035",  # Cuyahoga County
        help="County FIPS code for ACS (default=035 for Cuyahoga).",
    )
    args = parser.parse_args()

    # Cleveland API load → harmonized DATE OCC / LAT / LON / CENSUS_TRACT_GEOID
    df = load_cleveland_crime_from_api(api_url=args.input)

    # Extract all unique GEOIDs from Cleveland API
    unique_geoids = (
        df["CENSUS_TRACT_GEOID"]
        .dropna()
        .astype(str)
        .str.zfill(11)
        .unique()
    )

    tracts_df = load_acs_tracts_by_geoid(
        target_geoids=unique_geoids,
        api_key=args.census_api_key,
        year=args.census_year,
    )

    build_panel(df, args.outdir, grid_step=args.grid_step, tracts_df=tracts_df)


if __name__ == "__main__":
    main()

# Run: python src/preprocessing.py --outdir data/processed --census_api_key 6044b1b3aa5148f4029a4fb97f072aa0e645f7cc