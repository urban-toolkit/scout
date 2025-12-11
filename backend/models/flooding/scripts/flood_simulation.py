import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.geometry import mapping
from rasterio.mask import mask
import numpy as np
import pandas as pd
import os


def simulate_flood_projection(
    region,
    output,
    year="2020 - 2040",
    use_NBS_classes=[
        "Bioswales/Infiltration trenches",
        "Permeable pavements",
        "Retention ponds",
        "Infiltration trench",
        "Bioswales",
        "Constructed wetlands",
    ],
):
    # print(region, output, year, use_NBS_classes)
    dict_use_NBS_classes = {
        21: "Bioswales/Infiltration trenches",
        31: "Permeable pavements",
        43: "Retention ponds",
        52: "Infiltration trench",
        71: "Bioswales",
        81: "Bioswales",
        90: "Constructed wetlands",
        95: "Constructed wetlands",
    }

    # convert list to set: use_NBS_classes
    use_NBS_classes = set(use_NBS_classes)

    # Codes where we use NBS flood values
    use_NBS_classes_ = {
        code for code, name in dict_use_NBS_classes.items()
        if name in use_NBS_classes
    }

    year_dict = {
        "2020 - 2040": "2020_2040",
        "2050 - 2080": "2050_2080",
        "2080 - 2100": "2080_2100",
    }

    year = year_dict.get(year)

    mask_path       = "./models/flooding/data_substitutes/NBS_others_5m_4326_cropped_cleaned_resampled.tif"
    nbs_flood_path  = f"./models/flooding/data_substitutes/{year}_NbS_4326_cropped.tif"
    nonbs_flood_path= f"./models/flooding/data_substitutes/{year}_noNbS_4326_cropped.tif"
    output_path     = f"./data/served/raster/{output}.tif"

    with rasterio.open(mask_path) as src_mask, \
         rasterio.open(nbs_flood_path) as src_nbs, \
         rasterio.open(nonbs_flood_path) as src_nonbs:

        # all rasters must be on same grid
        assert src_mask.crs == src_nbs.crs == src_nonbs.crs
        assert src_mask.transform == src_nbs.transform == src_nonbs.transform

        # ---------------------------------------------------------------------
        # 1) Crop by bbox or GeoJSON
        # ---------------------------------------------------------------------
        if isinstance(region, (list, tuple)) and len(region) == 4:
            # region is a bbox: (min_lon, min_lat, max_lon, max_lat)
            min_lon, min_lat, max_lon, max_lat = region

            window = from_bounds(
                min_lon, min_lat, max_lon, max_lat,
                transform=src_mask.transform,
            )

            mask_data_   = src_mask.read(1, window=window)
            nbs_data_    = src_nbs.read(1, window=window)
            nonbs_data_  = src_nonbs.read(1, window=window)

            out_transform = src_mask.window_transform(window)
            out_height, out_width = mask_data_.shape

        else:
            # region is assumed to be a GeoJSON name (without extension)
            geojson_path = f"./data/boundaries/{region}.geojson"
            gdf = gpd.read_file(geojson_path)

            if gdf.crs != src_mask.crs:
                gdf = gdf.to_crs(src_mask.crs)

            geoms = [mapping(geom) for geom in gdf.geometry]

            # mask returns (bands, rows, cols)
            mask_arr, out_transform = mask(
                src_mask, geoms, crop=True, nodata=src_mask.nodata
            )
            nbs_arr, _ = mask(
                src_nbs, geoms, crop=True, nodata=src_nbs.nodata
            )
            nonbs_arr, _ = mask(
                src_nonbs, geoms, crop=True, nodata=src_nonbs.nodata
            )

            mask_data_  = mask_arr[0]
            nbs_data_   = nbs_arr[0]
            nonbs_data_ = nonbs_arr[0]

            print("Cropped shapes:", mask_data_.shape, nbs_data_.shape, nonbs_data_.shape)

            out_height, out_width = mask_data_.shape

        # ---------------------------------------------------------------------
        # 2) Combine logic: NBS classes vs noNBS
        # ---------------------------------------------------------------------
        cond_use_NBS = np.isin(mask_data_, list(use_NBS_classes_))
        combined = np.where(cond_use_NBS, nbs_data_, nonbs_data_)

        print("Combined flood raster shape:", combined.shape)

        # ---------------------------------------------------------------------
        # 3) Clean up nodata: replace huge negative with 0
        # ---------------------------------------------------------------------
        flood_nodata = src_nbs.nodata  # typically -1.797e308

        if flood_nodata is not None:
            # Turn that nodata sentinel into 0 so web client can treat 0 as "no flood"
            combined = np.where(combined == flood_nodata, np.nan, combined)

        # ---------------------------------------------------------------------
        # 4) Build output profile
        # ---------------------------------------------------------------------
        profile = src_nbs.profile.copy()
        profile.update(
            dtype=combined.dtype,
            height=out_height,
            width=out_width,
            transform=out_transform,
            nodata=0.0,  # we now treat 0 as nodata / no flood
        )

        metric_path = f"./data/served/metric/{output}.csv"

        median_val = np.nanmedian(combined)
        mean_val   = np.nanmean(combined)
        max_val    = np.nanmax(combined)
        min_val    = np.nanmin(combined)
        stddev_val = np.nanstd(combined)

        # ensure folder exists
        os.makedirs(os.path.dirname(metric_path), exist_ok=True)

        # save to CSV
        df = pd.DataFrame([{
            "median flood depth": median_val,
            "mean flood depth": mean_val,
            # "flood depth (max)": max_val,
            # "flood depth (min)": min_val,
            # "flood depth stddev": stddev_val
        }])

        df.to_csv(metric_path, index=False)

    # -------------------------------------------------------------------------
    # 5) Write output raster
    # -------------------------------------------------------------------------
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(combined, 1)

    print("Saved combined flood raster to:", output_path)