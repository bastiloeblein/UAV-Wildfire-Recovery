import os
import pickle
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box, mapping
from keras.models import load_model
from pathlib import Path


def repair_geometry(geom):
    """Repair invalid polygon geometries before extracting a patch."""
    if geom is None or geom.is_empty:
        return None

    if not geom.is_valid:
        repaired = geom.buffer(0)
        if repaired is not None and not repaired.is_empty:
            geom = repaired

    if geom.is_empty:
        return None

    return geom


def find_patch_center(geom, src, image):
    """Find one stable patch centre inside the polygon and raster extent."""
    raster_extent = box(
        src.bounds.left,
        src.bounds.bottom,
        src.bounds.right,
        src.bounds.top,
    )

    geom_in_raster = geom.intersection(raster_extent)
    if geom_in_raster.is_empty or geom_in_raster.area == 0:
        return None, None, "outside_raster"

    centre_point = geom.centroid
    source = "centroid"

    # Concave polygons can have a centroid outside the polygon.
    if not geom_in_raster.covers(centre_point):
        centre_point = geom_in_raster.representative_point()
        source = "inside_point"

    c_row, c_col = rasterio.transform.rowcol(
        src.transform,
        centre_point.x,
        centre_point.y,
    )

    # Keep the centre inside the raster grid.
    c_row = int(np.clip(c_row, 0, image.shape[0] - 1))
    c_col = int(np.clip(c_col, 0, image.shape[1] - 1))

    if image[c_row, c_col, 0] != -1.0:
        return c_row, c_col, source

    # Search for the closest valid raster pixel inside the polygon.
    minx, miny, maxx, maxy = geom_in_raster.bounds
    row_a, col_a = rasterio.transform.rowcol(
        src.transform,
        minx,
        maxy,
    )
    row_b, col_b = rasterio.transform.rowcol(
        src.transform,
        maxx,
        miny,
    )

    r0 = max(0, min(row_a, row_b))
    r1 = min(image.shape[0], max(row_a, row_b) + 1)
    c0 = max(0, min(col_a, col_b))
    c1 = min(image.shape[1], max(col_a, col_b) + 1)

    if r1 > r0 and c1 > c0:
        window = Window(c0, r0, c1 - c0, r1 - r0)
        polygon_mask = geometry_mask(
            [mapping(geom_in_raster)],
            out_shape=(r1 - r0, c1 - c0),
            transform=src.window_transform(window),
            invert=True,
        )
        valid_pixels = (
            polygon_mask
            & (image[r0:r1, c0:c1, 0] != -1.0)
        )

        if valid_pixels.any():
            local_rows, local_cols = np.where(valid_pixels)
            global_rows = local_rows + r0
            global_cols = local_cols + c0
            distances = (
                (global_rows - c_row) ** 2
                + (global_cols - c_col) ** 2
            )
            nearest = int(np.argmin(distances))
            return (
                int(global_rows[nearest]),
                int(global_cols[nearest]),
                "nearest_valid",
            )

    # The CNN still receives the complete surrounding patch if the
    # polygon itself only contains NoData in the first channel.
    return c_row, c_col, "nodata_center"


def extract_patch(image, row, col, patch_size):
    """Extract one fixed-size patch and pad image borders with -1."""
    patch_height, patch_width = patch_size
    half_height = patch_height // 2
    half_width = patch_width // 2

    y_min = row - half_height
    y_max = y_min + patch_height
    x_min = col - half_width
    x_max = x_min + patch_width

    patch = np.full(
        (patch_height, patch_width, image.shape[-1]),
        -1.0,
        dtype=image.dtype,
    )

    img_y_min = max(0, y_min)
    img_y_max = min(image.shape[0], y_max)
    img_x_min = max(0, x_min)
    img_x_max = min(image.shape[1], x_max)

    if img_y_max <= img_y_min or img_x_max <= img_x_min:
        return None

    patch_y_min = img_y_min - y_min
    patch_y_max = patch_y_min + (img_y_max - img_y_min)
    patch_x_min = img_x_min - x_min
    patch_x_max = patch_x_min + (img_x_max - img_x_min)

    patch[
        patch_y_min:patch_y_max,
        patch_x_min:patch_x_max,
    ] = image[
        img_y_min:img_y_max,
        img_x_min:img_x_max,
    ]

    return patch


def remove_existing_shapefile(path):
    """Remove an old Shapefile together with its sidecar files."""
    stem = os.path.splitext(path)[0]
    for extension in [
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
        ".qix",
        ".fix",
    ]:
        candidate = stem + extension
        if os.path.exists(candidate):
            os.remove(candidate)


def main():
    # =========================================================
    # 1. SETUP PATHS 
    # =========================================================
    curr_dir = Path.cwd()
    base_path = curr_dir.parent.parent
    print(f"Base path automatically set to: {str(base_path)}")

    # Get Image
    data_dir = os.path.join(base_path, "Data", "8_Images_with_Indices")
    image_path = os.path.join(data_dir, "250821_7_channel.tif")

    # Get the FULL, unlabeled Shapefile to classify
    labels_dir = os.path.join(base_path, "Data", "9_Training_Data")
    shapefile_path = os.path.join(labels_dir, "training_data_final.shp") 

    # Define Output Dir
    result_dir = os.path.join(base_path, "Data", "10_Landcover_Classification", "CNN")
    
    # --- CONFIGURABLE EXPERIMENT PARAMETERS ---
    target_window_size = (128, 128)
    selected_channels = [0, 1, 2, 3, 4, 5, 6]
    
    # Reconstruct the exact folder name generated during training
    ch_suffix = "".join(map(str, selected_channels))
    model_folder = os.path.join(result_dir, f"v5_w{target_window_size[0]}_ch{ch_suffix}")
    
    model_path = os.path.join(model_folder, "cnn_model.h5")
    encoder_path = os.path.join(model_folder, "label_encoder.pkl")
    
    # Final output map that will be loaded into QGIS and evaluated
    output_shp = os.path.join(
        model_folder,
        "classified_full_map_cnn.shp",
    )

    # =========================================================
    # 2. LOAD DEPENDENCIES AND DATA
    # =========================================================
    for required_path in [model_path, encoder_path, shapefile_path, image_path]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required file not found: {required_path}")

    print("Loading LabelEncoder to map predictions back to original Class IDs...")
    with open(encoder_path, "rb") as file:
        label_encoder = pickle.load(file)

    print("Loading trained CNN model...")
    model = load_model(model_path)

    print(f"Loading massive vector dataset: {os.path.basename(shapefile_path)}...")
    gdf = gpd.read_file(shapefile_path)

    print(f"Loading raster image: {os.path.basename(image_path)}...")
    src = rasterio.open(image_path)

    if gdf.crs is not None and src.crs is not None and gdf.crs != src.crs:
        src.close()
        raise ValueError(f"CRS mismatch: vector={gdf.crs}, raster={src.crs}")

    image = src.read().astype(np.float32)
    image = np.moveaxis(image, 0, -1)
    image = np.nan_to_num(
        image,
        nan=-1.0,
        posinf=-1.0,
        neginf=-1.0,
    )
    image[image == -9999.0] = -1.0

    print(f"Extracting selected channels: {selected_channels}")
    image = image[:, :, selected_channels]

    # NaN makes missing predictions visible during processing.
    gdf["Predicted"] = np.nan
    gdf["PatchSrc"] = ""

    batch_size = 10000
    X_batch = []
    index_batch = []
    source_batch = []
    unusable_indices = []

    def predict_batch():
        nonlocal X_batch, index_batch, source_batch

        if not X_batch:
            return

        X_array = np.asarray(X_batch, dtype=np.float32)
        probabilities = model.predict(X_array, verbose=0)
        encoded_classes = np.argmax(probabilities, axis=1)
        class_ids = label_encoder.inverse_transform(encoded_classes)

        for dataframe_idx, class_id, source in zip(
            index_batch,
            class_ids,
            source_batch,
        ):
            gdf.at[dataframe_idx, "Predicted"] = int(class_id)
            gdf.at[dataframe_idx, "PatchSrc"] = source

        X_batch = []
        index_batch = []
        source_batch = []

    print(f"\nStarting predictions for {len(gdf)} polygons ...")

    # =========================================================
    # 3. ITERATE THROUGH POLYGONS 
    # =========================================================
    for position, (idx, row) in enumerate(gdf.iterrows(), start=1):
        geom = repair_geometry(row["geometry"])

        if geom is None:
            unusable_indices.append(idx)
            continue

        c_row, c_col, source = find_patch_center(
            geom,
            src,
            image,
        )

        if c_row is None or c_col is None:
            unusable_indices.append(idx)
            continue

        patch = extract_patch(
            image,
            c_row,
            c_col,
            target_window_size,
        )

        if patch is None:
            unusable_indices.append(idx)
            continue

        X_batch.append(patch)
        index_batch.append(idx)
        source_batch.append(source)

        if len(X_batch) >= batch_size:
            predict_batch()
            print(f" -> Processed {position} / {len(gdf)} polygons...")

    predict_batch()
    src.close()

    # No final map is written if any polygon is missing a prediction.
    missing_predictions = gdf["Predicted"].isna()
    if unusable_indices or missing_predictions.any():
        failed_indices = sorted(
            set(unusable_indices)
            | set(gdf.index[missing_predictions].tolist())
        )
        debug_shp = os.path.join(
            model_folder,
            "unclassified_polygons_debug.shp",
        )
        remove_existing_shapefile(debug_shp)
        gdf.loc[failed_indices].to_file(debug_shp)
        raise RuntimeError(
            f"Could not classify {len(failed_indices)} polygons. "
            f"They were saved to {debug_shp}. "
            "Fix empty geometries or polygons outside the raster and rerun."
        )

    gdf["Predicted"] = gdf["Predicted"].astype(int)

    allowed_classes = set(int(value) for value in label_encoder.classes_)
    unexpected_classes = set(gdf["Predicted"].unique()) - allowed_classes
    if unexpected_classes:
        raise RuntimeError(
            f"Unexpected predicted classes found: {unexpected_classes}"
        )

    if (gdf["Predicted"] == -1).any():
        raise RuntimeError("Predicted still contains -1 values.")

    # =========================================================
    # 4. EXPORT
    # =========================================================
    remove_existing_shapefile(output_shp)
    print(f"\nSaving classified map to: {os.path.basename(output_shp)}...")
    gdf.to_file(output_shp)

    print(f"Classified polygons: {len(gdf)} / {len(gdf)}")
    print("Patch centres used:")
    print(gdf["PatchSrc"].value_counts())
    print("Predicted classes:")
    print(gdf["Predicted"].value_counts().sort_index())
    print("--- DONE! NEW CNN LANDCOVER MAP IS READY! ---")


if __name__ == "__main__":
    main()
