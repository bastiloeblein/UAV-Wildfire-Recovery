import os
import pickle
import numpy as np
import rasterio
import geopandas as gpd
from keras.models import load_model
from pathlib import Path

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
    shapefile_path = os.path.join(labels_dir, "final_training_data_qgis.shp") 

    # Define Output Dir (Where the CNN model is saved)
    result_dir = os.path.join(base_path, "Data", "10_Landcover_Classification", "CNN")
    
    # --- CONFIGURABLE EXPERIMENT PARAMETERS ---
    target_window_size = (128, 128)
    selected_channels = [0, 1, 2, 3, 4, 5, 6]
    
    # Reconstruct the exact folder name generated during training
    ch_suffix = "".join(map(str, selected_channels))
    model_folder = os.path.join(result_dir, f"cnn_results_w{target_window_size[0]}_ch{ch_suffix}")
    
    model_path = os.path.join(model_folder, "cnn_model.h5")
    encoder_path = os.path.join(model_folder, "label_encoder.pkl")
    
    # Final output map that will be loaded into QGIS and evaluated
    output_shp = os.path.join(model_folder, "Classified_Full_Map_CNN.shp")

    # =========================================================
    # 2. LOAD DEPENDENCIES AND DATA
    # =========================================================
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}. Run training first!")
        return

    print("Loading LabelEncoder to map predictions back to original Class IDs...")
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    print("Loading trained CNN model...")
    model = load_model(model_path)

    print(f"Loading massive vector dataset: {os.path.basename(shapefile_path)}...")
    gdf = gpd.read_file(shapefile_path)

    print(f"Loading raster image: {os.path.basename(image_path)}...")
    src = rasterio.open(image_path)
    image = src.read()
    image = np.moveaxis(image, 0, -1)  # Convert to Keras format (H, W, C)

    # Match the training NoData value
    image[image == -9999.0] = -1.0
    
    print(f"Extracting selected channels: {selected_channels}")
    image = image[:, :, selected_channels]
    num_channels = image.shape[-1]
    
    # Create empty column for CNN predictions (Named 'Predicted' for the eval script)
    gdf['Predicted'] = -1
    
    # Batch processing variables to prevent RAM overflow
    batch_size = 10000
    X_batch = []
    index_batch = []
    patch_radius = target_window_size[0] // 2

    print(f"\nStarting predictions for {len(gdf)} polygons ...")

    # =========================================================
    # 3. ITERATE THROUGH POLYGONS (STRICT SCALE EXTRACTION)
    # =========================================================
    for idx, row in gdf.iterrows():
        geom = row['geometry']
        if geom is None:
            continue

        # Extract window around the CENTROID to preserve physical scale.
        centroid = geom.centroid
        c_row, c_col = rasterio.transform.rowcol(src.transform, centroid.x, centroid.y)

        # Define ideal window boundaries
        y_min = c_row - patch_radius
        y_max = c_row + patch_radius
        x_min = c_col - patch_radius
        x_max = c_col + patch_radius

        # Create a blank patch filled with NoData (-1.0)
        patch = np.full((target_window_size[0], target_window_size[1], num_channels), -1.0, dtype=image.dtype)

        # Calculate valid image boundaries (to prevent crashing at the image edge)
        img_y_min, img_y_max = max(0, y_min), min(image.shape[0], y_max)
        img_x_min, img_x_max = max(0, x_min), min(image.shape[1], x_max)

        # Calculate where to place the valid image data inside the blank patch
        patch_y_min = img_y_min - y_min
        patch_y_max = target_window_size[0] - (y_max - img_y_max)
        patch_x_min = img_x_min - x_min
        patch_x_max = target_window_size[1] - (x_max - img_x_max)

        # If the polygon is somewhat inside the image, fill the patch and append
        if img_y_max > img_y_min and img_x_max > img_x_min:
            patch[patch_y_min:patch_y_max, patch_x_min:patch_x_max] = image[img_y_min:img_y_max, img_x_min:img_x_max]
            X_batch.append(patch)
            index_batch.append(idx)

        # Trigger prediction when batch is full or at the very last polygon
        if len(X_batch) == batch_size or idx == len(gdf) - 1:
            if len(X_batch) > 0:
                X_array = np.array(X_batch)
                
                # Execute prediction
                preds = model.predict(X_array, verbose=0)
                pred_classes = np.argmax(preds, axis=1)

                # Decode the Keras integer back to your real Class ID (e.g., 20)
                real_class_ids = label_encoder.inverse_transform(pred_classes)
                
                # Write back to GeoDataFrame
                for i, real_class_id in enumerate(real_class_ids):
                    gdf.at[index_batch[i], 'Predicted'] = real_class_id

            # Clear lists for the next batch
            X_batch = []
            index_batch = []
            print(f" -> Processed {idx + 1} / {len(gdf)} polygons...")

    src.close()

    # =========================================================
    # 4. EXPORT
    # =========================================================
    print(f"\nSaving classified map to: {os.path.basename(output_shp)}...")
    gdf.to_file(output_shp)
    print("--- DONE! YOUR NEW CNN LANDCOVER MAP IS READY! ---")

if __name__ == "__main__":
    main()