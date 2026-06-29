import os
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

# This code is adapted from: https://github.com/pajevicnina/inspire1-seg


def calculate_statistics(work_dir, classified_shp, train_shp, val_shp):
    print("\n--- Starting Post-Processing & Statistics ---")
    
    # Paths for output files
    fixed_shp_out = os.path.join(work_dir, "Classified_Map_fixed.shp")
    area_csv_out = os.path.join(work_dir, "class_area_percentage.csv")
    count_csv_out = os.path.join(work_dir, "number_of_polygons.csv")

    # =========================================================
    # PART 1: Clean Shapefile and Calculate Area Percentages
    # =========================================================
    print("1. Cleaning classified shapefile and calculating coverage areas...")
    gdf = gpd.read_file(classified_shp)
    
    # Filter out absolute NoData background.
    # Since your data is standardized between 0 and 1, any valid polygon will have a mean >= 0.0.
    # Backgrounds/Shadows that received -9999.0 or -1.0 will be safely removed here.
    gdf_clean = gdf[gdf['meanB0'] >= 0.0].copy()
    
    # Save the cleaned shapefile for final mapping in QGIS
    gdf_clean.to_file(fixed_shp_out)
    print(f" -> Cleaned map saved to: {os.path.basename(fixed_shp_out)}")
    
    # Calculate geographical area for each polygon 
    # (Make sure your CRS is projected, e.g., UTM, so area is in square meters!)
    gdf_clean['area'] = gdf_clean['geometry'].area
    
    # Aggregate total area by predicted class
    class_area = gdf_clean.groupby('Predicted')['area'].sum().reset_index()
    
    # Calculate percentage
    total_area = class_area['area'].sum()
    class_area['percent'] = (class_area['area'] / total_area) * 100
    
    # Save area stats to CSV
    class_area.to_csv(area_csv_out, index=False)
    print(f" -> Area statistics saved to: {os.path.basename(area_csv_out)}")

    # =========================================================
    # PART 2: Count Training and Validation Polygons
    # =========================================================
    print("\n2. Counting training and validation sample distributions...")
    train_gdf = gpd.read_file(train_shp)
    val_gdf = gpd.read_file(val_shp)
    
    # Get unique classes from the 'Final_Clas' column in the training set
    unique_classes = sorted(train_gdf['Final_Clas'].unique())
    
    counts = []
    for cl in unique_classes:
        # Count occurrences in train and val datasets
        train_count = np.count_nonzero(train_gdf['Final_Clas'] == cl)
        val_count = np.count_nonzero(val_gdf['Final_Clas'] == cl)
        
        counts.append({
            'Class_ID': cl,
            'Training_Polygons': train_count,
            'Validation_Polygons': val_count,
            'Total_Polygons': train_count + val_count
        })
        
    count_df = pd.DataFrame(counts)
    count_df.to_csv(count_csv_out, index=False)
    
    print(f" -> Polygon distribution saved to: {os.path.basename(count_csv_out)}")
    print("\n✅ Post-Processing Complete!")


def main():
    # Setup Base Paths exactly like in the CNN and RF scripts
    curr_dir = Path.cwd()
    base_path = curr_dir.parent.parent
    
    # 1. Define where the Random Forest saved the split data
    rf_dir = os.path.join(base_path, "Data", "10_Landcover_Classification", "RF", "v3")
    train_shp = os.path.join(rf_dir, "rf_train_split.shp")
    val_shp = os.path.join(rf_dir, "rf_val_split.shp")
    
    # 2. Define where your fully classified map is located
    classified_shp = os.path.join(rf_dir, "Classified_Full_Map_RF.shp") 
    
    # Safety Check: Did OTB actually output the full map yet?
    if not os.path.exists(classified_shp):
        print(f"❌ ERROR: Could not find {classified_shp}.")
        print("Please ensure you ran the OTB VectorClassifier on your FULL image/shapefile first.")
        return
        
    if not os.path.exists(train_shp) or not os.path.exists(val_shp):
        print("❌ ERROR: Could not find the train/val split files. Run the RF training script first.")
        return
        
    # Run the statistics engine
    calculate_statistics(rf_dir, classified_shp, train_shp, val_shp)


if __name__ == "__main__":
    main()