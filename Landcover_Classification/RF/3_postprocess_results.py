import os
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

# This code is adapted from: https://github.com/pajevicnina/inspire1-seg


def calculate_statistics(work_dir, classified_shp, train_path, validation_path, test_path):
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
    
    gdf_clean = gdf[(gdf["meanB0"] >= 0.0) & (gdf["Predicted"] >= 0)].copy()
    
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
    train_gdf = gpd.read_file(train_path)
    validation_gdf = gpd.read_file(validation_path)
    test_gdf = gpd.read_file(test_path)
        
    # Get unique classes from the 'Final_Clas' column in the training set
    all_classes = sorted(
        set(train_gdf["Final_Clas"])
        | set(validation_gdf["Final_Clas"])
        | set(test_gdf["Final_Clas"])
    )
    
    counts = []
    
    for class_id in all_classes:
        train_count = int(
            np.count_nonzero(
                train_gdf["Final_Clas"] == class_id
            )
        )
        validation_count = int(
            np.count_nonzero(
                validation_gdf["Final_Clas"] == class_id
            )
        )
        test_count = int(
            np.count_nonzero(
                test_gdf["Final_Clas"] == class_id
            )
        )

        counts.append(
            {
                "Class_ID": class_id,
                "Training_Polygons": train_count,
                "Validation_Polygons": validation_count,
                "Test_Polygons": test_count,
                "Total_Polygons": (
                    train_count
                    + validation_count
                    + test_count
                ),
            }
        )
        
    count_df = pd.DataFrame(counts)
    count_df.to_csv(count_csv_out, index=False)
    
    print(f" -> Polygon distribution saved to: {os.path.basename(count_csv_out)}")
    print("\n✅ Post-Processing Complete!")


def main():
    # Setup Base Paths exactly like in the CNN and RF scripts
    curr_dir = Path.cwd()
    base_path = curr_dir.parent.parent
    
    # 1. Define where the Random Forest saved the split data
    split_dir = os.path.join(
        base_path,
        "Data",
        "9_Training_Data",
        "spatial_split_v4",
    )

    train_path = os.path.join(split_dir, "train.gpkg")
    validation_path = os.path.join(
        split_dir,
        "validation.gpkg",
    )
    test_path = os.path.join(split_dir, "test.gpkg")
        
    # 2. Define where your fully classified map is located
    rf_dir = os.path.join(
        base_path,
        "Data",
        "10_Landcover_Classification",
        "RF",
        "v5",
    )
    classified_shp = os.path.join(rf_dir, "classified_full_map_rf.shp") 
    
    # Safety Check: Did OTB actually output the full map yet?
    if not os.path.exists(classified_shp):
        print(f"❌ ERROR: Could not find {classified_shp}.")
        print("Please ensure you ran the OTB VectorClassifier on your FULL image/shapefile first.")
        return
        
    if not os.path.exists(train_path) or not os.path.exists(validation_path) or not os.path.exists(test_path):
        print("❌ ERROR: Could not find the train/validation/test split files. Run your split script first.")
        return
        
    # Run the statistics engine
    calculate_statistics(rf_dir, classified_shp, train_path, validation_path, test_path)


if __name__ == "__main__":
    main()