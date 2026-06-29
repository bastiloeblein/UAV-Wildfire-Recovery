import os
import geopandas as gpd
from rasterstats import zonal_stats
import numpy as np
from pathlib import Path

notebook_dir = Path.cwd()
base_path = notebook_dir.parent.parent / "Data" 
base_path_str = str(base_path)
print(f"Base path automatically set to: {base_path_str}")


work_dir = os.path.join(base_path, "8_Images_with_Indices")
image_path = os.path.join(work_dir, "250821_7_channel.tif")
label_dir = os.path.join(base_path, "9_Training_Data")
input_shp_path = os.path.join(label_dir, "final_training_data_qgis_v2.shp") 
output_shp_path = os.path.join(label_dir, "training_data_final.shp")

print(f"Loading shapefile: {os.path.basename(input_shp_path)}...")
gdf = gpd.read_file(input_shp_path)    

print("Extracting zonal statistics for all 7 bands...")
    
# Loop through bands 1 to 7 (rasterio uses 1-based indexing)
for tif_band in range(1, 8):
    col_idx = tif_band - 1  # For column names (B0 to B6)
    print(f" -> Processing Band {tif_band}/7 (creates meanB{col_idx} & varB{col_idx})...")
    
    # Extract mean and standard deviation (ignoring -9999.0 background)
    stats = zonal_stats(gdf, image_path, band=tif_band, stats=['mean', 'std'], nodata=-9999.0)
    
    # Write Mean to dataframe (use np.nan if polygon is empty/invalid)
    gdf[f'meanB{col_idx}'] = [s['mean'] if s['mean'] is not None else np.nan for s in stats]
    
    # Calculate and write Variance (Standard Deviation squared)
    gdf[f'varB{col_idx}'] = [(s['std'] ** 2) if s['std'] is not None else np.nan for s in stats]

# Calculate overall Brightness using the RGB channels (B0, B1, B2)
# print("Calculating engineered feature: Brightness...")
# gdf['Brightness'] = gdf['meanB0'] + gdf['meanB1'] + gdf['meanB2']

print(f"Saving updated shapefile to: {output_shp_path}...")
gdf.to_file(output_shp_path)

print("\n=========================================")
print("✅ FEATURE EXTRACTION COMPLETE!")
print("=========================================")