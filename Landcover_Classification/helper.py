import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit


def set_global_seeds(seed=42):
    """Sets all global seeds (for reproducibility)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"🌱 Global Seed set to {seed}.")


def split_polygons_leakage_free(gdf, label_column='Final_Clas', test_size=0.2, seed=42, grid_size=50):
    """
    Splits shapefile into Train/Test using a spatial grid to prevent data leakage.
    Polygons in the same geographic grid cell will ALWAYS be kept together in either
    Train or Test, preventing overlapping CNN patches.
    
    Args:
        grid_size (int): Size of the grid cells in meters (assuming your CRS is metric, like UTM).
                         Set this to at least your patch size in meters to prevent overlap.
    """
    print("\n🪓 Performing SPATIAL Polygon Split...")
    
    # 1. Clean: Only keep labeled polygons
    gdf = gdf.dropna(subset=[label_column])
    gdf = gdf[(gdf[label_column] != -1) & (gdf[label_column] != -1.0) & (gdf[label_column] != "")]
    
    # 2. Extract Centroids to determine spatial position
    centroids = gdf.geometry.centroid
    
    # 3. Create a spatial grid
    # We assign each polygon a "Group ID" based on which grid cell it falls into.
    # The math discretizes the continuous X/Y coordinates into discrete grid bins.
    min_x, min_y, max_x, max_y = gdf.total_bounds
    
    # Create grid cell IDs
    gdf['grid_x'] = np.floor((centroids.x - min_x) / grid_size).astype(int)
    gdf['grid_y'] = np.floor((centroids.y - min_y) / grid_size).astype(int)
    
    # Combine X and Y into a unique string ID for the grid cell
    gdf['spatial_group'] = gdf['grid_x'].astype(str) + "_" + gdf['grid_y'].astype(str)
    
    print(f" -> Created {gdf['spatial_group'].nunique()} spatial clusters of size {grid_size}x{grid_size}m.")

    # 4. Perform Group-based Split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    
    # gss.split returns indices for the split
    train_idx, test_idx = next(gss.split(gdf, groups=gdf['spatial_group']))
    
    train_gdf = gdf.iloc[train_idx].copy()
    test_gdf = gdf.iloc[test_idx].copy()
    
    # Cleanup temporary columns
    train_gdf = train_gdf.drop(columns=['grid_x', 'grid_y', 'spatial_group'])
    test_gdf = test_gdf.drop(columns=['grid_x', 'grid_y', 'spatial_group'])

    # 5. Print Distributions
    print(f"\n -> Training: {len(train_gdf)} Polygons")
    print("--- TRAIN: CLASS DISTRIBUTION ---")
    counts = train_gdf[label_column].value_counts(dropna=False)
    for cls_name, count in counts.items():
        print(f"Class '{cls_name}': {count} polygons")
    print("--------------------------------------")

    print(f"\n -> Testing:  {len(test_gdf)} Polygons")
    print("--- TEST: CLASS DISTRIBUTION ---")
    counts = test_gdf[label_column].value_counts(dropna=False)
    for cls_name, count in counts.items():
        print(f"Class '{cls_name}': {count} polygons")
    print("--------------------------------------\n")
            
    return train_gdf, test_gdf

# def split_polygons_leakage_free(gdf, label_column='Final_Clas', test_size=0.2, seed=42):
#     """
#     Splits shapefile based on polygons. 
#     """
#     # 1. Clean: Only keep labeled polygons
#     gdf = gdf.dropna(subset=[label_column])
#     gdf = gdf[(gdf[label_column] != -1) & (gdf[label_column] != -1.0) & (gdf[label_column] != "")]
    
#     # 2. Do split (with stratification for class balance)
#     train_gdf, test_gdf = train_test_split(
#         gdf, test_size=test_size, random_state=seed, stratify=gdf[label_column]
#     )
    
#     print(f"🪓 Polygon split done:")
#     print(f" -> Training: {len(train_gdf)} Polygons")
#     print("\n--- TRAIN: CLASS DISTRIBUTION ---")
#     counts = train_gdf['Final_Clas'].value_counts(dropna=False)
#     for cls_name, count in counts.items():
#         print(f"Class '{cls_name}': {count} polygons")
#     print("--------------------------------------\n")

#     print(f" -> Testing:  {len(test_gdf)} Polygons")
#     print("\n--- TEST: CLASS DISTRIBUTION ---")
#     counts = test_gdf['Final_Clas'].value_counts(dropna=False)
#     for cls_name, count in counts.items():
#         print(f"Class '{cls_name}': {count} polygons")
#     print("--------------------------------------\n")
            
#     return train_gdf, test_gdf