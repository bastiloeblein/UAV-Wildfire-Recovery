import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def set_global_seeds(seed=42):
    """Sets all global seeds (for reproducibility)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"🌱 Global Seed set to {seed}.")

def split_polygons_leakage_free(gdf, label_column='Final_Clas', test_size=0.2, seed=42):
    """
    Splits shapefile based on polygons. 
    """
    # 1. Clean: Only keep labeled polygons
    gdf = gdf.dropna(subset=[label_column])
    gdf = gdf[(gdf[label_column] != -1) & (gdf[label_column] != -1.0) & (gdf[label_column] != "")]
    
    # 2. Do split (with stratification for class balance)
    train_gdf, test_gdf = train_test_split(
        gdf, test_size=test_size, random_state=seed, stratify=gdf[label_column]
    )
    
    print(f"🪓 Polygon split done:")
    print(f" -> Training: {len(train_gdf)} Polygons")
    print("\n--- TRAIN: CLASS DISTRIBUTION ---")
    counts = train_gdf['Final_Clas'].value_counts(dropna=False)
    for cls_name, count in counts.items():
        print(f"Class '{cls_name}': {count} polygons")
    print("--------------------------------------\n")

    print(f" -> Testing:  {len(test_gdf)} Polygons")
    print("\n--- TEST: CLASS DISTRIBUTION ---")
    counts = test_gdf['Final_Clas'].value_counts(dropna=False)
    for cls_name, count in counts.items():
        print(f"Class '{cls_name}': {count} polygons")
    print("--------------------------------------\n")
            
    return train_gdf, test_gdf