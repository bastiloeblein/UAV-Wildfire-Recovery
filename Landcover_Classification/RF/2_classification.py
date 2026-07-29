import os
from pathlib import Path
import otbApplication as otb
import geopandas as gpd

# This code is adapted from: https://github.com/pajevicnina/inspire1-seg

# --- GLOBAL FEATURES ---
# CRITICAL: This list MUST be exactly identical to the features used in train_random_forest.py!
FEATURES = [
    "meanB0", "meanB1", "meanB2", "meanB3", "meanB4", "meanB5", "meanB6",
    "varB0",  "varB1",  "varB2",  "varB3",  "varB4",  "varB5",  "varB6",
    # "Brightness"
]

def classify_vector_data(input_shp, model_path, output_shp):
    """
    Applies the trained Random Forest model to the full shapefile.
    Predicts the class for every single polygon (including unclassified ones).
    """
    print(f"\nStarting Full Classification on: {os.path.basename(input_shp)}...")
    print(f"Using RF Model: {os.path.basename(model_path)}...")
    
    app = otb.Registry.CreateApplication("VectorClassifier")
    
    # 1. Inputs
    app.SetParameterString("in", input_shp)
    app.SetParameterString("model", model_path)
    
    # 2. Features
    app.SetParameterStringList("feat", FEATURES)
    
    # 3. Output Field
    # This creates a NEW column in your shapefile where the predicted class ID will be stored.
    app.SetParameterString("cfield", "Predicted") 
    
    # 4. Output Shapefile
    app.SetParameterString("out", output_shp)
    
    # Execute the classification
    print("Classifying hundreds of thousands of polygons... This might take a minute.")
    app.ExecuteAndWriteOutput()
    
    print("\n✅ --- Full Classification Complete! ---")
    print(f"Your final mapped shapefile is saved at: {output_shp}")


def main():
    # 1. Setup Base Paths (Matching your architecture)
    curr_dir = Path.cwd()
    base_path = curr_dir.parent.parent
    print(f"Base path automatically set to: {str(base_path)}")

    # Input Master Shapefile 
    labels_dir = os.path.join(base_path, "Data", "9_Training_Data")
    input_vector = os.path.join(labels_dir, "training_data_rf_ready.shp") 
    
    # Directory where your Random Forest model lives
    rf_dir = os.path.join(base_path, "Data", "10_Landcover_Classification", "RF", "v5")
    model_file = os.path.join(rf_dir, "rf_model.txt")
        
    # Output file: The completely classified map ready for QGIS
    output_vector = os.path.join(
        rf_dir,
        "classified_full_map_rf.shp",
    )
    
    # 2. Safety Checks before running
    if not os.path.exists(model_file):
        print(f"❌ ERROR: Could not find the trained model at {model_file}")
        print("Run the RF training script first!")
        return
        
    if not os.path.exists(input_vector):
        print(f"❌ ERROR: Could not find the master shapefile at {input_vector}")
        return

    gdf = gpd.read_file(input_vector)

    if "Predicted" in gdf.columns:
        gdf = gdf.drop(columns=["Predicted"])
        clean_input = os.path.join(
            rf_dir,
            "full_map_without_prediction.shp",
        )
        gdf.to_file(clean_input)
        input_vector = clean_input
        
    # 3. Execute Pipeline
    classify_vector_data(input_vector, model_file, output_vector)


if __name__ == "__main__":
    # Reminder: source /net/home/sloeblein/otb-9.1.0/otbenv.profile
    main()