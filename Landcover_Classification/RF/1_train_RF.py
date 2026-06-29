import os
import sys
import otbApplication as otb
import geopandas as gpd
from pathlib import Path
import numpy as np

# Machine Learning / Metrics Imports (Same as CNN script)
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

curr_dir = Path.cwd()
parent_dir = curr_dir.parent
sys.path.append(str(parent_dir))
print(parent_dir)
from helper import set_global_seeds, split_polygons_leakage_free

# This code is adapted from: https://github.com/pajevicnina/inspire1-seg

# Define features globally so train and evaluate functions use the exact same list
FEATURES = [
    "meanB0", "meanB1", "meanB2", "meanB3", "meanB4", "meanB5", "meanB6",
    "varB0",  "varB1",  "varB2",  "varB3",  "varB4",  "varB5",  "varB6",
    # "Brightness"
]

def prepare_data_splits(master_shp, train_shp_out, val_shp_out):
    """
    Uses the global helper function to perform a stratified, leakage-free split.
    Saves the outputs as physical shapefiles required by OTB.
    """
    print(f"Loading master shapefile: {os.path.basename(master_shp)}...")
    gdf_full = gpd.read_file(master_shp)
    
    # Use the exact same split logic as the CNN to ensure fair model comparison
    train_gdf, test_gdf = split_polygons_leakage_free(
        gdf_full, 
        label_column="Final_Clas", 
        test_size=0.2, 
        seed=42
    )
    
    print("Saving stratified subsets to disk for OTB consumption...")
    # OTB requires physical files, so we write the split dataframes back to disk
    train_gdf.to_file(train_shp_out)
    test_gdf.to_file(val_shp_out)
    
    return len(train_gdf), len(test_gdf)

def train_random_forest(train_shp, val_shp, output_folder):
    """
    Uses OTB TrainVectorClassifier to train a Random Forest model.
    """
    print("\nStarting OTB Random Forest Training...")
    
    model_out = os.path.join(output_folder, "rf_model.txt")
    conf_mat_out = os.path.join(output_folder, "rf_confusion_matrix_v1.csv")
    
    app = otb.Registry.CreateApplication("TrainVectorClassifier")
    
    # Inputs
    app.SetParameterStringList("io.vd", [train_shp])
    app.SetParameterStringList("valid.vd", [val_shp])
    app.SetParameterString("cfield", "Final_Clas") 
    
    app.SetParameterStringList("feat", FEATURES)
    
    # Model parameters
    app.SetParameterString("classifier", "rf")
    # Maximum depth of the tree (15 is a solid default to prevent overfitting)
    app.SetParameterInt("classifier.rf.max", 15)
    # Minimum number of samples in a node (5 is standard)
    app.SetParameterInt("classifier.rf.min", 5)
    # Number of trees in the forest (Increased to 100 for better stability and accuracy)
    app.SetParameterInt("classifier.rf.nbtrees", 100) 
    
    # Outputs
    app.SetParameterString("io.out", model_out)
    app.SetParameterString("io.confmatout", conf_mat_out)
    
    # Execute Training
    print("Training Random Forest... .")
    app.ExecuteAndWriteOutput()
    
    print("\n--- Training Complete! ---")
    print(f"Model saved to: {model_out}")
    print(f"Confusion Matrix saved to: {conf_mat_out}")

    return model_out

def evaluate_random_forest(val_shp, model_path, output_folder):
    """
    Uses the trained RF model to predict the validation set, then applies
    scikit-learn and seaborn to generate the EXACT same reports as the CNN.
    """
    print("\nStarting Evaluation to match CNN output format...")
    predicted_shp = os.path.join(output_folder, "rf_val_predictions.shp")
    
    # 1. Inference with OTB (Apply model to validation data)
    app = otb.Registry.CreateApplication("VectorClassifier")
    app.SetParameterString("in", val_shp)
    app.SetParameterString("model", model_path)
    app.SetParameterStringList("feat", FEATURES)
    # The new column where OTB will write the predictions
    app.SetParameterString("cfield", "Predicted") 
    app.SetParameterString("out", predicted_shp)
    app.ExecuteAndWriteOutput()
    
    # 2. Load predictions back into Python
    print("Calculating Scikit-Learn Metrics...")
    gdf = gpd.read_file(predicted_shp)
    
    # Extract True Labels and Predicted Labels
    y_true = gdf["Final_Clas"].astype(int).values
    y_pred = gdf["Predicted"].astype(int).values
    
    # 3. Calculate identical metrics to CNN
    accuracy = np.mean(y_true == y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Extract unique class names and sort them for consistent matrix labeling
    classes = np.unique(np.concatenate((y_true, y_pred)))
    class_names_str = [str(c) for c in classes]
    
    report = classification_report(y_true, y_pred, target_names=class_names_str)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # 4. Save metrics to text (Identical format)
    with open(os.path.join(output_folder, "rf_evaluation_results.txt"), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Kappa Index: {kappa:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        
    # 5. Save confusion matrix plot (Identical format)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_str, yticklabels=class_names_str)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(output_folder, "rf_confusion_matrix.png"))
    plt.close()
    
    print(f"Evaluation saved successfully in {output_folder}")


def main():
    # 1. Set global seed to match CNN environment
    set_global_seeds(42)
    
    # 2. Setup Paths (Matched to CNN Script)
    base_path = curr_dir.parent.parent
    print(f"Base path automatically set to: {str(base_path)}")

    # Get Labels
    labels_dir = os.path.join(base_path, "Data", "9_Training_Data")
    master_shp = os.path.join(labels_dir, "training_data_final.shp") 

    # Define Output Dir for RF Results
    result_dir = os.path.join(base_path, "Data", "10_Landcover_Classification", "RF",  "v3")
    os.makedirs(result_dir, exist_ok=True)
    
    # Output Paths for the temporary split files
    train_shp = os.path.join(result_dir, "rf_train_split.shp")
    val_shp = os.path.join(result_dir, "rf_val_split.shp")
    
    # 3. Execute Pipeline
    print("=== Step 1: Stratified Data Split ===")
    train_count, val_count = prepare_data_splits(master_shp, train_shp, val_shp)
    print(f" -> Training features written: {train_count}")
    print(f" -> Validation features written: {val_count}")
    
    print("\n=== Step 2: Training Model ===")
    model_path = train_random_forest(train_shp, val_shp, result_dir)

    print("\n=== Step 3: Evaluating Model ===")
    evaluate_random_forest(val_shp, model_path, result_dir)

if __name__ == "__main__":
    # Do this in console: source /net/home/sloeblein/otb-9.1.0/otbenv.profile
    main()