import os
import sys
import json
import otbApplication as otb
import geopandas as gpd
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

curr_dir = Path.cwd()
parent_dir = curr_dir.parent
sys.path.append(str(parent_dir))
from helper import set_global_seeds


# This code is adapted from:
# https://github.com/pajevicnina/inspire1-seg

# Define features globally so train and evaluate functions use the exact same list
FEATURES = [
    "meanB0", "meanB1", "meanB2", "meanB3", "meanB4", "meanB5", "meanB6",
    "varB0", "varB1", "varB2", "varB3", "varB4", "varB5", "varB6",
]


def convert_split_to_shapefile(input_gpkg, output_shp):
    """
    Keep the original OTB/shapefile workflow, but use the already
    generated spatial split as input.
    """
    gdf = gpd.read_file(input_gpkg)
    gdf.to_file(output_shp)
    return len(gdf)


def train_random_forest(train_shp, validation_shp, output_folder):
    """
    Train the original OTB Random Forest using the spatial training set
    and the separate validation set.
    """
    print("\nStarting OTB Random Forest Training...")

    model_out = os.path.join(output_folder, "rf_model.txt")
    conf_mat_out = os.path.join(
        output_folder,
        "rf_otb_validation_confusion_matrix.csv",
    )

    app = otb.Registry.CreateApplication("TrainVectorClassifier")

    app.SetParameterStringList("io.vd", [train_shp])
    app.SetParameterStringList("valid.vd", [validation_shp])
    app.SetParameterString("cfield", "Final_Clas")
    app.SetParameterStringList("feat", FEATURES)

    app.SetParameterString("classifier", "rf")
    app.SetParameterInt("classifier.rf.max", 15)
    app.SetParameterInt("classifier.rf.min", 5)
    app.SetParameterInt("classifier.rf.nbtrees", 100)
    app.SetParameterInt("rand", 42)

    app.SetParameterString("io.out", model_out)
    app.SetParameterString("io.confmatout", conf_mat_out)

    print("Training Random Forest...")
    app.ExecuteAndWriteOutput()

    print(f"Model saved to: {model_out}")
    return model_out


def predict_vector(input_shp, model_path, output_shp):
    """Apply an existing OTB model to one vector split."""
    app = otb.Registry.CreateApplication("VectorClassifier")
    app.SetParameterString("in", input_shp)
    app.SetParameterString("model", model_path)
    app.SetParameterStringList("feat", FEATURES)
    app.SetParameterString("cfield", "Predicted")
    app.SetParameterString("out", output_shp)
    app.ExecuteAndWriteOutput()


def evaluate_random_forest(
    input_shp,
    model_path,
    output_folder,
    split_name,
):
    """
    Predict and evaluate one split using the same report structure as
    the CNN.
    """
    print(f"\nEvaluating Random Forest on {split_name}...")
    predicted_shp = os.path.join(
        output_folder,
        f"rf_{split_name}_predictions.shp",
    )
    predict_vector(input_shp, model_path, predicted_shp)

    gdf = gpd.read_file(predicted_shp)
    y_true = gdf["Final_Clas"].astype(int).values
    y_pred = gdf["Predicted"].astype(int).values

    labels = sorted(
        np.unique(np.concatenate((y_true, y_pred))).tolist()
    )
    class_names = [str(c) for c in labels]

    accuracy = float(np.mean(y_true == y_pred))
    balanced_accuracy = float(
        balanced_accuracy_score(y_true, y_pred)
    )
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro")
    )
    kappa = float(cohen_kappa_score(y_true, y_pred))

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "kappa": kappa,
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    print(metrics)
    print(report)

    prefix = f"rf_{split_name}"

    with open(
        os.path.join(output_folder, f"{prefix}_metrics.json"),
        "w",
    ) as file:
        json.dump(metrics, file, indent=2)

    with open(
        os.path.join(
            output_folder,
            f"{prefix}_evaluation_results.txt",
        ),
        "w",
    ) as file:
        for name, value in metrics.items():
            file.write(f"{name}: {value:.4f}\n")
        file.write("\nClassification Report:\n")
        file.write(report)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.savefig(
        os.path.join(
            output_folder,
            f"{prefix}_confusion_matrix.png",
        )
    )
    plt.close()

    return metrics


def main():
    set_global_seeds(42)

    base_path = curr_dir.parent.parent
    print(f"Base path: {str(base_path)}")

    split_dir = os.path.join(
        base_path,
        "Data",
        "9_Training_Data",
        "spatial_split_v4",
    )
    train_gpkg = os.path.join(split_dir, "train.gpkg")
    validation_gpkg = os.path.join(
        split_dir,
        "validation.gpkg",
    )

    result_dir = os.path.join(
        base_path,
        "Data",
        "10_Landcover_Classification",
        "RF",
        "v4",
    )
    os.makedirs(result_dir, exist_ok=True)

    train_shp = os.path.join(result_dir, "rf_train_split.shp")
    validation_shp = os.path.join(
        result_dir,
        "rf_validation_split.shp",
    )

    print("Converting precomputed spatial splits for OTB...")
    train_count = convert_split_to_shapefile(
        train_gpkg,
        train_shp,
    )
    validation_count = convert_split_to_shapefile(
        validation_gpkg,
        validation_shp,
    )
    print(f"Training polygons: {train_count}")
    print(f"Validation polygons: {validation_count}")

    model_path = train_random_forest(
        train_shp,
        validation_shp,
        result_dir,
    )

    # Validation is used for model-family selection.
    evaluate_random_forest(
        validation_shp,
        model_path,
        result_dir,
        split_name="validation",
    )

    print(
        "\nRF training complete. "
        "Do not evaluate the test set until CNN and RF "
        "have been compared on validation."
    )


if __name__ == "__main__":
    # source /net/home/sloeblein/otb-9.1.0/otbenv.profile
    main()
