import os
import importlib.util
from pathlib import Path

module_path = Path(__file__).with_name("1_train_RF.py")
spec = importlib.util.spec_from_file_location("train_rf_module", module_path)
train_rf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_rf_module)

convert_split_to_shapefile = train_rf_module.convert_split_to_shapefile
evaluate_random_forest = train_rf_module.evaluate_random_forest


def main():
    curr_dir = Path.cwd()
    base_path = curr_dir.parent.parent

    split_dir = os.path.join(
        base_path,
        "Data",
        "9_Training_Data",
        "spatial_split_v4",
    )
    test_gpkg = os.path.join(split_dir, "test.gpkg")

    result_dir = os.path.join(
        base_path,
        "Data",
        "10_Landcover_Classification",
        "RF",
        "v5",
    )

    model_path = os.path.join(result_dir, "rf_model.txt")
    test_shp = os.path.join(result_dir, "rf_test_split.shp")

    # OTB-compatible conversion
    convert_split_to_shapefile(test_gpkg, test_shp)

    evaluate_random_forest(
        test_shp,
        model_path,
        result_dir,
        split_name="test",
    )


if __name__ == "__main__":
    main()
