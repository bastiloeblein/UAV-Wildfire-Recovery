import os
import pickle
import importlib.util
from pathlib import Path

import geopandas as gpd
from keras.models import load_model
from keras.utils import to_categorical


module_path = Path(__file__).with_name("1_train_CNN.py")
spec = importlib.util.spec_from_file_location("train_cnn_module", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not import training module: {module_path}")

train_cnn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_cnn_module)
CNNSegmentor = train_cnn_module.CNNSegmentor


def main():
    curr_dir = Path.cwd()
    base_path = curr_dir.parent.parent

    image_path = os.path.join(
        base_path,
        "Data",
        "8_Images_with_Indices",
        "250821_7_channel.tif",
    )

    split_dir = os.path.join(
        base_path,
        "Data",
        "9_Training_Data",
        "spatial_split_v4",
    )
    test_path = os.path.join(split_dir, "test.gpkg")

    result_dir = os.path.join(
        base_path,
        "Data",
        "10_Landcover_Classification",
        "CNN",
        "v5",
    )
    selected_channels = [0, 1, 2, 3, 4, 5, 6]
    result_folder = (
        f"{result_dir}_w128_ch"
        f"{''.join(map(str, selected_channels))}"
    )

    model_path = os.path.join(result_folder, "cnn_model.h5")
    encoder_path = os.path.join(result_folder, "label_encoder.pkl")

    for required_path in [test_path, model_path, encoder_path, image_path]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required file not found: {required_path}")

    test_gdf = gpd.read_file(test_path)

    segmentor = CNNSegmentor(
        image_path,
        label_column="Final_Clas",
        base_result_folder=result_dir,
        img_size=(128, 128),
        channels_to_use=selected_channels,
    )

    X_test, labels_test = segmentor.load_data(
        test_gdf,
        sampling_mode="centroid",
    )

    print(
        f"Test polygons: {len(test_gdf)}, "
        f"test patches: {len(labels_test)}"
    )

    with open(encoder_path, "rb") as file:
        label_encoder = pickle.load(file)

    unseen = set(labels_test) - set(label_encoder.classes_)
    if unseen:
        raise ValueError(f"Test contains unseen classes: {unseen}")

    y_test_enc = label_encoder.transform(labels_test)
    y_test_cat = to_categorical(
        y_test_enc,
        num_classes=len(label_encoder.classes_),
    )

    model = load_model(model_path)

    segmentor.evaluate_cnn(
        model,
        X_test,
        y_test_cat,
        label_encoder,
        split_name="test",
    )


if __name__ == "__main__":
    main()
