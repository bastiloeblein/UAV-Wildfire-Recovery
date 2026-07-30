import os
import sys
import json
import pickle
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box, mapping
from pathlib import Path

# Helper Import
curr_dir = Path.cwd()
parent_dir = curr_dir.parent
sys.path.append(str(parent_dir))
from helper import set_global_seeds

# Deep Learning Imports
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Machine Learning / Metrics Imports
from sklearn.metrics import (
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


# This code is adapted from:
# https://github.com/ranavaqcar1989/CNN-Training-pipeline


class CNNSegmentor:
    def __init__(
        self,
        image_path,
        label_column="Final_Clas",
        base_result_folder="cnn_results",
        img_size=(128, 128),
        channels_to_use=None,
    ):
        self.image_path = image_path
        self.label_column = label_column
        self.img_size = img_size
        self.channels_to_use = channels_to_use

        # Keep original experiment-folder logic
        ch_suffix = (
            "".join(map(str, self.channels_to_use))
            if self.channels_to_use is not None
            else "all"
        )
        folder_name = f"{base_result_folder}_w{self.img_size[0]}_ch{ch_suffix}"
        self.result_folder = folder_name
        os.makedirs(self.result_folder, exist_ok=True)
        self.channels = None

    @staticmethod
    def _repair_geometry(geom):
        """Repair invalid polygon geometries before extracting a patch."""
        if geom is None or geom.is_empty:
            return None

        if not geom.is_valid:
            repaired = geom.buffer(0)
            if repaired is not None and not repaired.is_empty:
                geom = repaired

        if geom.is_empty:
            return None

        return geom

    @staticmethod
    def _find_patch_center(geom, src, image):
        """
        Find one stable patch centre inside the polygon and raster extent.

        The geometric centroid is used whenever possible. For concave polygons
        or NoData centres, an interior or nearby valid pixel is used instead.
        """
        raster_extent = box(
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
        )

        geom_in_raster = geom.intersection(raster_extent)
        if geom_in_raster.is_empty or geom_in_raster.area == 0:
            return None, None, "outside_raster"

        centre_point = geom.centroid
        source = "centroid"

        # Concave polygons can have a centroid outside the polygon.
        if not geom_in_raster.covers(centre_point):
            centre_point = geom_in_raster.representative_point()
            source = "inside_point"

        c_row, c_col = rasterio.transform.rowcol(
            src.transform,
            centre_point.x,
            centre_point.y,
        )

        # Keep the centre inside the raster grid.
        c_row = int(np.clip(c_row, 0, image.shape[0] - 1))
        c_col = int(np.clip(c_col, 0, image.shape[1] - 1))

        if image[c_row, c_col, 0] != -1.0:
            return c_row, c_col, source

        # Search for the closest valid raster pixel inside the polygon.
        minx, miny, maxx, maxy = geom_in_raster.bounds
        row_a, col_a = rasterio.transform.rowcol(
            src.transform,
            minx,
            maxy,
        )
        row_b, col_b = rasterio.transform.rowcol(
            src.transform,
            maxx,
            miny,
        )

        r0 = max(0, min(row_a, row_b))
        r1 = min(image.shape[0], max(row_a, row_b) + 1)
        c0 = max(0, min(col_a, col_b))
        c1 = min(image.shape[1], max(col_a, col_b) + 1)

        if r1 > r0 and c1 > c0:
            window = Window(c0, r0, c1 - c0, r1 - r0)
            polygon_mask = geometry_mask(
                [mapping(geom_in_raster)],
                out_shape=(r1 - r0, c1 - c0),
                transform=src.window_transform(window),
                invert=True,
            )
            valid_pixels = (
                polygon_mask
                & (image[r0:r1, c0:c1, 0] != -1.0)
            )

            if valid_pixels.any():
                local_rows, local_cols = np.where(valid_pixels)
                global_rows = local_rows + r0
                global_cols = local_cols + c0
                distances = (
                    (global_rows - c_row) ** 2
                    + (global_cols - c_col) ** 2
                )
                nearest = int(np.argmin(distances))
                return (
                    int(global_rows[nearest]),
                    int(global_cols[nearest]),
                    "nearest_valid",
                )

        # The patch is still extracted if the polygon itself is NoData.
        # Surrounding pixels may still contain useful spatial information.
        return c_row, c_col, "nodata_center"

    def _extract_patch(self, image, row, col):
        """Extract one fixed-size patch and pad image borders with -1."""
        patch_height, patch_width = self.img_size
        half_height = patch_height // 2
        half_width = patch_width // 2

        y_min = row - half_height
        y_max = y_min + patch_height
        x_min = col - half_width
        x_max = x_min + patch_width

        patch = np.full(
            (patch_height, patch_width, self.channels),
            -1.0,
            dtype=image.dtype,
        )

        img_y_min = max(0, y_min)
        img_y_max = min(image.shape[0], y_max)
        img_x_min = max(0, x_min)
        img_x_max = min(image.shape[1], x_max)

        if img_y_max <= img_y_min or img_x_max <= img_x_min:
            return None

        patch_y_min = img_y_min - y_min
        patch_y_max = patch_y_min + (img_y_max - img_y_min)
        patch_x_min = img_x_min - x_min
        patch_x_max = patch_x_min + (img_x_max - img_x_min)

        patch[
            patch_y_min:patch_y_max,
            patch_x_min:patch_x_max,
        ] = image[
            img_y_min:img_y_max,
            img_x_min:img_x_max,
        ]

        return patch

    def load_data(self, gdf, sampling_mode="centroid"):
        """
        Extract one centroid-based patch per labelled polygon.

        The same extraction logic is used for training, validation, testing,
        and full-map prediction so every polygon has the same weight.
        """
        if sampling_mode != "centroid":
            raise ValueError(
                "This pipeline uses sampling_mode='centroid' for all splits."
            )

        image_patches = []
        labels = []
        patch_sources = {}
        unusable_indices = []

        print(f"Loading raster image: {os.path.basename(self.image_path)}...")
        with rasterio.open(self.image_path) as src:
            if gdf.crs is not None and src.crs is not None and gdf.crs != src.crs:
                raise ValueError(
                    f"CRS mismatch: vector={gdf.crs}, raster={src.crs}"
                )

            image = src.read().astype(np.float32)
            image = np.moveaxis(image, 0, -1)
            image = np.nan_to_num(
                image,
                nan=-1.0,
                posinf=-1.0,
                neginf=-1.0,
            )
            image[image == -9999.0] = -1.0

            if self.channels_to_use is not None:
                print(f"Extracting user-selected channels: {self.channels_to_use}")
                image = image[:, :, self.channels_to_use]

            self.channels = image.shape[-1]
            print(f"Active image shape for CNN tracking: {image.shape}")
            print(f"Extracting one {self.img_size} patch per polygon...")

            for idx, row in gdf.iterrows():
                geom = self._repair_geometry(row["geometry"])

                if geom is None:
                    unusable_indices.append(idx)
                    continue

                c_row, c_col, source = self._find_patch_center(
                    geom,
                    src,
                    image,
                )

                if c_row is None or c_col is None:
                    unusable_indices.append(idx)
                    continue

                patch = self._extract_patch(
                    image,
                    c_row,
                    c_col,
                )

                if patch is None:
                    unusable_indices.append(idx)
                    continue

                image_patches.append(patch)
                labels.append(row[self.label_column])
                patch_sources[source] = patch_sources.get(source, 0) + 1

        if unusable_indices:
            raise RuntimeError(
                f"Could not extract a patch for {len(unusable_indices)} polygons. "
                f"First affected indices: {unusable_indices[:10]}. "
                "Check for empty geometries or polygons outside the raster."
            )

        if len(image_patches) != len(gdf):
            raise RuntimeError(
                f"Patch count mismatch: {len(image_patches)} patches for "
                f"{len(gdf)} polygons."
            )

        print(f"SUCCESS! Extracted {len(image_patches)} patches.")
        print(f"Patch centres used: {patch_sources}")

        X = np.asarray(
            image_patches,
            dtype=np.float32,
        ).reshape(
            -1,
            self.img_size[0],
            self.img_size[1],
            self.channels,
        )
        return X, labels

    def preprocess_labels(
        self,
        labels_train,
        labels_validation,
        labels_test=None,
    ):
        """
        Fit the label encoder on training labels only, then transform
        validation and optional test labels.
        """
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        label_encoder.fit(labels_train)

        known_classes = set(label_encoder.classes_)
        for split_name, split_labels in [
            ("validation", labels_validation),
            ("test", labels_test),
        ]:
            if split_labels is None:
                continue
            unseen = set(split_labels) - known_classes
            if unseen:
                raise ValueError(
                    f"{split_name} contains classes absent from training: {unseen}"
                )

        with open(
            os.path.join(self.result_folder, "label_encoder.pkl"),
            "wb",
        ) as file:
            pickle.dump(label_encoder, file)

        num_classes = len(label_encoder.classes_)

        y_train_enc = label_encoder.transform(labels_train)
        y_validation_enc = label_encoder.transform(labels_validation)
        y_train_cat = to_categorical(
            y_train_enc,
            num_classes=num_classes,
        )
        y_validation_cat = to_categorical(
            y_validation_enc,
            num_classes=num_classes,
        )

        if labels_test is None:
            return (
                y_train_enc,
                y_validation_enc,
                y_train_cat,
                y_validation_cat,
                label_encoder,
            )

        y_test_enc = label_encoder.transform(labels_test)
        y_test_cat = to_categorical(
            y_test_enc,
            num_classes=num_classes,
        )
        return (
            y_train_enc,
            y_validation_enc,
            y_test_enc,
            y_train_cat,
            y_validation_cat,
            y_test_cat,
            label_encoder,
        )

    def build_cnn(self, num_classes):
        print("Building CNN architecture...")
        model = Sequential()

        model.add(
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(
                    self.img_size[0],
                    self.img_size[1],
                    self.channels,
                ),
            )
        )
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
            )
        )
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
            )
        )
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                256,
                (3, 3),
                activation="relu",
                padding="same",
            )
        )
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation="softmax"))

        custom_adam = Adam(learning_rate=0.0005)
        model.compile(
            optimizer=custom_adam,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train_cnn(
        self,
        model,
        X_train,
        y_train_cat,
        y_train_enc,
        X_validation,
        y_validation_cat,
    ):
        """Calculate class weights and train using validation only."""
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train_enc),
            y=y_train_enc,
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class weights: {class_weight_dict}")

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
            verbose=1,
        )

        datagen = ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True,
        )
        train_generator = datagen.flow(
            X_train,
            y_train_cat,
            batch_size=32,
            shuffle=True,
            seed=42,
        )

        validation_datagen = ImageDataGenerator()
        validation_generator = validation_datagen.flow(
            X_validation,
            y_validation_cat,
            batch_size=32,
            shuffle=False,
        )

        print("Starting training...")
        history = model.fit(
            train_generator,
            epochs=200,
            validation_data=validation_generator,
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weight_dict,
        )

        # restore_best_weights=True means this saves the best validation epoch.
        model.save(os.path.join(self.result_folder, "cnn_model.h5"))
        return history

    def evaluate_cnn(
        self,
        model,
        X,
        y_cat,
        label_encoder,
        split_name,
    ):
        """Evaluate one split and save metrics."""
        print(f"Evaluating CNN on {split_name}...")
        y_prob = model.predict(X, batch_size=32)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(y_cat, axis=1)

        labels = np.arange(len(label_encoder.classes_))
        class_names = [str(c) for c in label_encoder.classes_]

        accuracy = float(np.mean(y_pred == y_true))
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

        prefix = f"cnn_{split_name}"

        with open(
            os.path.join(
                self.result_folder,
                f"{prefix}_metrics.json",
            ),
            "w",
        ) as file:
            json.dump(metrics, file, indent=2)

        with open(
            os.path.join(
                self.result_folder,
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
                self.result_folder,
                f"{prefix}_confusion_matrix.png",
            )
        )
        plt.close()

        return metrics


def main():
    set_global_seeds(42)

    base_path = curr_dir.parent.parent
    print(f"Base path: {str(base_path)}")

    data_dir = os.path.join(
        base_path,
        "Data",
        "8_Images_with_Indices",
    )
    image_path = os.path.join(
        data_dir,
        "250821_7_channel.tif",
    )

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

    result_dir = os.path.join(
        base_path,
        "Data",
        "10_Landcover_Classification",
        "CNN",
        "v5",
    )

    target_window_size = (128, 128)
    selected_channels = [0, 1, 2, 3, 4, 5, 6]

    print("Loading precomputed spatial splits...")
    train_gdf = gpd.read_file(train_path)
    validation_gdf = gpd.read_file(validation_path)

    segmentor = CNNSegmentor(
        image_path,
        label_column="Final_Clas",
        base_result_folder=result_dir,
        img_size=target_window_size,
        channels_to_use=selected_channels,
    )

    # Extract one centroid patch per training polygon.
    print("\n--- Extracting Training Patches ---")
    X_train, labels_train = segmentor.load_data(
        train_gdf,
        sampling_mode="centroid",
    )

    # One centroid patch per polygon for a fair polygon-level validation
    # and consistency with predict_cnn.py.
    print("\n--- Extracting Validation Patches ---")
    X_validation, labels_validation = segmentor.load_data(
        validation_gdf,
        sampling_mode="centroid",
    )

    (
        y_train_enc,
        y_validation_enc,
        y_train_cat,
        y_validation_cat,
        label_encoder,
    ) = segmentor.preprocess_labels(
        labels_train,
        labels_validation,
    )

    model = segmentor.build_cnn(
        num_classes=len(label_encoder.classes_)
    )
    segmentor.train_cnn(
        model,
        X_train,
        y_train_cat,
        y_train_enc,
        X_validation,
        y_validation_cat,
    )

    # Validation is used for model-family selection.
    segmentor.evaluate_cnn(
        model,
        X_validation,
        y_validation_cat,
        label_encoder,
        split_name="validation",
    )

    print(
        "\nCNN training complete. "
        "Do not evaluate the test set until CNN and RF "
        "have been compared on validation."
    )


if __name__ == "__main__":
    main()
