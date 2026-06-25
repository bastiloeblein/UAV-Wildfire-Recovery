import os
import sys
import pickle
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# Helper Import!
curr_dir = Path.cwd()
parent_dir = curr_dir.parent
sys.path.append(str(parent_dir))
from helper import set_global_seeds, split_polygons_leakage_free

# Deep Learning Imports
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Machine Learning / Metrics Imports
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


# This code is adapted from: https://github.com/ranavaqcar1989/CNN-Training-pipeline


class CNNSegmentor:
    def __init__(self, image_path, label_column="Final_Clas", base_result_folder="cnn_results", img_size=(128, 128), channels_to_use=None):
        self.image_path = image_path
        self.label_column = label_column
        self.img_size = img_size 
        self.channels_to_use = channels_to_use 
        
        # Automatically generate an informative unique folder name for tracking experiments
        ch_suffix = "".join(map(str, self.channels_to_use)) if self.channels_to_use is not None else "all"
        folder_name = f"{base_result_folder}_w{self.img_size[0]}_ch{ch_suffix}"
        self.result_folder = folder_name
        os.makedirs(self.result_folder, exist_ok=True)
        self.channels = None  

    def load_data(self, gdf):
        """
        Extracts multiple patches from polygons using a grid-based approach.
        Filters channels dynamically and converts -9999.0 NoData markers to -1.0 for CNN stability.
        """
        image_patches = []
        labels = []
        
        print(f"Loading raster image: {os.path.basename(self.image_path)}...")
        with rasterio.open(self.image_path) as src:
            # Read all bands
            image = src.read()
            # Convert from (Channels, Height, Width) to Keras format (Height, Width, Channels)
            image = np.moveaxis(image, 0, -1)
            
            # --- Dynamic Channel Selection ---
            if self.channels_to_use is not None:
                print(f"Extracting user-selected channels: {self.channels_to_use}")
                image = image[:, :, self.channels_to_use]
            
            self.channels = image.shape[-1]
            print(f"Active image shape for CNN tracking: {image.shape}")
            
            # --- CNN NoData Fix (Zero-Padding Implementation) ---
            # Replaces hazardous -9999.0 background/shadow values with to -1.0
            image[image == -9999.0] = -1.0
            print("Converted -9999.0 NoData values to -1.0 for stable convolutional processing.")
            
            patch_radius = self.img_size[0] // 2 
            stride = self.img_size[0] // 2 
            
            print(f"Extracting {self.img_size} patches using grid sampling...")
            
            for idx, row in gdf.iterrows():
                label = row[self.label_column]
                geom = row['geometry']
                
                # Get the bounding box of the polygon
                minx, miny, maxx, maxy = geom.bounds
                row_min, col_min = rasterio.transform.rowcol(src.transform, minx, maxy)
                row_max, col_max = rasterio.transform.rowcol(src.transform, maxx, miny)
                
                # Ensure correct ordering of min/max coordinates
                r_start, r_end = min(row_min, row_max), max(row_min, row_max)
                c_start, c_end = min(col_min, col_max), max(col_min, col_max)
                
                points_to_check = []
                
                # If the polygon is tiny, we just take the centroid
                if (r_end - r_start) < stride or (c_end - c_start) < stride:
                    centroid = geom.centroid
                    c_row, c_col = rasterio.transform.rowcol(src.transform, centroid.x, centroid.y)
                    points_to_check.append((centroid.x, centroid.y, c_row, c_col))
                else:
                    # Grid sampling for large polygons
                    for r in range(r_start, r_end, stride):
                        for c in range(c_start, c_end, stride):
                            # Convert pixel coordinates back to geographic coordinates
                            geo_x, geo_y = src.xy(r, c)
                            points_to_check.append((geo_x, geo_y, r, c))
                
                # Process all sampled points
                for geo_x, geo_y, r, c in points_to_check:
                    # Crucial check: Is this specific point actually inside the polygon boundaries?
                    if geom.contains(Point(geo_x, geo_y)):
                        y_min = r - patch_radius
                        y_max = r + patch_radius
                        x_min = c - patch_radius
                        x_max = c + patch_radius
                        
                        # Ensure the patch doesn't exceed image boundaries
                        if (x_min >= 0 and y_min >= 0 and x_max <= image.shape[1] and y_max <= image.shape[0]):
                            patch = image[y_min:y_max, x_min:x_max]
                            
                            # Safety check for exact size
                            if patch.shape[:2] == self.img_size:
                                image_patches.append(patch)
                                labels.append(label)
                                
        print(f"SUCCESS! Extracted {len(image_patches)} undistorted training patches.")
        X = np.array(image_patches).reshape(-1, self.img_size[0], self.img_size[1], self.channels)
        return X, labels

    def preprocess_labels(self, labels_train, labels_test):
        """ Combines labels for encoder to ensure all classes are known, then encodes them separately."""
        print("Encoding labels...")     
        label_encoder = LabelEncoder()

        all_labels = labels_train + labels_test
        label_encoder.fit(all_labels)

        with open(os.path.join(self.result_folder, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(label_encoder, f)

        y_train_enc = label_encoder.transform(labels_train)
        y_test_enc = label_encoder.transform(labels_test)
        
        y_train_cat = to_categorical(y_train_enc) 
        y_test_cat = to_categorical(y_test_enc) 
        
        return y_train_enc, y_test_enc, y_train_cat, y_test_cat, label_encoder

    def augment_data(self, X, y):
        """Augmentation (Rotation/Flip), to increase robustness."""
        print("Applying data augmentation (Rotation/Flip) on training data...")
        datagen = ImageDataGenerator(
            rotation_range=20, 
            horizontal_flip=True,
            vertical_flip=True
        )

        # datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
        #                              height_shift_range=0.1, shear_range=0.2,
        #                              zoom_range=0.1, horizontal_flip=True)
        
        # Doubles data set
        X_aug, y_aug = next(datagen.flow(X, y, batch_size=len(X), shuffle=False))
        
        X_combined = np.concatenate((X, X_aug), axis=0)
        y_combined = np.concatenate((y, y_aug), axis=0)
        
        return X_combined, y_combined

    # def augment_data(self, X, y):
    #     """Applies data augmentation to balance the classes perfectly."""
    #     print("Augmenting data to balance class distribution...")
    #     class_counts = np.bincount(np.argmax(y, axis=1))
    #     max_count = class_counts.max()
        
    #     augmented_images = []
    #     augmented_labels = []

    #     datagen = ImageDataGenerator(
    #         rotation_range=15, 
    #         width_shift_range=0.1,
    #         height_shift_range=0.1, 
    #         shear_range=0.2,
    #         zoom_range=0.1, 
    #         horizontal_flip=True
    #     )

    #     for class_idx in range(len(class_counts)):
    #         class_mask = np.argmax(y, axis=1) == class_idx
    #         class_images = X[class_mask]
    #         class_labels = y[class_mask]

    #         if class_counts[class_idx] < max_count:
    #             augment_size = max_count - class_counts[class_idx]
    #             i = 0
    #             for batch in datagen.flow(class_images, class_labels, batch_size=1):
    #                 # Re-shaping correctly based on the dynamically selected channels
    #                 augmented_images.append(batch[0].reshape(self.img_size[0], self.img_size[1], self.channels))
    #                 augmented_labels.append(batch[1].reshape(y.shape[1]))
    #                 i += 1
    #                 if i >= augment_size:
    #                     break

    #     if augmented_images:
    #         X_augmented = np.array(augmented_images)
    #         y_augmented = np.array(augmented_labels)
    #         X = np.concatenate((X, X_augmented), axis=0)
    #         y = np.concatenate((y, y_augmented), axis=0)

    #     print(f"Data perfectly balanced! Total training samples: {len(X)}.")
    #     return X, y

    def build_cnn(self, num_classes):
        print("Building CNN architecture...")
        model = Sequential()
        
        # Block 1
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.img_size[0], self.img_size[1], self.channels)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Block 2
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Block 3
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Block 4 (Optional, empfohlen bei 128x128 oder größer)
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(GlobalAveragePooling2D())
        
        # Classifier
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        custom_adam = Adam(learning_rate=0.0005) 
        model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def train_cnn(self, model, X_train, y_train_cat, y_train_enc, X_test, y_test_cat):
        """Calculates class weights and trains the model."""
        
        # Calculate class_weights to account for class imbalance-
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_enc),
            y=y_train_enc
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class Weights applied for imbalance: {class_weight_dict}")

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
        
        history = model.fit(
            X_train, y_train_cat, 
            epochs=200, 
            batch_size=32, 
            validation_data=(X_test, y_test_cat),
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weight_dict 
        )
        model.save(os.path.join(self.result_folder, "cnn_model.h5"))
        return history        

    def evaluate_cnn(self, model, X_test, y_test, label_encoder):
        """Evaluates the model and generates accuracy reports and matrices."""
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = np.mean(y_pred_classes == y_true)
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        class_names_str = [str(c) for c in label_encoder.classes_]
        
        cm = confusion_matrix(y_true, y_pred_classes)
        report = classification_report(y_true, y_pred_classes, target_names=class_names_str)
        kappa = cohen_kappa_score(y_true, y_pred_classes)
        
        # Save metrics to text
        with open(os.path.join(self.result_folder, "cnn_evaluation_results.txt"), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Kappa Index: {kappa:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(report)
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_str, yticklabels=class_names_str)
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.savefig(os.path.join(self.result_folder, "cnn_confusion_matrix.png"))
        plt.close()


def main():

    # 1. Set global seed
    set_global_seeds(42)
    
    # --- Define Paths ---
    base_path = curr_dir.parent.parent
    print(f"Base path automatically set to: {str(base_path)}")

    # Get Image
    data_dir = os.path.join(base_path, "Data", "8_Images_with_Indices")
    image_path = os.path.join(data_dir, "250821_7_channel.tif")

    # Get Labels
    labels_dir = os.path.join(base_path, "Data", "9_Training_Data")
    labels_path = os.path.join(labels_dir, "final_training_data_qgis.shp") 

    # Define Output Dir
    result_dir = os.path.join(base_path, "Data", "10_Landcover_Classification", "CNN")
    
    # --- CONFIGURABLE EXPERIMENT PARAMETERS ---
    target_window_size = (128, 128)
    
    # Indices mapped to your new band array order:
    # 0=R, 1=G, 2=B, 3=CHM, 4=ExG, 5=VARI, 6=NGRDI
    # To train with all 7 channels, pass: [0, 1, 2, 3, 4, 5, 6]
    selected_channels = [0, 1, 2, 3, 4, 5, 6]

    # Load shape file and split data
    print("Load Labelsfile for Polygon-Split...")
    gdf_full = gpd.read_file(labels_path)
    train_gdf, test_gdf = split_polygons_leakage_free(gdf_full, label_column="Final_Clas", test_size=0.2, seed=42)

    #Initialize Segmentor
    segmentor = CNNSegmentor(
        image_path, 
        label_column="Final_Clas", 
        base_result_folder=result_dir,
        img_size=target_window_size,
        channels_to_use=selected_channels
    )

    # Extract patches separately
    print("\n--- Extracting Training Patches ---")
    X_train, labels_train = segmentor.load_data(train_gdf)
    
    print("\n--- Extracting Test Patches ---")
    X_test, labels_test = segmentor.load_data(test_gdf)

    # Process Labels
    y_train_enc, y_test_enc, y_train_cat, y_test_cat, label_encoder = segmentor.preprocess_labels(labels_train, labels_test)

    # AUGMENTATION
    X_train_aug, y_train_cat_aug = segmentor.augment_data(X_train, y_train_cat)
    # y_train_enc musst be doubled for class_weights
    y_train_enc_aug = np.concatenate((y_train_enc, y_train_enc), axis=0)

    # TRAINING & EVALUATION
    model = segmentor.build_cnn(num_classes=len(label_encoder.classes_))
    segmentor.train_cnn(model, X_train_aug, y_train_cat_aug, y_train_enc_aug, X_test, y_test_cat)
    segmentor.evaluate_cnn(model, X_test, y_test_cat, label_encoder)
    
    print(f"\n--- DONE! Results can be found in: {segmentor.result_folder} ---")


if __name__ == "__main__":
    main()