# UAV Wildfire Recovery Monitoring

> **A deep-learning time-series analysis of vegetation recovery following a wildfire in Patras, Greece.**

## 📖 Project Overview
This research investigates how different landcover classes (vines, olive trees, soil, and burned areas) develop and recover over time after a wildfire event. Using high-resolution UAV (Unmanned Aerial Vehicle) imagery, we analyze temporal changes through RGB-based vegetation indices (ExG, VARI, NGRDI) and structural data (Canopy Height Models). 

By leveraging both Convolutional Neural Networks (CNNs) and Random Forest (RF) models for precise landcover classification and subsequently monitoring post-fire recovery for different land cover classes of interest across multiple flight dates, this project aims to provide quantitative insights into post-fire ecosystem recovery.

## 🗂️ Dataset Description
The study area is located in a mixed agricultural zone (vineyards and olive groves) in **Patras, Greece**. The first flight was conducted shortly after the wildfire, capturing the immediate damage, followed by several monitoring flights in the following months.

* **Sensor:** DJI Mavic 3 Enterprise - RGB Camera
* **Spatial Resolution:**  ~1.5 cm to ~2.5 cm/pixel
  *(Note: Native Ground Sampling Distance (GSD) varies slightly across flight dates, ranging from ~1.5 cm to ~2.5 cm/pixel. All datasets were standardized to a uniform resolution during preprocessing).*
* **Processing Software:** Pix4Dmapper (Version 4.8.4) for August and September 2025 images and ArcGIS Drone2Map (Version 2025.2.1) for October 2025 and February 2026 images
* **Provided Data:** RGB Imagery + standalone Digital Surface Model (DSM)
* **Flight Dates:**
  * `21.08.2025` (Immediate post-fire)
  * `01.09.2025`
  * `26.10.2025`
  * `04.02.2026`
  * `22.02.2026`
  * *(Ongoing monitoring...)*

*Visual examples of the study area over time:*
<p align="center">
  <img src="docs/images/250821_original.png" width="30%">
  <img src="docs/images/251026_True_Ortho.png" width="30%">
  <img src="docs/images/260222_original.png" width="30%"><br>
  <b>Fig 1:</b> Immediate Post-Fire (21.08.2025) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Fig 2:</b> 2 Months Recovery (26.10.2025) <b>Fig 3:</b> 6 Months Recovery (22.02.2026)
</p>

## ⚙️ Methodology & Pipeline

The pipeline is structured into several sequential processing steps to ensure spatial accuracy and meaningful feature extraction before applying both the Random Forest (RF) and Deep Learning (CNN) models for Landcover Classification.

### Step 1: Data Preparation, Coregistration & Resampling
To perform accurate time-series analysis, all pixels across all dates must align perfectly. If a pixel represents a specific area in August 2025, the exact same pixel must represent the exact same area in February 2026.

1. **Data Stacking:** The separate DSM is stacked with the RGB imagery to form a 4-channel (R-G-B-DSM) composite for each flight date.
2. **Master Reference Selection:** The image from **October 26, 2025**, serves as the master reference. This mid-sequence date was selected to minimize the relative temporal and structural shift between all images and to assure similarity to each image.
3. **Coregistration ([AROSICS](https://github.com/GFZ/arosics)):** To guarantee high-precision spatial alignment, we employ the AROSICS (Automated and Robust Open-Source Image Co-Registration Software) algorithm in a two-step approach:
   * **Global Coregistration:** First, a global spatial X/Y shift is calculated and applied to correct for the absolute geolocation error between the target images and the master reference.
   * **Local Coregistration:** Subsequently, the globally corrected images are coregistered using the local coregistration algorithm. This accounts for fine-scale, localized geometric distortions and sub-pixel misregistrations.
4. **Resampling:** Finally, all images are reprojected and resampled to a strict unified spatial resolution of **2.5 cm/pixel** using `rasterio` (applying `cubic` interpolation). This finalizes the coregistration process, guaranteeing that every image grid is perfectly congruent and ensuring true pixel-to-pixel correspondence across the entire temporal sequence.

### Step 2: Canopy Height Model (CHM) Derivation
The original DSM captures the absolute elevation (including the terrain). To isolate the height of the vegetation (vines and olive trees), we mathematically simulate a Digital Terrain Model (DTM) and subtract it from the DSM.

* **Method:** Sliding-window morphological filtering (~8m window size to bridge tree crowns).
* **Process:**
  1. *Erosion (Minimum Filter):* Removes the above-ground vegetation to find the bare earth.
  2. *Dilation (Maximum Filter):* Corrects artificial terracing on sloped terrain.
  3. *Gaussian Blur:* Smooths the generated DTM for natural transitions.
* **Equation:** `CHM = DSM - DTM`
* **Post-Processing:** Negative height artifacts are clipped to 0,0 and image boundaries are safely padded with `-9999.0` NoData values to prevent morphological border artifacts.

### Step 3: AOI Cropping & Feature Engineering
Before proceeding to classification, the data boundaries needed to be strictly synchronized, and additional spectral features were added to provide the neural network with more predictive power.

1. **Area of Interest (AOI) Cropping:** The spatial extent varies between flight campaigns. To ensure a robust multi-temporal comparison, all datasets were clipped to a common Area of Interest (AOI). This AOI was defined by the first dataset (August 2025), which possessed the smallest spatial footprint (~6.94 ha / 17.16 acres) compared to later flights (e.g., ~39.3 ha in Feb 2026). Furthermore, this clipping safely removes edge artifacts and areas with missing Canopy Height Model (CHM) data that occur along the flight boundaries.
2. **RGB Vegetation Indices:**
   To emphasize vegetative health and recovery over time, three specific RGB-based vegetation indices were derived for all datasets (inspired by [Rodrigo-Comino et al., 2026](https://doi.org/10.1007/s11119-026-10363-4)):
   * **ExG** (Excess Green)
   * **VARI** (Visible Atmospherically Resistant Index)
   * **NGRDI** (Normalized Green Red Difference Index)
3. **Standardization & Final Composite:**
   All bands were linearly standardized to a strict `[0, 1]` range for later CNN application. This results in a final, uniform **7-channel composite** `(R, G, B, CHM, ExG, VARI, NGRDI)` for every single flight date.

### Step 4: Object-Based Segmentation & Automated + Manual Labeling
Since the study's core objective is to track the multi-temporal recovery of static landcover units (e.g., a burned vineyard, olive trees, etc.), the spatial baseline classification only needed to be established once, using the initial (August 2025) image. 

#### 1. Object-Based Image Analysis (OBIA)
Following the methodology outlined by [Ivošević et al., 2025](https://doi.org/10.1038/s41597-025-04437-7), we opted for Object-Based Image Analysis rather than pixel-based classification.
* **Algorithm:** Large-Scale Mean Shift (LSMS), executed via the Orfeo ToolBox (OTB), adapted from [Pajevic's implementation](https://github.com/pajevicnina/inspire1-seg).
* **Rationale:** LSMS clusters pixels into homogeneous, meaningful physical objects (polygons) based on spatial and radiometric proximity. This generates smoother boundaries and allows the classification model to utilize regional statistics (e.g., texture, mean, variance) rather than noisy, isolated pixel values.
* **Parameters:** A strict `range radius` of 3 was uniformly applied to fine-tune the sensitivity toward the high spatial dynamics of the 2.5 cm resolution imagery.
* 
#### 2. Feature Analysis: Class Separability
Before applying automated thresholds, the spectral and structural separability of the landcover classes was analyzed using extracted zonal statistics. The boxplots below visually demonstrate why the Canopy Height Model (CHM) and the Excess Green Index (ExG) were selected as the primary features for the heuristic labeling:

* **CHM (Structural Isolation):** Exhibits a stark contrast between Olive Trees and all other classes, making it the ideal candidate for isolating tall canopy structures from the ground and low-lying vines.
* **ExG (Spectral Isolation):** Shows a clear separation between photosynthetically active vegetation (Vital Vines and Trees) and non-vital surfaces (Bare Soil and Burned Areas).

<p align="center">
  <img src="docs/images/boxplot_chm.png" width="45%">
  <img src="docs/images/boxplot_exg.png" width="45%"><br>
  <b>Fig A:</b> CHM Feature Distribution &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Fig B:</b> ExG Feature Distribution
</p>

#### 3. Automated Heuristic Pre-Labeling (Pixel-Level)
To generate a baseline of training data without exhaustive manual labeling, we developed a two-step automated labeling heuristic utilizing the `OTSU threshold` implementation of `scikit-image`, directly leveraging the feature separability established above:

* **Tree Identification:** An Otsu threshold (CHM > `0.4199`) was calculated for the Canopy Height Model. Pixels exceeding this threshold reliably separated tall structures (Olive Trees, both healthy and burned) from ground features and lower-lying vegetation. 
* **Healthy Vine Identification:** A secondary Otsu threshold was applied specifically to the ExG channel (ExG > `0.4433`). This isolates all healthy, photosynthetically active vegetation. By subsequently subtracting the previously identified Tree pixels from this vegetation mask, the remaining pixels robustly identify Healthy Vines.

*Visualizing the pixel-level heuristic labels:*
<p align="center">
  <img src="docs/images/pixel_trees_chm.png" width="30%">
  <img src="docs/images/pixel_healthy_exg.png" width="30%">
  <img src="docs/images/pixel_vine_isolated.png" width="30%"><br>
  <b>Fig 1:</b> Trees (CHM Otsu) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Fig 2:</b> Healthy Veg (ExG Otsu) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Fig 3:</b> Isolated Vines
</p>

#### 4. Label Transfer to Polygons
The pixel-level heuristic masks were then spatially joined with the LSMS polygons generated in Step 3.1. A strict majority-rule was enforced: **If a polygon contained more than 80% pixels of a specific class (Tree or Vine), the entire polygon inherited that class label.**
This automated process rapidly yielded a strong foundational dataset of:
* **Tree Polygons:** `44,195` 
* **Vine Polygons:** `25,557`

*Visualizing the final automated object labels:*
<p align="center">
  <img src="docs/images/poly_trees.png" width="30%">
  <img src="docs/images/poly_vines.png" width="30%">
  <img src="docs/images/combined_overview.png" width="30%"><br>
  <b>Fig A:</b> Trees (Class 20) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Fig B:</b> Vines (Class 10) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Fig C:</b> Combined
</p>

#### 5. Manual Enrichment & Labeled Training Dataset
While the automated heuristic successfully mapped vital vegetation and trees, the remaining post-fire landscape features required manual classification. Using QGIS, the automated dataset was extensively enriched by manually labeling and correcting polygons across five distinct target classes:
* `10`: Vineyard (Vital)
* `15`: Vineyard (Burned)
* `20`: Olive Tree (Vital & Burned)
* `30`: Bare Soil
* `40`: Burned Area

Merging the automated heuristic labels with the manual QGIS annotations resulted in the final, comprehensive training dataset. The data exhibits a natural, real-world class imbalance, totaling over 160,000 labeled samples:

| Class ID | Landcover Class | Polygon Count | Share (%) |
| :--- | :--- | :--- | :--- |
| **-1** | Unlabeled (Target) | 370,494 | 69.06% |
| **30** | Bare Soil | 65,011  | 12.12% |
| **20** | Olive Tree | 45,836 | 8.54% |
| **10** | Vineyard (Vital) | 26,966 | 5.03% |
| **40** | Burned Area | 23,955 | 4.46% |
| **15** | Vineyard (Burned) | 4,253  | 0.79% |
| **Total** | | **536,515** | **100%** |

*(Polygons marked as `-1` serve as the target area for CNN prediction, while the remaining ~164k samples represent the robust training foundation).*

The complete segmented dataset consisted of 536,515 polygons, including **166,021 labeled polygons** for model development and **370,494 unlabeled target polygons** for subsequent full-map classification.
   
### Step 5: Deep Learning Landcover Classification (CNN)
With the labeled polygons and the 7-channel August 2025 image ready, a Convolutional Neural Network (CNN) was trained to classify the landscape. The architecture and training pipeline were adapted from [Rana Vaqcar's CNN implementation](https://github.com/ranavaqcar1989/CNN-Training-pipeline) and customized for this specific application.


#### 1. Spatially Separated Train–Validation–Test Design
A common pitfall in remote sensing is spatial data leakage, where nearby observations are randomly distributed across training and evaluation subsets. Because neighboring polygons originate from the same image and may share similar spectral and structural characteristics, such a random split can produce overly optimistic performance estimates.

To reduce optimistic performance estimates caused by spatial dependence between neighbouring observations, the labelled polygons were partitioned into spatial groups before model training.
The data were divided into three non-overlapping subsets:
- **Training set:** used to fit model parameters.
- **Validation set:** used for early stopping, model development, and comparison between CNN and Random Forest.
- **Test set:** kept untouched during model development and used once for final performance assessment.

The split was created using a Spatial Block-Split methodology:

- **Grid Clustering:** The Area of Interest was divided into predefined spatial grid blocks. Each labeled polygon was assigned to a block based on its geographic position.
- **Group-Based Assignment:** All polygons belonging to the same spatial block were assigned entirely to either the training, validation, or test subset. Individual polygons were therefore never randomly distributed across the subsets.
- **Spatial Guard Buffer:** Polygons located too close to a held-out validation or test area were removed from the respective training subset. This additional buffer reduces dependence between neighboring observations and lowers the risk that spatially overlapping or nearly adjacent CNN patches occur across subsets.
- **Shared Partitions:** The resulting train, validation, and test partitions were precomputed once and subsequently used for both the CNN and Random Forest workflows.

The table in the **Step 4.5**  describes the complete polygon dataset before spatial partitioning.
After applying the spatial block split and the 5 m guard buffer, **131,829 labeled polygons** remained for model development and evaluation: **77,067** for **training**, **21,749** for **validation**, and **33,013** for **testing**. The removal of nearby training observations reduced spatial dependence between the subsets while
largely preserving the original class distribution.

The spatial distribution of the three subsets is shown below. Training, validation, and test polygons occupy distinct parts of the study area, illustrating the block-based partitioning and the spatial 
separation between model fitting and evaluation data.

<p align="center">
  <img src="docs/images/Training_Split.png" width="23%">
  <img src="docs/images/Validation_Split.png" width="23%">
  <img src="docs/images/Test_Split.png" width="23%">
  <img src="docs/images/Combined_Splits.png" width="23%"><br>
  <b>Fig A:</b> Training subset &nbsp;&nbsp;
  <b>Fig B:</b> Validation subset &nbsp;&nbsp;
  <b>Fig C:</b> Test subset &nbsp;&nbsp;
  <b>Fig D:</b> Combined spatial partitions
</p>

This design provides a more conservative estimate of classification performance on spatially held-out areas within the same August 2025 orthomosaic. It is intended to evaluate the reliability of the landcover classification for this study area and acquisition, rather than transferability to different UAV campaigns, sensors, dates, or geographical locations.

#### 2. Patch Extraction & CNN Architecture
For each polygon, one 128 x 128 pixel patch was extracted around its geometric centroid, capturing the 7 spectral and structural channels (R, G, B, CHM, ExG, VARI, NGRDI). The same centroid-based extraction procedure was applied during training, validation, final testing, and full-map prediction. Polygons for which no complete and valid centroid patch could be extracted were excluded from CNN modelling and evaluation.

To ensure stable gradient descent and optimal model convergence, all input variables across the 7 channels were standardized to a [0, 1] range.

The CNN follows a robust, 4-block architecture designed for spatial feature extraction:

- **Feature Extraction:** Four sequential blocks consisting of 2D Convolutions (Conv2D with increasing filters: 32 → 64 → 128 → 256), BatchNormalization (to stabilize learning), and MaxPooling2D (to reduce spatial dimensions).
- **Classification Head:** A GlobalAveragePooling2D layer flattens the feature maps, followed by a fully connected Dense layer (256 units), Dropout (0.5 to prevent overfitting), and a final Softmax output layer mapping to the 5 target classes.

#### 3. Training Dynamics
To ensure the model learns effectively despite the heavily imbalanced dataset, several advanced training strategies were employed:
- **Class Weighting:** The loss function was penalized using balanced class weights, forcing the network to pay significantly more attention to minority classes (e.g., Burned Vines) compared to majority classes (e.g., Bare Soil).
- **On-the-Fly Data Augmentation:** Training patches were dynamically rotated and flipped. This artificially expands the dataset and makes the model invariant to the orientation of trees or vine rows.
- **Optimization:** Trained over a maximum of 200 epochs using the Adam optimizer (`lr=0.0005`), supported by `ReduceLROnPlateau` (to fine-tune learning as it converges) and `EarlyStopping` (to halt training and restore the best weights if validation loss stops improving).
- **Independent Final Evaluation:** The spatial test subset was not used for early stopping, architecture decisions, or model comparison. 

### Step 6: Random Forest Classification (Baseline Comparison)
To evaluate the Deep Learning results against a traditional Machine Learning approach, a Random Forest (RF) classifier was implemented, adapted from [Pajevic's implementation](https://github.com/pajevicnina/inspire1-seg).

- **Spatially Aligned Comparison:** The RF model used the same precomputed spatial training, validation, and test partitions as the CNN. This ensures that both approaches are evaluated on the same held-out parts of the study area. Because the CNN requires a complete and valid centroid patch, the number of evaluated CNN samples is additionally reported and checked against the corresponding RF support.
- **Feature Extraction:** While the CNN extracts spatial context directly from image patches, the RF relies on engineered statistical features. Zonal statistics (Mean and Variance) for all 7 channels were calculated for each polygon, resulting in 14 distinct numerical features per sample.
- **Hyperparameters:** The model was trained using 100 trees with a maximum depth of 15 and a minimum node size of 5.
- **Model Development:** The validation subset was used to assess RF performance and compare it with the CNN. The spatial test subset remained untouched until the final evaluation.


## 📊 Classification Results & Model Evaluation
The CNN and Random Forest were evaluated in two separate stages. First, the spatial validation subset was used for model development and model-type comparison. After that, both models were evaluated once on the independent spatial test subset.

Because the labeled dataset is imbalanced, **Macro-F1** was used as the primary model-comparison metric. It assigns equal importance to every landcover class regardless of its sample size. Balanced Accuracy, Overall Accuracy, and Cohen's Kappa are reported as complementary metrics.

### Validation Results

The following table summarizes performance on the spatial validation subset. These results were used to compare the CNN and Random Forest approaches.

| Model | Accuracy | Balanced Accuracy | Macro-F1 | Kappa |
|---|---:|---:|---:|---:|
| Random Forest | 94.07% | 86.97% | 88.30% | 91.44% |
| CNN | 94.74% | 92.92% | 92.50% | 92.50% |

*(Note: Detailed precision, recall, and F1-scores for each specific class can be found in the respective `rf_evaluation_results.txt` and `cnn_evaluation_results.txt` files generated during the validation phase).*

### Validation Confusion Matrices
The validation confusion matrices illustrate class-specific performance during model training. In addition to the overall number of correct predictions, they make it possible to identify systematic confusion between spectrally or structurally similar classes and to assess performance on minority classes such as burned vineyards.

<p align="center">
  <img src="docs/images/rf_validation_confusion_matrix.png" width="45%">
  <img src="docs/images/cnn_validation_confusion_matrix.png" width="45%"><br>
  <b>Fig A:</b> Random Forest Confusion Matrix &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Fig B:</b> CNN Confusion Matrix
</p>

The CNN achieved the stronger validation performance, with a slightly higher Overall Accuracy (0.9474 vs. 0.9407) and clearly better Balanced Accuracy (0.9292 vs. 0.8697) and Macro-F1 (0.9250 vs. 0.8830). The main advantage of the CNN was its more balanced class-specific performance, particularly for Burned Vineyard (class 15) and Burned Area (class 40), while both models performed very strongly for Vital Vineyard, Olive Tree, and Bare Soil.

## Final Test Results
The final test subset remained untouched throughout training, early stopping, hyperparameter selection, and model-family comparison. It was evaluated only after the CNN and RF configurations had been fixed. The resulting metrics therefore provide the final spatially separated performance estimate for the August 2025 orthomosaic.

| Model | Accuracy | Balanced Accuracy | Macro-F1 | Kappa |
|---|---:|---:|---:|---:|
| Random Forest | 93.31% | 84.56% | 84.87% | 90.62% |
| CNN | 93.71% | 93.38% | 92.52% | 91.24% |

*(Note: Detailed precision, recall, and F1-scores for each specific class can be found in the respective `rf_test_evaluation_results.txt` and `cnn_test_evaluation_results.txt` files generated during the final evaluation phase. The corresponding summary metrics are also stored in the generated JSON files.)*

## Final Test Confusion Matrices

The final test confusion matrices provide the class-specific assessment on spatially held-out data that were not used during model development.

<p align="center">
  <img src="docs/images/rf_test_confusion_matrix.png" width="45%">
  <img src="docs/images/cnn_test_confusion_matrix.png" width="45%"><br>
  <b>Fig A:</b> Random Forest Test Confusion Matrix &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Fig B:</b> CNN Test Confusion Matrix
</p>

The CNN also showed the stronger performance on the independent spatial test set, achieving a higher Overall Accuracy (0.9371 vs. 0.9331), Balanced Accuracy (0.9338 vs. 0.8456), and Macro-F1 (0.9252 vs. 0.8487). Its largest advantage was the reliable classification of the minority Burned Vineyard class, for which the CNN achieved an F1-score of 0.88 compared with 0.51 for the Random Forest, whereas both models remained highly accurate for the dominant classes.

Based on the predefined Macro-F1 criterion, the CNN should therefore be as the preferred classification approach for the subsequent landcover mapping and recovery analysis.

### Final Landcover Classification Maps
The true utility of the models is visualized when inference is run across the entire Area of Interest. The following maps represent the final spatial predictions, providing a high-resolution, quantitative overview of the post-fire ecosystem and the varying degrees of vegetation recovery.

<p align="center">
  <img src="docs/images/RF_Classified_Map.png" width="48%">
  <img src="docs/images/cnn_classified_map_final.png" width="48%"><br>
  <b>Map 1:</b> Final Prediction (Random Forest) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Map 2:</b> Final Prediction (CNN)
</p>

---

### Step 7: Temporal Evolution Analysis
Following the classification, we conducted a preliminary analysis to quantify ecosystem recovery. By leveraging the co-registered temporal stack of the 7-channel imagery, we examined the evolution of vegetation indices across five distinct flight dates, ranging from immediate post-fire conditions (August 2025) to six months of recovery (February 2026).

* **Methodology:** We analyzed the spectral trajectory of the ExG, VARI, and NGRDI indices for each landcover class. By masking the imagery with the final Random Forest classification map, we isolated the mean index values for each specific landcover type to visualize their respective recovery dynamics over time.
* **Objective:** This temporal assessment provides quantitative insights into the regenerative capacity of different vegetation types (specifically vital vs. burned vineyards) and serves as a baseline for understanding the ecological impact of the wildfire event on the local landscape.

#### Vegetation Index Trajectories (ExG, VARI, NGRDI)
The plot below illustrates the mean index values for each landcover class throughout the monitoring period. The diverging trends underscore the distinct recovery rates between photosynthetically active vegetation and soil/burned surfaces.

<p align="center">
  <img src="docs/images/temporal_evolution_indices.png" width="90%"><br>
  <b>Fig 4:</b> Temporal evolution of vegetation indices per landcover class, documenting ecosystem recovery from August 2025 to February 2026.
</p>

---
*Repository structure, processing scripts, and further temporal monitoring documentation will be expanded as the research progresses.*