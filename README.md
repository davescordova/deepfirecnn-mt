# 🔥 DeepFireCNN: Wildfire Risk Prediction in Mato Grosso

![Status](https://img.shields.io/badge/Status-Production-success)
![Framework](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C)
![Environment](https://img.shields.io/badge/Environment-WSL%20%7C%20Ubuntu-orange)

## 📌 1. Project Overview
**DeepFireCNN** is an advanced, end-to-end Deep Learning pipeline designed to predict the spatial probability of wildfires in the state of Mato Grosso, Brazil. By combining historical climate data, satellite imagery, and anthropogenic variables, the model generates highly accurate, multi-horizon fire risk maps for **1, 7, 14, and 30 days** into the future.

The model's theoretical foundation is directly aligned with the **IPCC AR6 Synthesis Report**, mathematically modeling the regional "forming processes" of wildfires by capturing *Fire Weather* (via Vapor Pressure Deficit and Temperature), *Ecological Drought* (via Soil Moisture), and *Anthropogenic Vulnerability* (via land use and road proximity).

---

## 🧬 2. The Forming Process & Data Architecture
To accurately predict fire occurrences, the neural network ingests three distinct layers of geographic and atmospheric realities. Data is automatically fetched and synchronized via **Google Earth Engine (GEE)**:

### A. The Fuel (Biosphere & Topography) - *Static & Slow-Changing*
* **LULC (Land Use and Land Cover):** MapBiomas (Forest, pasture, agriculture).
* **Vegetation Vigor:** MODIS EVI (Enhanced Vegetation Index).
* **Physiography:** SRTM 30m (DEM/Altitude, Slope, Aspect).

### B. The Predisposition (Atmosphere & Hydrology) - *Dynamic Time-Series*
* **Fire Weather:** ECMWF ERA5-Land (Temperature, Precipitation, Dewpoint, and **VPD** - Vapor Pressure Deficit).
* **Ecological Drought:** NASA SMAP L3 Active/Passive (Soil Moisture).
* **Biomass Emissions:** Copernicus Sentinel-5P (Carbon Monoxide).

### C. The Ignition (Anthropogenic Vector) - *Spatial Proximity*
* **Human Infrastructure:** Euclidean distance to Federal Highways (DNIT) and State Highways (INTERMAT), acting as a proxy for the expansion of the agricultural frontier and human ignition sources.

**Target Variable:** MODIS MOD14A1 (Active Fire Ignitions) — Used as the binary target for classification (Fire vs. No Fire).

---

## 🧠 3. Deep Learning Architecture
The core of the predictive engine is a custom **Convolutional Neural Network (CNN)** built in PyTorch.

* **Spatio-Temporal Ingestion:** The model does not just look at a single day. It ingests a **14-day trailing sequence** of environmental variables alongside static geographic layers.
* **Inception Blocks:** Utilizes multi-scale convolutional filters to extract both micro-local patterns (e.g., a specific farm boundary) and macro-regional weather fronts simultaneously.
* **Focal Loss:** Wildfire datasets are severely imbalanced (millions of "no-fire" pixels vs. a few "fire" pixels). The model uses a Focal Loss function to heavily penalize errors on actual fire events, prioritizing high **Recall**.
* **Dynamic Thresholding:** Probability thresholds are dynamically adjusted (e.g., 0.07 instead of the standard 0.5) to maximize the detection of true positive anomalies.

---

## ⚙️ 4. Pipeline & Execution Flow (10 km vs. 1 km Resolution)

The project methodology is divided into two distinct phases: an exploratory/training phase built on a **10 km spatial resolution**, and a final deployment phase operating at a high-definition **1 km spatial resolution**.

### Phase 1: Exploratory Analysis & Training (10 km Resolution)
The root directory contains notebooks `1` through `5`, detailing the scientific pipeline and model validation using a broader 10 km grid to optimize computational efficiency during experimentation:

* **`1. Obtaining the data.ipynb`:** Connects to the GEE API to asynchronously extract, process, and export all historical environmental and atmospheric variables at a 10 km spatial resolution to Google Drive.
* **`2. Correlation Matrices.ipynb`:** Performs detailed statistical and geospatial analyses to assess the collinearity and predictive power of the variables (e.g., the relationship between Vapor Pressure Deficit and fire occurrences).
* **`3. Optimized Batched Training.ipynb`:** The core deep learning engine. It slices the massive 10 km `.tif` datasets into temporal `.pt` tensor sequences and trains the PyTorch CNN across the 1d, 7d, 14d, and 30d prediction horizons.
* **`4. Optimized model.ipynb`:** Handles model evaluation, focusing on Recall, False Negative Rates, and the application of dynamic thresholding for highly imbalanced datasets.
* **`5. Prediction and Cartography.ipynb`:** Executes the forward pass of the 10 km model and utilizes Matplotlib/Folium to generate the preliminary cartographic outputs.

### Phase 2: Final Deployment (1 km Resolution)
* **`Production/` folder:** This directory houses the ultimate, state-of-the-art version of the project. The entire pipeline architecture from Phase 1 was scaled up and retrained using a high-definition **1 km spatial resolution** grid. This folder contains the operational script (`Queimadas_production.ipynb`) that runs the final, high-accuracy inference, generating the 1200 DPI JPEG panels and interactive Folium HTML web maps.

---

## ☁️ 5. Infrastructure & Required Components
This pipeline utilizes a hybrid **Cloud-to-Local-to-Cloud** architecture. To run this project, the following components are integrated:

* **Google Earth Engine (GEE):** Used as the primary data extraction engine.
* **Google Drive:** Acts as the cloud storage bridge, receiving the heavy `.tif` files generated by GEE.
* **Rclone:** Mounts the Google Drive directly inside the Linux environment using a VFS (Virtual File System) cache to allow PyTorch to read cloud data smoothly.
* **Windows Subsystem for Linux (WSL / Ubuntu):** The entire Python, PyTorch, and geospatial environment runs natively in Linux on a Windows machine to ensure maximum compatibility and GPU acceleration.

> **⚠️ Rclone Cache Constraint:** Heavy I/O operations (like slicing raw `.tif` files into thousands of `.pt` tensors) must occur on the *physical local SSD*. Writing thousands of tiny files directly to the Rclone mount triggers a Google Drive API rate limit error.

---

## 📂 6. Repository Structure
> **⚠️ Note on File Sizes:** Due to GitHub's size constraints, large datasets, trained model weights (`.pth`), sliced tensors (`.pt`), and raw raster arrays (`.tif`) are strictly excluded from this repository via `.gitignore`. 

```text
├── Queimadas_Data/                  # Vector Shapefiles (MT_Municipios, Federal/State Highways)
├── 1. Obtaining the data.ipynb      # GEE data extraction (10 km)
├── 2. Correlation Matrices.ipynb    # Statistical analysis & collinearity checks
├── 3. Optimized Batched Training.ipynb # PyTorch temporal slicing & CNN training loop
├── 4. Optimized model.ipynb         # Validation, Focal Loss & Thresholding metrics
├── 5. Prediction and Cartography.ipynb # Cartographic visualization for the 10 km model
├── Production/              
│   └── Queimadas_production.ipynb   # FINAL OPERATIONAL MODEL (1 km resolution)
├── .gitignore                       # Strict rules preventing heavy data uploads
└── README.md                        # Project documentation
```

---

## ✍️ 7. Authorship & Credits
**DeepFireCNN** was developed as part of ongoing research in territorial intelligence and environmental monitoring at the Federal University of Mato Grosso.

* **Author:** Daves de Azevedo Cordova
* **e-mail:** daves.cordova@gmail.com
* **Institution:** POSGEO / Federal University of Mato Grosso (UFMT)

*Developed for continuous monitoring and territorial intelligence in the State of Mato Grosso.*
