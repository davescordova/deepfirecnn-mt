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

## ⚙️ 4. Pipeline & Execution Flow
The project is divided into distinct execution phases:

1. **`Obtaining the data.ipynb`:** Connects to the GEE API, verifies local inventory, and dispatches asynchronous batch exports for all required raster layers at 10km and 1km resolutions to Google Drive.
2. **`Data Slicing`:** A local script that reads massive `.tif` files and slices them into thousands of manageable PyTorch `.pt` tensors.
3. **`Model Training`:** The PyTorch training loop that optimizes weights for the 1d, 7d, 14d, and 30d prediction horizons.
4. **`Production Inference (Step 5)`:** Executes the forward pass of the trained models on the most recent 14-day data patch. It outputs:
   * **GeoTIFFs:** Raw probability maps.
   * **Static Dashboard:** An ultra-high-resolution (1200 DPI) 1x4 JPG grid.
   * **Interactive Web Map:** A Folium/HTML map featuring precise BR/MT highway vector styling, municipal boundaries, and a Branca probability colormap.

---

## ☁️ 5. Infrastructure & Cloud Setup
This pipeline utilizes a hybrid **Cloud-to-Local-to-Cloud** architecture, heavily relying on **Windows Subsystem for Linux (WSL/Ubuntu)** and **Rclone**.

### Rclone VFS (Virtual File System) Constraints
Because Google Drive is mounted as a virtual local drive, specific engineering rules are enforced to prevent bottlenecks and API bans:
* **Heavy I/O Remains Local:** Slicing raw `.tif` files into thousands of small `.pt` tensors directly on the mounted Google Drive will result in an `HTTP 403: Rate Limit Exceeded` ban. Therefore, all `.pt` patch generation and training occur exclusively on the **physical local SSD**.
* **VFS Caching:** When PyTorch reads a `.tif` from the mount, Rclone caches it in Linux. Initial reads are network-bound, while subsequent reads are disk-bound.
* **Asynchronous Writing:** Python script completion (e.g., `[+] Saved: map.jpg`) only means the file was written to the local VFS cache. Rclone handles the actual cloud upload in the background.

**WSL Maintenance:**
To prevent the WSL virtual disk (`.vhdx`) from bloating over time, the Rclone cache must be cleared periodically:
```bash
rm -rf ~/.cache/rclone/vfs/*
rm -rf ~/.cache/rclone/vfsMeta/*
