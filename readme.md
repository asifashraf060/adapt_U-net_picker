# **Feature-Enhanced Neural Networks for Seismic Phase Detection**

This repository implements a complete workflow for **automated P-wave phase picking** using a **feature-augmented U-Net model**. The workflow demonstrates how adding **signal-derived features**—such as STA/LTA ratios, spectral envelopes, and amplitude volatility—can dramatically improve model accuracy, interpretability, and data efficiency, even with small training datasets.

### 🧭 **Overview**
The pipeline consists of three major stages:
1. **Data Mining** (`data-mine.py`) — Automatically mines waveform data for earthquakes listed in a CSV catalog (e.g., USGS catalog export), processes traces from FDSN servers, and stores them in a structured SQLite database with PhaseNet-derived reference picks.
2. **Manual Picking** (`picker.py`) — Interactive waveform inspection and manual correction of P-wave picks.
3. **Model Training and Evaluation** (`ML-pipeline.py`) — Trains a 1D U-Net model on raw and signal-derived features and evaluates its performance on different dataset sizes.
The workflow is modular: you can run each component independently or chain them for full reproducibility—from catalog data to a trained and evaluated model.

### ⚙️ **Installation**
Clone this repository and install dependencies:
```
pip install -r requirements.txt
```

### 🗂️ **Pipeline Workflow**

#### 1. ** Data Mining --** `data-mine.py`
This script downloads, processes, and stores waveform data from FDSN servers for earthquakes listed in a CSV catalog—including those exported directly from the [USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/

**Inputs**
- A CSV catalog file downloaded from USGS (or any similar source) containing:
### **Inputs**

| Column       | Description                                    |
|:--------------|:-----------------------------------------------|
| `time`        | Origin time of the earthquake in UTC           |
| `latitude`    | Event latitude in decimal degrees              |
| `longitude`   | Event longitude in decimal degrees             |
| `depth`       | Hypocentral depth in kilometers                |
| `mag`         | Magnitude of the event                         |
| `place`       | Event location or region description           |
| `id`          | Unique event identifier (optional)             |

**Example input file:** `query.csv` — a CSV file exported directly from the [USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/).

Example filename: `query.csv`

**Process**
1. Creates a **SQLite database** (`seismic_data.db`) with tables for earthquake metadata and waveform records.
2. Reads the catalog and queries multiple FDSN data centers (`IRIS`, `NCEDC`, `SCEDC`) for waveform availability within a user-defined radius.
3. Downloads traces for selected channels (e.g., `HNE`, `HNN`, `HNZ`) and automatically removes instrument response using available station metadata.
4. Applies **PhaseNet** (via `seisbench`) to estimate initial P-picks for reference.
5. Stores waveform data, pick times, and metadata (network, station, channel, sampling rate, etc.) in the database.

**Configurable Parameters**
Set these directly in the script or as command-line arguments:
```
csv_path='query.csv'
db_path='seismic_data.db'
radius_km=250
channels=['HNE', 'HNN', 'HNZ']
pre_time=3
post_time=120
```

**Outputs**
- **SQLite database** containing:
	- `earthquake` table -- event time, location, magnitude, and ID
	- `waveforms` table -- waveform data (as serialized arrays), sampling rate, PhaseNet picks, and event-station metadata
Example Output:
```
seismic_data.db
├── earthquakes (event metadata)
└── waveforms (serialized traces, picks, and attributes)
```

#### 2. **Manual Picking --** `picker.py`
A lightweight, interactive waveform visualization tool for verifying and correcting P-wave picks.

**Inputs**
- `seismic_data.db` (from `data-mine.py`)

**Features**
- Visualize waveforms and existing picks (manual or PhaseNet).
- Add, move, or confirm picks using mouse/keyboard.
- Directly writes updated pick times to the database (`manual_p_pick_time` field).

**Controls**

| Action               | Key / Mouse       | Description                              |
|:---------------------|:------------------|:-----------------------------------------|
| Place manual pick    | Mouse click       | Marks a P-wave arrival on the waveform   |
| Accept existing pick | `a`               | Confirms the PhaseNet or auto pick       |
| Skip trace           | `s`               | Skips the current waveform               |
| Navigate next / prev | `→` / `←`         | Move to next or previous waveform        |
| Save and quit        | `q`               | Saves all picks and closes the program   |

#### 3. **Model Training and Evaluation--** `ML-pipeline.py`
Trains a **feature-enhanced U-Net model** that combines the raw waveform with engineered signal-derived features.

**Inputs**
- `seismic_data.db` from previous steps
- Configurable parameters (inside the script):
```
use_signal_features = True
epochs = 100
batch_size = 8
learning_rate = 1e-4
channels = ['HNE', 'HNN', 'HNZ']
```

**Feature Set**

Each trace is augmented with:

- Multi-scale STA/LTA ratios
- Hilbert envelope and derivative
- Instantaneous frequency
- Bandpass energy/envelope (low, mid, high frequency)
- Maximum amplitude volatility

**Model Architecture**

- U-Net encoder–decoder with skip connections and attention blocks
- Learnable feature weights to dynamically scale feature importance
- Cross-entropy loss with class weighting
- Early stopping and learning-rate scheduling

**Outputs**
- best_adaptive_model.pth: trained model weights
- performance_plots: training loss and error distributions
- predictions: waveform overlays with true and predicted picks
- feature_weights: bar charts showing learned feature importance

### 🧩 **Customization**

- Change radius or channels directly in data-mine.py:
```
radius_km = 200
channels = ['HNZ']
```
- Add new catalog events:
Use any CSV downloaded from the USGS Earthquake Catalog with matching columns.

- Adjust network depth in ML-pipeline.py:
```
features = [32, 64, 128, 256]
```

### 🚀 **Usage Example**

```
# Step 1: Download USGS catalog (query.csv) and mine data
python data-mine.py

# Step 2: Optionally verify picks manually
python picker.py

# Step 3: Train and evaluate the model
python ML-pipeline.py
```

💡 Future Extensions

Although this study uses a simple U-Net to isolate the impact of signal-derived features, the same workflow can be extended using advanced models such as Transformer-based hybrid architectures to further improve performance, context sensitivity, and phase-picking accuracy.