# **Seismic Phase Picker ML Pipeline**

A comprehensive machine learning pipeline for automated P-wave detection in seismic data using physics-informed deep learning and adaptive U-Net architecture.

### **Overview**

This repository contains a complete end-to-end pipeline for detecting P-wave arrivals in seismic waveforms. The system combines traditional seismological feature extraction with modern deep learning techniques to achieve accurate and robust phase picking.

##### **Key Features**

- Dual-mode operation: Run with physics-informed features or raw waveforms only
- Adaptive U-Net architecture with attention mechanisms for improved feature focus
- Automated data mining from NCEDC public seismic data repository
- Physics-informed feature extraction including STA/LTA ratios, frequency bands, and envelope analysis
- SQLite database for efficient data storage and retrieval
- Comprehensive visualization tools for features and predictions

##### Requirements

```
pip install torch numpy scipy matplotlib tqdm
pip install obspy seisbench boto3 sqlite3
```

##### Quick Start

1. **Data Collection**

First, mine seismic data from the NCEDC repository:

```
python data_mine.py
```

This will:

- Connect to the NCEDC public S3 bucket
- Download waveforms for predefined earthquakes
- Process and store data in SQLite database (`seismic_data_2.db`)
- Apply PhaseNet for the initial P-wave pick reference

1. **Model Training**

Train the phase picker model:

```
python ML-pipeline.py
```
Configure training mode by editing the configuration section in ML-pipeline.py:
```
# Toggle physics-informed features ON/OFF
USE_PHYSICS_FEATURES = True   # Set to False for raw waveform only

# Toggle max amplitude feature ON/OFF  
USE_MAX_AMPLITUDE = False

# Set number of training epochs
TRAINING_EPOCHS = 200
```

##### Architecture

**Data Mining (`data_mine.py`)**
- Accesses NCEDC public seismic data via AWS S3
- Processes earthquakes from California and Nevada
- Applies instrument response removal
- Stores processed waveforms in SQLite database

**ML Pipeline (`ML-pipeline.py`)**
The pipeline implements:

1. Physics-Informed Features:

- Multi-scale STA/LTA ratios
- Frequency band decomposition (1-5 Hz, 5-15 Hz, 15-45 Hz)
- Envelope and instantaneous phase analysis
- Max amplitude features with configurable windows


2. Adaptive U-Net Model:

- Encoder-decoder architecture with skip connections
- Attention blocks for feature refinement
- Batch normalization and dropout for regularization
- Learnable feature weighting for physics-informed mode


3. Training Features:

- Early stopping with patience
- Learning rate scheduling
- Class-weighted loss function
- Gradient clipping for stability

##### Database Schema
The SQLite database contains two tables:

- earthquakes: Event metadata (time, location, magnitude, depth)
- waveforms: Seismic traces with pick times and station information

##### Performance Metrics
The model evaluates pick accuracy using:

- Mean/median absolute error in seconds
- Performance categories (Excellent: <0.5s, Good: 0.5-1s, Fair: 1-2s, Poor: >2s)
- Distribution analysis and visualization

##### Output Files
After training, the pipeline generates:

- `best_adaptive_model.pth`: Best model checkpoint
- `features`: Visualizations of extracted features
- `predictions`: Model prediction plots with ground truth comparison
- Training loss curves and error distribution plots

##### Customization
Adding New Earthquakes
Edit the earthquakes list in `data_mine.py`:

```
earthquakes = [
    {
        "time": 'YYYY-MM-DD HH:MM:SS',
        "lat": latitude,
        "lon": longitude, 
        "mag": magnitude,
        "depth": depth_km,
        "radius": search_radius_km,
        "name": "Event Name"
    }
]
```

##### Modifying Network Architecture
Adjust the U-Net depth and feature channels in `ML-pipeline.py`:
```
model = AdaptiveUNet1D(
    in_channels=n_input_channels,
    out_channels=2,
    features=[32, 64, 128, 256],  # Modify encoder/decoder depths
    dropout=0.1
)
```



