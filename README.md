# GraphCast Lite (PyTorch)

## Overview

GraphCast Lite is a lightweight PyTorch adaptation of DeepMind's GraphCast architecture designed to run on local GPUs such as:

- NVIDIA RTX A1000 8GB
- RTX 3060 / 4060
- Colab GPUs

The project preserves the scientific and geometric logic of GraphCast while simplifying:

- JAX → PyTorch
- xarray → `.npy`
- TPU distributed execution → single GPU training

---

# Architecture

```text
ERA5 .npy
downarrow
Dataset Loader
downarrow
Grid Features
downarrow
Grid → Mesh GNN
downarrow
Mesh Message Passing
downarrow
Mesh → Grid GNN
downarrow
Prediction [B,C,H,W]
```

---

# Main Features

* Icosahedral spherical mesh
* Graph Neural Network message passing
* Grid2Mesh / Mesh2Mesh / Mesh2Grid
* Residual learning
* Latitude weighted loss
* Autoregressive forecasting
* TISR forcings
* ERA5 compatible
* PyTorch AMP mixed precision
* Multi-step forecasting support

---

# Dataset Format

Inputs:

```text
X shape:
[B,T,C,H,W]
```

Targets:

```text
Y shape:
[B,C,H,W]
```

Where:

* B = batch size
* T = input timesteps
* C = channels
* H,W = spatial resolution

---

# Channels

The 189 channels contain:

## Surface Variables

* 2m temperature
* mean sea level pressure
* 10m u wind
* 10m v wind

## Pressure Level Variables (37 levels)

For:

* temperature
* geopotential
* u wind
* v wind
* specific humidity

---

# Project Structure

```text
graphcast_lite/
│
├── autoregressive_lite.py
├── config.py
├── dataset.py
├── deep_typed_graph_net_lite.py
├── graphcast_lite.py
├── grid_mesh_connectivity_lite.py
├── icosahedral_mesh_lite.py
├── losses_lite.py
├── mlp_lite.py
├── model_utils_lite.py
├── normalization_lite.py
├── predictor_base_lite.py
├── solar_radiation_lite.py
├── train.py
├── typed_graph_lite.py
├── typed_graph_net_lite.py
└── README.md
```

---

# Installation

## Create environment

```bash
conda create -n graphcastlite python=3.10
conda activate graphcastlite
```

## Install PyTorch

CUDA 12 example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Install dependencies

```bash
pip install numpy scipy trimesh tqdm
```

---

# Training

Run:

```bash
python train.py
```

---

# Default Configuration

| Parameter             | Value |
| --------------------- | ----- |
| Mesh Size             | 2     |
| Latent Size           | 64    |
| Message Passing Steps | 4     |
| Batch Size            | 1     |
| Resolution            | 64×64 |
| Input Timesteps       | 2     |
| Channels              | 189   |

---

# GPU Requirements

Recommended minimum:

* 8 GB VRAM
* 16 GB RAM

Tested target:

* NVIDIA RTX A1000 8GB

---

# Main Differences vs Original GraphCast

| Original GraphCast  | GraphCast Lite |
| ------------------- | -------------- |
| JAX + Haiku + Jraph | PyTorch        |
| xarray Dataset      | `.npy` tensors |
| TPU distributed     | Single GPU     |
| mesh_size=5/6       | mesh_size=2/3  |
| Huge memory usage   | Reduced memory |
| Production scale    | Local training |

---

# Scientific Logic Preserved

The following concepts remain identical to the original GraphCast:

* Icosahedral mesh
* Spherical geometry
* Relative edge positions
* Local coordinate rotations
* Message passing
* Residual learning
* Latitude weighted losses
* Autoregressive rollout
* ERA5 physical variables

---

# Output

Model output:

```text
[B,C,H,W]
```

Example:

```text
[1,189,64,64]
```

---

# Future Improvements

* Multi-GPU training
* Dynamic pressure-level weighting
* Regional high-resolution meshes
* Graph attention layers
* Long-horizon forecasting
* Distributed inference
* WeatherBench evaluation

---

# Author

GraphCast Lite adaptation for local GPU training based on the original DeepMind GraphCast architecture.
