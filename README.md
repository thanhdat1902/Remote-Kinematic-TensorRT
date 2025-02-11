
# AAAI Project

This repository contains the implementation and supporting resources for research paper **"Remote Kinematic Analysis for Mobility Scooter Riders Leveraging Edge AI"** presented at the **AAAI Fall Symposium 2024**. The project involves **marker detection** and dataset processing, alongside performance analysis using various metrics. [here](https://doi.org/10.1609/aaaiss.v4i1.31808)

---

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Overview

This repository implements marker detection and dataset processing workflows, optimized using tools like **TensorRT**. The project evaluates models using inference scripts and visualizes results through detailed plots.

## Repository Link

Find the project repository [here](https://github.com/Mobility-Scooter-Project/Remote-Kinematic-TensorRT.git).

## Directory Structure

- **Marker-Detection/**: Contains scripts and resources for detecting markers in datasets.
- **Process Dataset/**: Includes tools and scripts for preprocessing datasets for marker detection.
- **Result-metrics/**: Scripts for analyzing and visualizing performance metrics of the trained models.
- **ultralytics/**: A submodule or library dependency for training and inference with YOLO models.
- **inference.py**: Runs inference on test datasets, using trained models.
- **main.py**: Entry point for executing the primary workflow.
- **plot.py**: Generates visualizations of model performance, such as precision-recall curves.
- **README.md**: You're here!
- **.gitignore**: Lists files and directories ignored by Git.

---

## Features

- **Optimized Marker Detection**: Uses TensorRT for accelerated inference.
- **Dataset Processing**: Supports large-scale datasets.
- **Performance Metrics**: Includes scripts for precision, recall, F1 score analysis.
- **Visualization**: Tools for plotting key metrics in a visual format.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mobility-Scooter-Project/Remote-Kinematic-TensorRT.git
   cd AAAI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add submodules (if necessary):
   ```bash
   git submodule update --init --recursive
   ```

---

## Usage

1. Preprocess the dataset:
   ```bash
   python Process\ Dataset/process.py --input <path_to_data> --output <output_path>
   ```

2. Run marker detection:
   ```bash
   python Marker-Detection/detect.py --model <model_path> --dataset <path_to_data>
   ```

3. Perform inference:
   ```bash
   python inference.py --model <model_path> --data <test_data>
   ```

4. Visualize results:
   ```bash
   python plot.py --results <results_path>
   ```

---

## License

This project is licensed under [MIT License](LICENSE).
