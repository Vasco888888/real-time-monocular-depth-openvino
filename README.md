# Real-Time Spatial Geometry Reconstruction via Monocular Inference

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenVINO](https://img.shields.io/badge/OpenVINO-Runtime-orange?style=for-the-badge&logo=intel&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)

A real-time depth estimation and LiDAR simulation tool built with **Python**, **OpenCV**, and **OpenVINO**. This project transforms a standard 2D webcam feed into a 3D depth map and simulates a laser scanner for distance measurement.

## Demos

| Spatial Depth Map | LiDAR Scanning Mode |
| :---: | :---: |
| ![Depth Map](./assets/depth_demo.gif) | ![LiDAR Scanner](./assets/lidar_demo.gif) |

## Overview
This project demonstrates an end-to-end pipeline for **Monocular Depth Estimation**:

1.  **Input:** Captures a live video feed from a standard webcam.
2.  **Inference:** Uses the MiDaS Small model optimized with OpenVINO to predict relative depth. The model is compiled for Intel® Integrated Graphics (iGPU) to ensure real-time performance on consumer-grade hardware.
3.  **Post-Process:** Smoothing and normalization are applied to the raw depth map to reduce flicker and enhance visual clarity.
4.  **Application:**
    *   **Depth Map:** A heatmap visualization of the scene's depth.
    *   **LiDAR Scanner:** A simulated laser scanner that locks onto objects and calculates their distance to the camera in meters.

## Project Report
For a detailed technical analysis, including the specifics of the depth inference model, calibration methodology, and performance metrics, please refer to the full report:

**[Project Report (PDF)](./docs/depthinference.pdf)**

## Tech Stack

### **Core Pipeline (AI and Computer Vision)**
*   **OpenVINO Toolkit:** Optimizes and runs the deep learning model on the CPU/GPU for real-time performance.
*   **MiDaS (v2.1 Small):** A lightweight convolutional neural network (CNN) trained on multimodal datasets to estimate relative depth from a single image. Utilizes **Zero-shot cross-dataset transfer** capabilities, allowing the model to generalize to new environments without additional training.
*   **OpenCV:** Handles video capture, image preprocessing (resizing, color conversion), and the graphical user interface (GUI).

### **Hardware Acceleration**
*   **Target Hardware:** Optimized specifically for Intel® UHD Graphics.
*   **Efficiency:** By offloading the MiDaS inference to the integrated GPU via the OpenVINO GPU device plugin, the system maintains a high FPS while freeing up the CPU for application logic and HUD rendering.

### **Algorithms and Logic**
*   **Exponential Moving Average (EMA):** Applied to the raw depth map to smooth out frame-to-frame noise (`alpha = 0.2`).
*   **Non-Linear Depth Calibration:** A custom formula derived to map the unitless AI output to real-world meters:

$$ D = \frac{K}{(x - \text{offset})^{P}} $$

Where $K \approx 6482$, $P \approx 1.43$, and $OFFSET \approx 119$.

---

## How to Run

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Vasco888888/real-time-monocular-depth-openvino.git
    cd real-time-monocular-depth-openvino
    ```

2.  **Install dependencies**
    ```bash
    pip install opencv-python numpy openvino
    ```

3.  **Download the Model**
    Run the helper script to fetch the optimized OpenVINO IR files.
    ```bash
    python download_model.py
    ```

4.  **Run the Applications**
    *   **LiDAR Scanner (Distance Measurement):**
        ```bash
        python src/lidar_scanner.py
        ```
    *   **Depth Map (Visualization):**
        ```bash
        python src/depth_map.py
        ```

---

## Controls

### **LiDAR Scanner**
*   **`q`**: Quit the application.
*   **`m`**: Cycle Modes:
    *   **SCANNING:** The laser automatically sweeps back and forth.
    *   **AUTO-LOCK:** The laser tracks the depth of the center object.
    *   **MANUAL:** User controls the laser position.
*   **`w` / `s`**: Move the laser Up/Down (in Manual Mode).

### **Depth Map**
*   **`q`**: Quit the application.

---

## Project Structure

```bash
├── assets/                 # Demos for README
├── docs/
│   └── depthinference.pdf  # Technical report and analysis
├── models/
│   ├── MiDaS_small.xml     # OpenVINO Intermediate Representation (Topology)
│   └── MiDaS_small.bin     # OpenVINO Intermediate Representation (Weights)
├── src/
│   ├── depth_map.py        # Heatmap visualization logic
│   └── lidar_scanner.py    # LiDAR simulation and distance logic
├── download_model.py       # Script to download model weights
├── LICENSE                 # Project license
└── README.md
```

## Challenges and Limitations
*   **Relative Depth:** The MiDaS model estimates relative depth, meaning it knows what is in front, but not exactly how far away it is. The metric distance is an estimation based on calibration.
*   **Edge Noise:** The depth map often flickers at object boundaries. We mitigated this using a temporal smoothing filter (EMA).
*   **Generalizability:** The calibration constants ($K$, $P$, $offset$) are tuned for the sensor's **Effective Operating Range** (0.5m - 3.0m). Accuracy decreases significantly outside this window or in varied lighting conditions.

---

## Future Work
*   **Stereo-Vision Integration:** Implementing dual-camera support to eliminate the need for manual metric calibration ($K$).
*   **Edge-Preserving Filters:** Replacing the current EMA filter with a **Bilateral Filter** to reduce edge noise and flicker without softening object boundaries.

