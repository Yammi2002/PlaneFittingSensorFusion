# LiDAR, Radar & Camera Fusion with IRLS Ground Fitting

**A robust framework made with Python and open3d for autonomous vehicle perception, designed to fuse LiDAR, Radar, and Camera data into a unified 3D representation.**

This project implements a complete sensor fusion pipeline using the MAN TruckScenes dataset. It aligns multi-modal sensor data into a common Ego-Vehicle coordinate system, performs texture mapping for visualization, and executes real-time ground plane segmentation using the IRLS approach.

For each scene, data from LiDARs and Radars are combined to create a point cloud around the truck. Using intrinsic and extrinsic parameters, camera data is used to obtain color information, which is fused together to color the points in the scene.

To capture the road plane, the IRLS approach is used; this information is color-coded in the final visualization.

---

## Key Features

* **Data Retrieval & Management:** Efficient parsing of metadata (JSON) to synchronize sensor streams (LiDAR, Radar, Camera) based on temporal tokens.
* **Geometric Fusion:** Transformation of raw point clouds from sensor-local coordinates to the **Ego-Vehicle Frame** using extrinsic calibration matrices.
* **Photometric Fusion (Texture Mapping):** Projection of 3D points onto 2D camera image planes using intrinsic matrices to colorize the point cloud with RGB information.
* **Scene Understanding:** Real-time implementation of the IRLS approach to segment the road from obstacles.
* **Interactive Visualization:** A custom non-blocking **Open3D** player that renders the fused scene as a continuous 3D video.

---

## Project Structure

```bash
├── data/
│   ├── metadata/           # JSON files (sample.json, sample_data.json, calibrated_sensor.json)
│   └── sensordata/         # Raw sensor files (pcd, jpg)
├── src/
│   ├── utils/
│   │   ├── Dataloader.py   # JSON parsing and calibration extraction and sensor synchronization logic
│   │   └── GeometryEngine.py # Core math: Transformations, Projection, coloring
│   └── main.py             # Main execution loop and visualization
└── README.md
```

## Setup

Clone this repository

```bash
git clone https://github.com/Yammi2002/PlaneFittingLidar.git
cd PlaneFittingLidar
```
Create the data folders

```bash
mkdir -p data/metadata
mkdir -p data/sensordata
```
Download dataset "mini" from https://brandportal.man/d/QSf8mPdU5Hgj/downloads#/-/dataset.
After the download finishes, extract folder contents in the correct project folders (metadata and sensordata).

Install the necessary libraries
```bash
pip install -r requirements.txt
```
Move to src folder
```bash
cd src
```
In the file main.py you can change CURRENT_TOKEN configuration in order to change scene. The possible values are under sample.json (you should choose a "token" where the "prev" tag is empty).

Run the main.py file to run the project and visualize the result.
```bash
python main.py
```
