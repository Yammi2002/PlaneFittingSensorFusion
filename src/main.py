import json
import open3d as o3d
from utils.Dataloader import get_sensors_for_sample
from utils.GeometryEngine import compute_point_cloud, color_point_cloud

"""
Main Pipeline Script: Sensor Fusion & Road Estimation.

This script orchestrates the entire perception pipeline for a specific scene sample.
It performs the following high-level operations:
1. Data Retrieval: Fetches synchronized LiDAR, Camera, and Radar data from the dataset.
2. Geometric Fusion: Aggregates multi-sensor point clouds into a single unified Ego-Vehicle frame.
3. Photometric Fusion: Projects 3D points onto camera images to assign RGB colors (Texture Mapping).
4. Scene Analysis: Uses RANSAC algorithm to segment the ground plane (Road) from obstacles.
5. Visualization: Renders the final 3D scene using Open3D.
"""

def main():
    
    # --- 1. CONFIGURATION & METADATA LOADING ---
    # Define paths to metadata files (Ensure these match your local directory structure)
    path_sample_data = "../data/metadata/man-truckscenes/v1.1-mini/sample_data.json"
    path_calib_data = "../data/metadata/man-truckscenes/v1.1-mini/calibrated_sensor.json"

    print("--- 1. Loading Metadata Database ---")
    with open(path_sample_data) as f:
        data = json.load(f)
        
    # Target Sample Token (Represents a specific timestamp in the scene)
    target_token = "deb7b3f332f042d49e7636d6e4959354" 
    
    print(f"Retrieving sensors for sample token: {target_token}")
    
    # Extract synchronized file paths for all available sensors
    lidars, cameras, radars = get_sensors_for_sample(data, target_token)
    
    print(f"  -> Found: {len(lidars)} LiDARS, {len(cameras)} Cameras, {len(radars)} Radars")

    # --- 2. GEOMETRIC FUSION (Ego-Frame Aggregation) ---
    print("\n--- 2. Geometric Fusion (Point Cloud Generation) ---")
    
    # Combine LiDAR and Radar points into a single coordinate system (Ego-Vehicle)
    # We pass the calibration path to compute extrinsic transformations (R|T)
    total_points = compute_point_cloud(path_calib_data, lidars, radars)

    print(f"  -> Aggregated Cloud Size: {total_points.shape[0]} points")

    if len(total_points) == 0:
        print("Error: Generated point cloud is empty. Exiting.")
        return

    # --- 3. PHOTOMETRIC FUSION (Colorization) ---
    print("\n--- 3. Photometric Fusion (3D-to-2D Projection) ---")
    
    # Project 3D points onto the 6 camera images to extract RGB values.
    # Points not visible to any camera will remain grey (default).
    final_colors = color_point_cloud(total_points, cameras, path_calib_data)

    # --- 4. DATA STRUCTURE CREATION ---
    print("\n--- 4. Building Open3D Object ---")
    pcd_final = o3d.geometry.PointCloud()
    
    # Assign Geometry (XYZ) and Texture (RGB)
    pcd_final.points = o3d.utility.Vector3dVector(total_points)
    pcd_final.colors = o3d.utility.Vector3dVector(final_colors)
    
    # Create a reference coordinate frame (Origin 0,0,0 is the center of the Truck)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    
    # --- 5. SCENE ANALYSIS (Road Fitting via RANSAC) ---
    print("\n--- 5. Road Segmentation (RANSAC) ---")
    
    # Apply RANSAC to find the dominant plane (The Road).
    # distance_threshold=0.25: Points within 25cm of the plane are considered 'Road'
    plane_model, inliers = pcd_final.segment_plane(distance_threshold=0.25,
                                             ransac_n=3,
                                             num_iterations=2000)
    
    [a, b, c, d] = plane_model
    print(f"  -> Road Plane Equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Split the cloud into two sets:
    # 1. Inliers = The Road
    # 2. Outliers = Everything else (Obstacles, Buildings, etc.)
    road_cloud = pcd_final.select_by_index(inliers)
    obstacle_cloud = pcd_final.select_by_index(inliers, invert=True)

    # VISUALIZATION TWEAK: Paint the road Red to distinguish it from the rest.
    # Comment out this line if you want to see the real asphalt color.
    road_cloud.paint_uniform_color([1.0, 0, 0]) 

    # --- 6. VISUALIZATION ---
    print("\n--- 6. Rendering Scene ---")
    o3d.visualization.draw_geometries([road_cloud, obstacle_cloud, axis], 
                                      window_name="Sensor Fusion + Road Fitting",
                                      width=1280, height=720)
        
if __name__ == "__main__":
    main()