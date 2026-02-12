import json
import time
import open3d as o3d
import numpy as np
from utils.Dataloader import get_sensors_for_sample
from utils.GeometryEngine import compute_point_cloud, color_point_cloud, fit_plane_irls

"""
Main Pipeline Script: Automatic Sensor Fusion Player with Road Segmentation.

This script iterates through a scene temporally, performing:
1. Geometric Fusion (LiDAR + Radar).
2. Photometric Fusion (Camera projection).
3. Road Segmentation (IRLS) in real-time.
4. Continuous 3D rendering.
"""

def main():
    
    # --- 1. CONFIGURATION & METADATA LOADING ---
    base_path = "../data/metadata/man-truckscenes/v1.1-mini"
    path_sample = f"{base_path}/sample.json"
    path_sample_data = f"{base_path}/sample_data.json" 
    path_calib = f"{base_path}/calibrated_sensor.json"

    CURRENT_TOKEN = "c697554e47f54d808bc04430ee0c096a" # Start Token (First frame of the scene)
    CAMERA_ZOOM = 0.1
    
    print("--- 1. Loading Metadata Database ---")
    
    with open(path_sample) as f:
        samples = json.load(f)
    
    samples_dict = {s['token']: s for s in samples}

    with open(path_sample_data) as f:
        data_sensor = json.load(f)

    # --- 2. INITIALIZATION ---
    
    # Setup Non-Blocking Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Truck Player", width=1280, height=720)
    
    # Initialize TWO empty point clouds: one for Road, one for Obstacles
    pcd_road = o3d.geometry.PointCloud()
    pcd_obstacles = o3d.geometry.PointCloud()
    
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    
    # Add geometries to the visualizer
    vis.add_geometry(axis)
    vis.add_geometry(pcd_road)      # Container for the road (Red)
    vis.add_geometry(pcd_obstacles) # Container for obstacles (RGB)

    # Visualization settings
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0

    print("\n--- STARTING PLAYBACK ---")
    print("Press 'Q' to exit.")

    first_frame = True

    # --- 3. TEMPORAL LOOP ---
    while CURRENT_TOKEN != "":
        start_time = time.time()
        
        if CURRENT_TOKEN not in samples_dict:
            break

        print(f"Processing Sample: {CURRENT_TOKEN}")

        # A. Data Retrieval
        lidars, cameras, radars = get_sensors_for_sample(data_sensor, CURRENT_TOKEN)
        
        # B. Geometric Fusion
        total_points = compute_point_cloud(path_calib, lidars, radars)
        
        if len(total_points) > 0:
            # C. Photometric Fusion (Colorize Cloud)
            final_colors = color_point_cloud(total_points, cameras, path_calib)

            # D. IRLS segmentation
            # Create a temporary Open3D object just for calculation
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(total_points)
            temp_pcd.colors = o3d.utility.Vector3dVector(final_colors)

            # Apply IRLS to find the road plane
            # distance_threshold=0.20 (20cm tolerance)
            plane_model, inliers = fit_plane_irls(total_points, n_iter=10, threshold=0.10)

            # Split Data
            road_cloud = temp_pcd.select_by_index(inliers)
            obstacle_cloud = temp_pcd.select_by_index(inliers, invert=True)

            # Paint Road RED for visualization clarity
            road_cloud.paint_uniform_color([1.0, 0, 0])

            # E. Visualizer
            # Update the persistent objects with new data
            pcd_road.points = road_cloud.points
            pcd_road.colors = road_cloud.colors
            
            pcd_obstacles.points = obstacle_cloud.points
            pcd_obstacles.colors = obstacle_cloud.colors

            # Notify visualizer
            if first_frame:
                vis.add_geometry(pcd_road)
                vis.add_geometry(pcd_obstacles)
                vis.add_geometry(axis)

                # --- Camera settings
                ctr = vis.get_view_control()
                
                # 1. Set camera center
                ctr.set_lookat([0, 0, 0]) 
                
                # 2. Set the Z axis
                ctr.set_up([0, 0, 1])
                
                # 3. Set camera front side
                ctr.set_front([-1.0, 0.0, 0.5]) 
                
                # 4. Zoom, lower values put camera closer
                ctr.set_zoom(CAMERA_ZOOM) 
                
                first_frame = False
            else:
                vis.update_geometry(pcd_road)
                vis.update_geometry(pcd_obstacles)
                vis.update_geometry(axis)

            vis.poll_events()
            vis.update_renderer()

        # F. Navigation & Timing
        next_token = samples_dict[CURRENT_TOKEN]['next']
        CURRENT_TOKEN = next_token 
            
    vis.destroy_window()
    print("Playback finished.")

if __name__ == "__main__":
    main()