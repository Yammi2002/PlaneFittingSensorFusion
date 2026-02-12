import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.Dataloader import get_calibration_data
import open3d as o3d
from PIL import Image

"""
Geometry Engine Module.

This module handles geometric transformations and sensor fusion operations.
It serves as the core mathematical engine to:
1. Convert sensor poses (translation/rotation) into homogeneous transformation matrices.
2. Project 3D points onto 2D image planes.
3. Aggregate point clouds from multiple sensors (LiDAR/Radar) into a unified Ego-Vehicle frame.
4. Colorize 3D point clouds by projecting them onto synchronized camera images.
"""

def compute_transformation_matrix(translation, rotation):
    """
    Constructs a 4x4 homogenous transformation matrix from translation and rotation.

    Args:
        translation (list): [x, y, z] translation vector.
        rotation (list): [w, x, y, z] quaternion rotation (NuScenes format).

    Returns:
        np.ndarray: A (4, 4) transformation matrix.
    """
    
    matrix = np.eye(4)
    matrix[:3, 3] = translation 
    
    # Scipy expects quaternions in [x, y, z, w] format, but dataset provides [w, x, y, z]
    w, x, y, z = rotation
    rotation_corretta = [x, y, z, w]
    
    rotation_matrix = R.from_quat(rotation_corretta).as_matrix()
    
    matrix[:3, :3] = rotation_matrix
    
    return matrix

def project_lidar_to_camera(pts_3d, T_transform, K, img_width, img_height):
    """
    Projects 3D points onto a 2D image plane.

    Args:
        pts_3d (np.ndarray): Array of 3D points with shape (N, 3).
        T_transform (np.ndarray): (4, 4) Transformation matrix (e.g., Ego -> Camera).
        K (np.ndarray): (3, 3) Camera Intrinsic matrix.
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.

    Returns:
        tuple: 
            - uv (np.ndarray): Pixel coordinates (u, v) with shape (N, 2).
            - depth (np.ndarray): Depth values for each point.
            - mask (np.ndarray): Boolean mask indicating points falling inside the image FOV.
    """
    # 1. Convert to Homogeneous Coordinates (N, 3) -> (N, 4)    if pts_3d.shape[1] == 3:
    pts_3d = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    
    # 2. Apply 3D Transformation (e.g., from Ego frame to Camera frame)
    # Calculation: P_cam = T * P_world
    pts_cam = (T_transform @ pts_3d.T).T  # (N, 4)
    
    # 3. Project to 2D Image Plane   
    xyz = pts_cam[:, :3].T # (3, N)
    uv_hom = K @ xyz       # (3, N)
    
    depth = xyz[2, :]
    u = uv_hom[0, :] / depth
    v = uv_hom[1, :] / depth
    
    # 4. Validity Filter (Depth > 0 and inside image boundaries)
    mask = (depth > 0) & (u >= 0) & (u < img_width - 1) & (v >= 0) & (v < img_height - 1)
    
    return np.stack([u, v], axis=1), depth, mask

def compute_point_cloud(calib_json_path, lidar_data, radar_data):
    """
    Aggregates point clouds from multiple sensors into a single Ego-Vehicle coordinate system.

    Args:
        calib_json_path (str): Path to 'calibrated_sensor.json'.
        lidar_data (list): List of lidar file dictionaries.
        radar_data (list): List of radar file dictionaries.

    Returns:
        np.ndarray: Aggregated point cloud array with shape (N_total, 3).
    """
    big_cloud_list = [] 
    dataset_root = "../data/sensordata/man-truckscenes" 

    # Combine lists to process all sensors in one loop
    all_sensors_data = lidar_data + radar_data

    for sensor_measure in all_sensors_data:
        # 1. Retrieve the calibration token
        calib_token = sensor_measure["calibrated_sensor_token"]
        
        # 2. Fetch calibration data
        try:
            sensor_info = get_calibration_data(calib_json_path, calib_token)
        except ValueError as e:
            print(e); continue

        # 3. Compute Sensor-to-Ego transformation matrix
        M_sensor = compute_transformation_matrix(sensor_info['translation'], sensor_info['rotation'])
        
        # 4. Load Point Cloud file
        full_path = f"{dataset_root}/{sensor_measure['filename']}"
        
        try:
            pcd = o3d.io.read_point_cloud(full_path)
            pts = np.asarray(pcd.points)
            if len(pts) == 0: continue

            # Convert to Homogeneous coordinates
            pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1))))
           
            # Apply Transformation: P_ego = M_sensor_to_ego * P_sensor
            pts_transformed = (M_sensor @ pts_hom.T).T
            
            # Store only XYZ coordinates (drop the homogeneous 1)
            big_cloud_list.append(pts_transformed[:, :3])
            
        except Exception as e:
            print(f"Errore file {full_path}: {e}")

    if not big_cloud_list:
        return np.array([])
        
    return np.vstack(big_cloud_list)
        
def color_point_cloud(total_points, camera_data, calib_json_path):
    """
    Colors a 3D point cloud by projecting points onto available camera images.

    Args:
        total_points (np.ndarray): Aggregated point cloud (N, 3) in Ego frame.
        camera_data (list): List of camera metadata dictionaries.
        calib_json_path (str): Path to 'calibrated_sensor.json'.

    Returns:
        np.ndarray: Array of RGB colors (N, 3) normalized between 0.0 and 1.0.
    """
    dataset_root = "../data/sensordata/man-truckscenes"
    
    point_colors = np.full((total_points.shape[0], 3), 0.5) 
    
    for cam_measure in camera_data:
        calib_token = cam_measure["calibrated_sensor_token"]
        
        try:
            cam_info = get_calibration_data(calib_json_path, calib_token)
        except ValueError: continue

        img_path = f"{dataset_root}/{cam_measure['filename']}"
        try:
            pil_img = Image.open(img_path)
            img_arr = np.asarray(pil_img)
            # Normalize to 0.0 - 1.0 range if necessary
            if img_arr.dtype == np.uint8:
                img_arr = img_arr.astype(np.float32) / 255.0
                
            H, W = img_arr.shape[:2]
        except Exception as e:
            print(f"Error loading img {img_path}: {e}")
            continue

        # Get Camera Matrices
        K = np.array(cam_info['camera_intrinsic'])
        M_cam_to_ego = compute_transformation_matrix(cam_info['translation'], cam_info['rotation'])
        
        # INVERSION: We need to transform FROM Ego frame TO Camera frame
        M_ego_to_cam = np.linalg.inv(M_cam_to_ego)
        
        # Project points
        uv, depth, mask = project_lidar_to_camera(total_points, M_ego_to_cam, K, W, H)
        
        print(f"-> Camera {cam_measure['filename'][-25:]}: matched {np.sum(mask)} points.")
        
        if np.sum(mask) > 0:
            # Round to nearest integer pixel coordinates
            u_valid = np.round(uv[mask, 0]).astype(int)
            v_valid = np.round(uv[mask, 1]).astype(int)
            
            # Clip to image boundaries (Safety check)
            u_valid = np.clip(u_valid, 0, W - 1)
            v_valid = np.clip(v_valid, 0, H - 1)
            
            # Extract colors from image array [row, col] -> [v, u]
            colors_extracted = img_arr[v_valid, u_valid, :]
            
            # Assign colors to the point cloud
            point_colors[mask] = colors_extracted

    return point_colors

def fit_plane_irls(points, threshold=0.10, n_iter=10):
    """
    Robust Plane Fitting using Iteratively Reweighted Least Squares (IRLS).
    
    This approach is deterministic and iteratively refines the plane model by down-weighting outliers.
    It is initialized assuming a roughly horizontal road surface.

    Args:
        points (np.array): (N, 3) Point cloud data.
        threshold (float): Distance threshold for inliers (e.g., 0.10m).
        n_iter (int): Number of refinement iterations.
        
    Returns:
        plane_model: [a, b, c, d] for equation ax + by + cz + d = 0
        inliers: Indices of points belonging to the plane.
    """
    N = len(points)
    if N < 3:
        return [0,0,1,0], []

    # Extract coordinates for vectorized operations
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    # 1. Initialization
    # We use a lower percentile (20th) to estimate the ground height.
    # This aligns with the "Lowest Point Representative" (LPR) heuristic commonly 
    # used in ground segmentation tasks.
    current_d = np.percentile(Z, 20)    
    # Initial parameters [a, b, d] for the explicit equation: z = ax + by + d
    params = np.array([0.0, 0.0, current_d]) 

    # Construct the Design Matrix A for the linear system: z = a*x + b*y + d*1.
    # This matrix contains the input coordinates used to solve the system.    
    # Shape: (N, 3) -> Columns are [X, Y, 1]
    A = np.column_stack((X, Y, np.ones(N)))
    
    # 2. Refinement loop  
    for i in range(n_iter):
        a, b, d_curr = params
        
        # Calculate expected Z values based on the current plane model
        expected_z = a * X + b * Y + d_curr
        
        # Calculate Residuals (Vertical distance from point to plane)
        residuals = np.abs(expected_z - Z)
        
        # Soft-start strategy
        # Problem: If the real road is sloped but we start flat (horizontal), 
        # distant points on the slope might exceed the 10cm threshold immediately.
        # Solution: In the first few iterations (i < 2), we relax the threshold (3x).
        # This allows the plane to "tilt" and catch the slope before tightening the grip.
        effective_threshold = threshold * 3.0 if i < 2 else threshold
        
        # Weight calculation
        # We assign weights based on residuals:
        # - Points close to the plane get weighted more with a value of 1.
        # - Points far from the plane get weighted less, with a value approximately 0.
        # This effectively filters out obstacles.
        weights = np.where(residuals < effective_threshold, 1.0, 0.0001)
        
        # Solve least squares
        # We solve the system: Weights * (A * x) = Weights * Z
        # To use standard solvers, we multiply A and Z by sqrt(weights).
        
        sqrt_w = np.sqrt(weights)[:, np.newaxis]
        A_weighted = A * sqrt_w
        Z_weighted = Z * sqrt_w.flatten()
        
        try:
            # Solve for new parameters [a, b, d] that minimize weighted error
            result = np.linalg.lstsq(A_weighted, Z_weighted, rcond=None)
            params = result[0] # Here we have the new parameters
        except np.linalg.LinAlgError:
            break 

    # 3. Final model conversion
    # We have parameters for: z = ax + by + d
    # Open3D expects implicit form: ax + by + cz + d = 0, so we have to rearrange values.
    a_final, b_final, d_final = params
    
    # Normalization: The normal vector (a, b, c) must have length 1.
    # This ensures that ax+by+cz+d represents the true Euclidean distance.
    normal_len = np.sqrt(a_final**2 + b_final**2 + 1.0)
    
    final_a = a_final / normal_len
    final_b = b_final / normal_len
    final_c = -1.0 / normal_len
    final_d = d_final / normal_len
    
    plane_model = [final_a, final_b, final_c, final_d]
    
    # 4. Inlier extracto
    # Re-calculate precise Euclidean distances using the final normalized model
    dist_final = np.abs(final_a * X + final_b * Y + final_c * Z + final_d)
    
    # Select indices where distance is within the strict threshold
    inliers = np.where(dist_final < threshold)[0]
    
    return plane_model, inliers