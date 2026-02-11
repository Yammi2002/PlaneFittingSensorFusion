import numpy as np
from PIL import Image
import open3d as o3d
from utils.GeometryEngine import compute_transformation_matrix, project_lidar_to_camera
from utils.Dataloader import get_calibration_data, get_sample_token_from_file, find_sensor_file

def main():
    # --- CONFIGURAZIONE ---
    base_path = "../data/metadata/man-truckscenes/v1.1-mini"
    data_root = "../data/sensordata/man-truckscenes"
    
    lidar_token = "3f53edcde9e44caaba5e689726c7aab7" 
    camera_token = "34b35c4e51844611bf27f578cad4fbc9"
    target_lidar_file = "LIDAR_TOP_FRONT_1692868171700983.pcd"

    # 1. RECUPERO DATI (Metadata)
    print("Recupero calibrazioni...")
    lidar_calib = get_calibration_data(f"{base_path}/calibrated_sensor.json", lidar_token)
    cam_calib = get_calibration_data(f"{base_path}/calibrated_sensor.json", camera_token)
    
    # Trova il sample token e l'immagine corrispondente
    sample_token = get_sample_token_from_file(f"{base_path}/sample_data.json", target_lidar_file)
    if not sample_token:
        print("Sample token non trovato!"); exit()
        
    cam_filename = find_sensor_file(f"{base_path}/sample_data.json", sample_token, "CAMERA_RIGHT_FRONT")
    if not cam_filename:
        print("Immagine non trovata!"); exit()

    # 2. CARICAMENTO ASSET (IO)
    print(f"Carico {cam_filename}...")
    img = np.asarray(Image.open(f"{data_root}/{cam_filename}"))
    if img.dtype == np.uint8: img = img.astype(np.float32) / 255.0
    H, W = img.shape[:2]

    pcd = o3d.io.read_point_cloud(f"{data_root}/samples/LIDAR_TOP_FRONT/{target_lidar_file}")
    pts = np.asarray(pcd.points)

    # 3. GEOMETRIA (Math)
    # Calcolo matrici
    M_lidar = compute_transformation_matrix(lidar_calib['translation'], lidar_calib['rotation'])
    M_cam = compute_transformation_matrix(cam_calib['translation'], cam_calib['rotation'])
    M_lidar_to_cam = np.linalg.inv(M_cam) @ M_lidar
    K = np.array(cam_calib['camera_intrinsic'])

    # Proiezione pura
    uv, depth, mask = project_lidar_to_camera(pts, M_lidar_to_cam, K, W, H)

    # 4. COLORING & VISUALIZATION
    print(f"Punti proiettati validi: {np.sum(mask)}")
    
    # Estrai colori
    uv_valid = np.round(uv[mask]).astype(int)
    colors = img[uv_valid[:, 1], uv_valid[:, 0]] # Nota: v, u
    
    # Crea nuvola finale
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(pts[mask])
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    # Segmentazione piano (Road Fitting)
    plane, inliers = pcd_colored.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=1000)
    print(f"Equazione strada: {plane}")
    
    o3d.visualization.draw_geometries([pcd_colored])

if __name__ == "__main__":
    main()