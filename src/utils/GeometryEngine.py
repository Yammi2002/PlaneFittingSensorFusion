import numpy as np
from scipy.spatial.transform import Rotation as R

def get_matrix(translation, rotation):
    matrix = np.eye(4)
    matrix[:3, 3] = translation 
    
    w, x, y, z = rotation
    rotation_corretta = [x, y, z, w]
    
    rotation_matrix = R.from_quat(rotation_corretta).as_matrix()
    
    matrix[:3, :3] = rotation_matrix
    
    return matrix

def compute_transformation_matrix(translation, rotation):
    """Wrapper per la tua get_matrix esistente."""
    return get_matrix(translation, rotation)

def project_lidar_to_camera(pts_3d, T_lidar_to_cam, K, img_width, img_height):
    """
    Proietta i punti 3D sull'immagine.
    Ritorna:
    - uv: coordinate pixel (N, 2)
    - depth: profondità (N,)
    - mask: indici dei punti validi
    """
    # 1. Coordinate Omogenee (N, 4)
    if pts_3d.shape[1] == 3:
        pts_3d = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    
    # 2. Trasformazione 3D (Lidar -> Camera)
    pts_cam = (T_lidar_to_cam @ pts_3d.T).T  # (N, 4)
    
    # 3. Proiezione 2D
    xyz = pts_cam[:, :3].T # (3, N)
    uv_hom = K @ xyz       # (3, N)
    
    depth = xyz[2, :]
    u = uv_hom[0, :] / depth
    v = uv_hom[1, :] / depth
    
    # 4. Filtro Validità
    mask = (depth > 0) & (u >= 0) & (u < img_width - 1) & (v >= 0) & (v < img_height - 1)
    
    return np.stack([u, v], axis=1), depth, mask