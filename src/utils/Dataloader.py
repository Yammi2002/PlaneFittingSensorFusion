import json

"""
Data Retrieval & Management Module.

This module provides utility functions to interact with the dataset metadata (JSON files).
It acts as an interface to retrieve:
- Sensor calibration parameters (Extrinsic matrices R/T and Intrinsic K).
- file paths for sensor data (LiDAR, Camera, Radar).
- Synchronization data based on sample tokens.

Usage:
    Use `get_calibration_data` to fetch specific sensor info.
    Use `get_sensors_for_sample` to aggregate data for a specific timestamp.
"""

def get_calibration_data(json_path, sensor_token):
    """
    Extracts rotation quaternion, translation vector, and camera intrinsic matrix 
    from the 'calibrated_sensor.json' file.
    
    Args:
        json_path (str): Path to the calibrated_sensor.json file.
        sensor_token (str): The specific 'calibrated_sensor_token' (UUID) to search for.
        
    Returns:
        dict: A dictionary containing 'translation', 'rotation', and 'camera_intrinsic'.
        
    Raises:
        ValueError: If the token is not found.
    """
    with open(json_path) as f:
        calib_data = json.load(f)
    
    for sensor in calib_data:
        if sensor['token'] == sensor_token:
            return sensor
    raise ValueError(f"Sensor token {sensor_token} not found!")

def get_sensors_for_sample(data, target_sample_token):
    lidars_data = []
    camera_data = []
    radar_data = []
    
    for measure in data:
        if measure["sample_token"] == target_sample_token:         
            if "LIDAR" in measure["filename"] and measure["is_key_frame"] == True:
                lidars_data.append(measure)
            elif "CAMERA" in measure["filename"]and measure["is_key_frame"] == True:
                camera_data.append(measure)
            elif "RADAR" in measure["filename"]and measure["is_key_frame"] == True:
                radar_data.append(measure)
                
    return lidars_data, camera_data, radar_data
