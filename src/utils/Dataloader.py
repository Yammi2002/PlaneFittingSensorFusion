import json

def get_calibration_data(json_path, sensor_token):
    """Estrae matrici di rotazione, traslazione e intrinsic dal JSON."""
    with open(json_path) as f:
        calib_data = json.load(f)
    
    for sensor in calib_data:
        if sensor['sensor_token'] == sensor_token:
            return sensor
    raise ValueError(f"Sensor token {sensor_token} not found!")

def find_sensor_file(json_path, sample_token, sensor_keyword):
    """
    Cerca il file (jpg o pcd) associato a un sample_token e un tipo di sensore.
    Esempio sensor_keyword: 'CAMERA_RIGHT_FRONT'
    """
    with open(json_path) as f:
        data = json.load(f)
        
    for entry in data:
        if entry['sample_token'] == sample_token and sensor_keyword in entry['filename']:
            # Filtro extra per sicurezza
            if entry['filename'].endswith(('.jpg', '.png', '.pcd')):
                return entry['filename']
    return None

def get_sample_token_from_file(json_path, filename):
    """Trova il sample_token dato il nome di un file."""
    with open(json_path) as f:
        data = json.load(f)
    for entry in data:
        if entry['filename'].endswith(filename):
            return entry['sample_token']
    return None