import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json
from astropy.io.fits import Header, PrimaryHDU, ImageHDU, BinTableHDU
from datetime import datetime
from pathlib import Path
import os
import re
import hera_spice


def is_valid_fits_file(path:str) -> Tuple[bool, Optional[str]]:
    path = Path(path)
    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Path is not a file: {path}"
    if path.suffix.lower() != '.fits':
        return False, f"File does not have a .fits extension: {path}"
    
    return True, None

def is_valid_json_file(path:str) -> Tuple[bool, Optional[str]]:
    path = Path(path)
    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Path is not a file: {path}"
    if path.suffix.lower() != '.json':
        return False, f"File does not have a .json extension: {path}"
    
    return True, None

def is_valid_meta_folder(path:str) -> Tuple[bool, Optional[str]]:
    required_files = {"calib.json", "config.json", "telemetry.json"}
    folder = Path(path)

    if not folder.exists():
        return False, f"Folder not found: {folder}"
    if not folder.is_dir(): 
        return False, f"Path is not a directory: {folder}"
    
    missing = [f for f in required_files if not (folder / f).is_file()]
    if missing:
        return False, f"Missing required files: {', '.join(missing)}"

    return True, None

def get_acq_folder(acq_path: str) -> Optional[str]:
    for name in os.listdir(acq_path):
        if os.path.isdir(os.path.join(acq_path, name)) and re.fullmatch(r'acq_\d{3}', name):
            return os.path.join(acq_path, name)
    return None  # If no matching folder is found

def get_acqSeq(acq_folder: str) -> Dict[str, str]:
    acqSeq = os.path.join(acq_folder, 'meta_acq/acqSeq.json')
    boolean, error_message = is_valid_json_file(acqSeq)
    if not boolean:
        print(error_message)
        return None

    with open(acqSeq, 'r') as file:
        data = json.load(file)
    return data
    
def get_static_metadata() -> Dict[str, Tuple[str, str]]:
    static_metadata = {
        'INSTRUME' : ('ASPECT', 'Camera ID'),
        'ORIGIN'   : ('ESA-HERA', ''),
        'DATE'     : ('', 'UTC time of file creation'),
        'FILENAME' : ('', 'Name of the actual fits file'),
        'SWCREATE' : ('', 'Software identification'),
        'ORIGFILE' : ('', 'Original file name'),
        'PROCLEVL' : ('0', 'Calibration level'),
        'MISSPHAS' : ('', 'HERA Mission Phase ID'),
        'OBSERVPH' : ('', 'HERA Observation ID'),
        'OBSTARGT' : ('', 'Observation target'),
        # Instrument data
        'DATE-OB'   : ('', 'UTC time of observation'),
        'OBJECT'    : ('', 'Observed Object'),
        'EXPOSURE'  : ('', 'Exposure time [s]'), 
        'CCDTEMP'   : ('', 'Detector temperature'),
        'AMBTEMP'   : ('', 'Ambient temperature'),
        'SC_CLK'    : ('', 'SC clock Hera instrument format'),
        'ERRORFLG'  : ('', 'Error flags for instrument'),
        # Instrument specific data
        'HIERARCH HSH_FILENAME'     : ('', 'Internal filename'),
        'HIERARCH WINDOWED_IMAGE'   : ('', 'Full frame / windowed image'),
        'OFFSET_X'                  : ('', 'Image offset'),
        'OFFSET_Y'                  : ('', 'Image offset'),
        'HIERARCH SENSOR_ADC_OFFSET': ('', ''),
        'HIERARCH SENSOR_PIXEL_CLK_FREQ' : ('', ''),
        'HIERACRH SENSOR_TEMPERATURE_DEGC' : ('', ''),
        'HIERACRH TEMP_FPA'         : ('', 'FPA temperature [K]'),
        'HIERACRH TEMP_BEE'         : ('', 'BEE tenperature [K]'),
        'HIERARCH TEMP_HSP'         : ('', 'HSP temperature [K]'),
        'HIERARCH TEMP_ICU1'        : ('', 'ICU1 temperature [K]'),
        'HIErARCH TEMP_TELE_1'      : ('', 'Telescope 1 temperature [K]'),
        # SPICE data
        'SPICE_MK'      : ('', 'SPICE meta kernel version'),
        'SPICECLK'      : ('', 'SC clock SPICE format'),
        'SUN_POSX'      : ('', 'Sun position vector X [km]'),
        'SUN_POSY'      : ('', 'Sun position vector Y [km]'),
        'SUN_POSZ'      : ('', 'Sun position vector Z [km]'),
        'SOLAR_D'       : ('', 'Solar distance [AU]'),
        'EARTPOSX'      : ('', 'Earth position vector X [km]'),
        'EARTPOSY'      : ('', 'Earth position vector Y [km]'),
        'EARTPOSZ'      : ('', 'Earth position vector Z [km]'),
        'EARTH_D'       : ('', 'Earth distance [AU]'),
        'TARGET'        : ('', 'Observation target (SPICE)'),
        'TRG_POSX'      : ('', 'Target position vector X [km]'),
        'TRG_POSY'      : ('', 'Target position vector Y [km]'),
        'TRG_POSZ'      : ('', 'Target position vector Z [km]'),
        'TRG_DIST'      : ('', 'Target distance [AU]'),
        'SC_QUAT0'      : ('', 'Spacecraft quaterion 0'),
        'SC_QUAT1'      : ('', 'Spacecraft quaterion 1'),
        'SC_QUAT2'      : ('', 'Spacecraft quaterion 2'),
        'SC_QUAT3'      : ('', 'Spacecraft quaterion 3'),
        'CAM_RA'        : ('', 'Camera axis RA [deg]'),
        'CAM_DEG'       : ('', 'Camera axis DEC [deg]'),
        'CAM_NAZ'       : ('', 'Camera axis north azimuth [deg]'),
        'SOL_ELNG'      : ('', 'Solar elongation [deg]'),
        # Calibration specific data
        'CALPHASE'      : ('', 'Calibration phase'),
        'SPHCUR1'       : ('', 'Integrating sphere current 1'),
        'SPHCUR2'       : ('', 'Integrating sphere current 2'),
        'BBLCUR'        : ('', 'Broad-band source current'),
        'BBLDIST'       : ('', 'Broad-band source distance'),
        'MONDIST'       : ('', 'Monochromator band'),
        'MONOWL'        : ('', 'Monochromator wavelength band'),
        'MONOBAND'      : ('', 'Monochromator band'),
        'MONOFLT'       : ('', 'Monochromator filter'),
        'MONOGRAT'      : ('', 'Monochromator grating'),
        'MONOSPH'       : ('', 'Monochromator sphere version'),
        'MONOPHC'       : ('', 'Monochormator photocurrent'),
        'DISTTRGT'      : ('', 'Distortion target description')
    }
    return static_metadata

# Based on SP1 and channel determine the order.
# The sp value should be taken from the index 3
# to prevent miss identification.
def check_order(sp: float, channel: str) -> str:
    match channel:
        case 'VIS' | 'NIR1':
            if sp > 19000:
                return 'h'
            else:
                return 'l'
        case 'NIR2':
            if sp > 20000:
                return 'h'
            else:
                return 'l'
        case 'SWIR':
            return ''

# Read metadata from config file
def read_config(configPath: str, channel:str) -> Dict[str, Any]:
    boolean, error_message = is_valid_json_file(configPath)
    if not boolean:
        print(error_message)
        return None
    
    meta_data = {}

    with open(configPath, 'r') as file:
        data = json.load(file)

        #read SP values for each image
        match channel:
            case 'VIS':
                taskFile = data['visTaskFile']
            case 'NIR1':
                taskFile = data['nir1TaskFile']
            case 'NIR2':
                taskFile = data['nir2TaskFile']
            case 'SWIR':
                taskFile = data['swirTaskFile']
        

        #Extract sp values from taskValues
        taskValues = [taskFile[i:i + 8] for i in range(0, len(taskFile), 8)]
        sp1Values = [taskValues[i][1] for i in range(0, len(taskValues))]
        sp2Values = [taskValues[i][2] for i in range(0, len(taskValues))]
        sp3Values = [taskValues[i][3] for i in range(0, len(taskValues))]
        #Extract exposure times
        exposureTimes = [taskValues[i][4] for i in range(0, len(taskValues))]

        #Check the order based on SP1 index 3
        order = check_order(sp1Values[3], channel)

        meta_data['ORDER'] = order
        meta_data['EXPOSURE'] = exposureTimes
        meta_data['SP1'] = sp1Values
        meta_data['SP2'] = sp2Values
        meta_data['SP3'] = sp3Values

    return meta_data

# Read metadata from telementry
def read_telemetry(telemetry_path: str, channel: str) -> Dict[str, Any]:

    boolean, error_message = is_valid_json_file(telemetry_path)
    if not boolean:
        print(error_message)
        return None
    
    meta_data = {} 
    with open(telemetry_path, 'r') as file:
        data = json.load(file)


        Acq_date = data['ACQ_DATE']
        dt = datetime.strptime(Acq_date, "%a %b %d %H:%M:%S %Y")
        meta_data['DATE-OB'] = dt.strftime("%Y-%m-%dT%H:%M:%S.000")

    return meta_data





def get_channel_frames_names(acq_folder:str) -> List[Tuple[str, int]]:

    channel_map = {
        0: 'VIS',
        1: 'NIR1',
        2: 'NIR2',
        3: 'SWIR'
    }

    channel_info: Dict[str, Tuple[int, str]] = {}
    pattern = re.compile(r'^dc_(\d)_')
    frame_pattern = re.compile(r'(exp_)\d{3}')

    # Add channel names, frame counts and an original filename to a dictionary
    for filename in os.listdir(acq_folder):
        match  = pattern.match(filename)
        if match:
            index = int(match.group(1))
            if index in channel_map:
                channel_name = channel_map[index]
                if channel_name in channel_info:
                    count, orig_name = channel_info[channel_name]
                    channel_info[channel_name] = (count + 1, orig_name)
                else: 
                    channel_info[channel_name] = (1, filename)

    # Replace frame number with 'XXX' if more than one frame
    for channel in channel_info:
        count, orig_name = channel_info[channel]
        if count > 1:
            modified = frame_pattern.sub(r'\1XXX', orig_name)
            channel_info[channel] = (count, modified)

    return channel_info
 
def collect_channel_acq_info(acq_path:str) -> Dict[str, Any]:
    acq_folder = get_acq_folder(acq_path)

    if acq_folder == None:
        print(f'no acq_XXX folder found inside: {acq_path}')
    
    meta_data = {}

    meta_data['channel_info'] = get_channel_frames_names(acq_folder)
    meta_data.update(get_acqSeq(acq_folder)) 

    return meta_data

def collect_metadata(meta_folder:str , channel:str ) -> Dict[str, Any]:
    boolean, error_message = is_valid_meta_folder(meta_folder)
    if not boolean:
        print(error_message)
        return None
    
    meta_data = {}

    config_path = os.path.join(meta_folder, 'config.json')
    telemetry_path = os.path.join(meta_folder, 'telemetry.json')
    calib_path = os.path.join(meta_folder, 'calib.json')

    config = read_config(config_path, channel)
    meta_data.update(get_acqSeq(config)) 


def collect_spice_metadata(telemetry:str, mk: str, channel:str)-> Dict[str, str]:
    """
    Collect specified spice kernel data for fits primary header.

    Parameters:
        telemetry (str): Path to the telemetry JSON file of the acquisition
        mk (str): Defines which meta kernel is loaded. Options: ops, plan
        channel (str): Identifies to which channel the spice data is retrived

    Returns: 
        A dicitionary of header keywords and values
    """
    spice_metadata = {}

    tele = read_telemetry(telemetry, channel)
    utc_ob = tele['DATE-OB']
    et = hera_spice.utc_2_et(utc_ob)
    milani_frame = 'MILANI_SPACECRAFT'
    
    hera_spice.load_meta_kernel(mk) # Load the meta kernel

    mk_id = hera_spice.query_mk_identifier() # Meta kernel version
    spice_metadata['SPICE_MK'] = mk_id

    sclk = hera_spice.get_sclk(et, milani_frame) # SC clock in spice format
    spice_metadata['SPICECLK'] = sclk

    # Sun position vector and distnace from observer
    sun_position, sun_distance_au = hera_spice.query_position_distance(target='SUN', et=et, frame='J2000', abcorr='NONE', observer=milani_frame)
    spice_metadata['SUN_POSX'] = sun_position[0]
    spice_metadata['SUN_POSY'] = sun_position[1]
    spice_metadata['SUN_POSZ'] = sun_position[2]
    spice_metadata['SOLAR_D']  = sun_distance_au

    # Sun position vector and distnace from observer
    earth_position, earth_distance_au = hera_spice.query_position_distance(target='EARTH', et=et, frame='J2000', abcorr='NONE', observer=milani_frame)
    spice_metadata['EARTPOSX'] = earth_position[0]
    spice_metadata['EARTPOSY'] = earth_position[1]
    spice_metadata['EARTPOSZ'] = earth_position[2]
    spice_metadata['EARTH_D']  = earth_distance_au





    hera_spice.unload_all_kernels() # Unload all kernels at the end




def combine_headers(vis: Header, nir1: Header, nir2: Header, swir: Header) -> Dict[str, Any]:
    header_dict = {}
    #VIS
    header_dict['V_ORDER'] = vis.get('ORDER')
    header_dict['V_WL'] = vis.get('WAVELEN')
    header_dict['V_EXPOS'] = vis.get('EXPOS')
    header_dict['V_SP1'] = vis.get('PIEZO1')
    header_dict['V_SP2'] = vis.get('PIEZO2')
    header_dict['V_SP3'] = vis.get('PIEZO3')
    header_dict['V_NUM'] = vis.get('NAXIS3')
    #NIR1
    header_dict['N1_ORDER'] = nir1.get('ORDER')
    header_dict['N1_WL'] = nir1.get('WAVELEN')
    header_dict['N1_EXPOS'] = nir1.get('EXPOS')
    header_dict['N1_SP1'] = nir1.get('PIEZO1')
    header_dict['N1_SP2'] = nir1.get('PIEZO2')
    header_dict['N1_SP3'] = nir1.get('PIEZO3')
    header_dict['N1_NUM'] = nir1.get('NAXIS3')
    #NIR2
    header_dict['N2_ORDER'] = nir2.get('ORDER')
    header_dict['N2_WL'] = nir2.get('WAVELEN')
    header_dict['N2_EXPOS'] = nir2.get('EXPOS')
    header_dict['N2_SP1'] = nir2.get('PIEZO1')
    header_dict['N2_SP2'] = nir2.get('PIEZO2')
    header_dict['N2_SP3'] = nir2.get('PIEZO3')
    header_dict['N2_NUM'] = nir2.get('NAXIS3')
    #SWIR
    header_dict['S_ORDER'] = swir.get('ORDER')
    header_dict['S_WL'] = swir.get('WAVELEN')
    header_dict['S_EXPOS'] = swir.get('EXPOS')
    header_dict['S_SP1'] = swir.get('PIEZO1')
    header_dict['S_SP2'] = swir.get('PIEZO2')
    header_dict['S_SP3'] = swir.get('PIEZO3')
    header_dict['S_NUM'] = swir.get('NAXIS2')

    return header_dict

def append_header(hdu: ImageHDU, dict: Dict[str, Any]) -> ImageHDU:
    header = hdu.header
    for key, value in dict.items():
        header[key] = value
    return hdu

def normalize_to_8bit(img: np.ndarray) -> np.ndarray:
    # Compute min and max values in the image
    min_val = np.min(img)
    max_val = np.max(img)

    # Avoid division by zero in case all values are the same
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.uint8)

    # Normalize image to range 0-255
    normalized = (img - min_val) / (max_val - min_val) * 255.0

    # Convert to 8-bit integer
    return normalized.astype(np.uint8)

def extract_diagnostics(image: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    # Define diagnostic pixel regions
    top = 5  # Five lines at the top
    bottom = 1  # One line at the bottom
    left = 4  # Four columns on the left
    right = 4  # Four columns on the right
    # To store the extracted pixels
    diagnosticPixels = []

    # Step 1: Extract the first 5 rows
    for row in image[:top]:
        diagnosticPixels.append(row.tolist())
    
    # Step 2: For the remaining rows (except the last one), extract the first 4 and last 4 values
    for row in image[top:-bottom]:
        left_values = row[:left]
        right_values = row[-right:]
        combined_row = np.concatenate((left_values, right_values)).tolist()
        diagnosticPixels.append(combined_row)
    
    # Step 3: Extract the last row as a separate list
    diagnosticPixels.append(image[-1].tolist())

    # Remove diagnostic pixels to create the cleaned image
    cleanedImage = image[
        top:-bottom,  # Remove top and bottom rows
        left:-right  # Remove left and right columns
    ]


    return (cleanedImage, diagnosticPixels)

def laplacian(img: np.ndarray) -> np.ndarray:

    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not found or unable to open")
    
    # Normalize the image to 8 bit integers
    img = normalize_to_8bit(img)

    # Apply gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3)

    return laplacian

def filter_by_orientation(matches, keypoints1, keypoints2, threshold=10) -> List[cv2.DMatch]:
    filtered_matches = []
    for m in matches:
        angle1 = keypoints1[m.queryIdx].angle
        angle2 = keypoints2[m.trainIdx].angle
        angle_diff = abs(angle1 - angle2)
        if angle_diff < threshold or angle_diff > (360 - threshold):
            filtered_matches.append(m)
    return filtered_matches

def filter_by_distance(matches: List[List[cv2.DMatch]]) -> List[cv2.DMatch]:
    ratio_thresh = 0.90  # Adjustable
    good_matches = []
    for m in matches:
        if len(m) == 2:
            match1, match2 = m
            if match1.distance < ratio_thresh * match2.distance:
                good_matches.append(match1)
        elif len(m) == 1:
            match1 = m[0]
            good_matches.append(match1)
    distance_thresh = 65  # Adjustable
    good_matches = [m for m in good_matches if m.distance < distance_thresh]
    return good_matches

def estimate_matrix(vis: np.ndarray, nir: np.ndarray, filter: bool = True) -> np.ndarray:
    """
    if filter = true, the feature matches are filtered byt the distance between the 1st and 2nd match suggestion. (This method is recommended)
    if filter = false, the matches are filtered based on the angle of the keypoint calculated by ORB
    """

    # Step 1: Edge detection
    edges1 = laplacian(vis)
    edges2 = laplacian(nir)


    # Step 2: Feature detection using ORB
    orb = cv2.ORB_create(nfeatures=2000) # create ORB feature detector
    # keypoints and binary descriptions
    keypoints1, descriptors1 = orb.detectAndCompute(edges1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(edges2, None)

    # Step 3: Match features
    # FLANN
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                    table_number=30,  # Number of hash tables
                    key_size=20,     # Size of the key
                    multi_probe_level=2)  # Number of probes
        
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params) # Initialize the FLANN

    flann_matches = flann.knnMatch(descriptors1, descriptors2, k=2) # Match features

    #Filter the matches based on the distance. Other option is filter_by_orientation
    if filter: 
        matches = filter_by_distance(flann_matches)
    else:
        matches = filter_by_orientation(flann_matches)


    # Step 4: Extract location of good matches and estimate transformation matrix
    # arrays to store x and y coordinates
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    #Extract keypoint coordinates 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Estimate transformation matrix
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 10.0)

    
    #Return the transformation matrix
    return(H)

def cropND(img: np.ndarray, bounding: tuple[int, int]) -> np.ndarray:
    start = tuple(map(lambda a, da: (a - da) // 2, np.shape(img), bounding))
    end = tuple(map(np.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]