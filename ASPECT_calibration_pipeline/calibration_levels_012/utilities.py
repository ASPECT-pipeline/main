import cv2
import numpy as np
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
from typing import Literal, Iterable, Callable
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import LinearNDInterpolator, interp1d
import inspect
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
from scipy.stats import norm
import warnings
import json
from astropy.io.fits import Header, PrimaryHDU, ImageHDU, BinTableHDU

# numerical eps
_num_eps = 1e-5

# Preprocess the data files

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

#Extract the cahnnel from calib.json file
def read_channel(calibPath: str) -> str:
    with open(calibPath, 'r') as file:
        data = json.load(file)
        if data == None: # As the example SWIR files do not have config data
            return 'SWIR'
        firstKey = list(data.keys())[0]  # Access the first top-level key
        secondKey = list(data[firstKey].keys())[0]  # Access the first sub-key

        # Access the key indicating channel
        channel = list(data[firstKey][secondKey].keys())[0]

    return(channel)

#read the meta data from config file
def read_config(configPath: str, channel:str) -> Tuple[str, List[str], List[str], List[str], List[str]]:
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

    return(order, exposureTimes, sp1Values, sp2Values, sp3Values)

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