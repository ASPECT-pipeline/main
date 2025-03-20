import cv2
import numpy as np
import matplotlib.pyplot as plt


def combine_headers(vis, nir1, nir2, swir):
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

def append_header(hdu, dict):
    header = hdu.header
    for key, value in dict.items():
        header[key] = value
    return hdu

def normalize_to_8bit(img):
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

def laplacian(img):
    try:
        # Check if the image was loaded successfully
        if img is None:
            print("Error: Image not found or unable to open")
            return
        
        # Normalize the image to 8 bit integers
        img = normalize_to_8bit(img)

        # Apply gaussian blur
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

        # Apply Laplacian operator
        laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3)

        return laplacian
    
    except Exception as e:
        print(f"Error: {e}")

def filter_by_orientation(matches, keypoints1, keypoints2, threshold=10):
    filtered_matches = []
    for m in matches:
        angle1 = keypoints1[m.queryIdx].angle
        angle2 = keypoints2[m.trainIdx].angle
        angle_diff = abs(angle1 - angle2)
        if angle_diff < threshold or angle_diff > (360 - threshold):
            filtered_matches.append(m)
    return filtered_matches

def filter_by_distance(matches):
    ratio_thresh = 0.90  # Adjust as needed
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    distance_thresh = 65  # Adjust based on your data
    good_matches = [m for m in good_matches if m.distance < distance_thresh]
    return good_matches

def estimate_matrix(vis, nir):

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
    matches = filter_by_distance(flann_matches)

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

def asteroid_mask(image):
    edges = laplacian(image) # Detect asteroid edges

    _, binary_mask = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY) # convert to binary mask

    kernel = np.ones((5,5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel) # Apply morphological closing

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the outermost contours

    asteroid_mask = np.zeros_like(image, dtype=np.uint16)

    cv2.drawContours(asteroid_mask, contours, -1, 65535, thickness=cv2.FILLED) # Draw the asteroid mask

    return asteroid_mask
