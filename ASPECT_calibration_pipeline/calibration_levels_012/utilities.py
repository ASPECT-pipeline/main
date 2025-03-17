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

def laplacian(img):
    try:

        # Check if the image was loaded successfully
        if img is None:
            print("Error: Image not found or unable to open")
            return

        # Apply Laplacian operator
        laplacian = cv2.Laplacian(img, cv2.CV_64F)

        # Convert to uint8
        laplacianAbs = cv2.convertScaleAbs(laplacian)

        return (laplacianAbs)
    
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

def estimate_matrix(vis, nir):

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(vis, cmap='gray')
    plt.title(f'vis')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(nir, cmap='gray')
    plt.title(f'nir')
    plt.axis('off')

    plt.show()


    # Step 1: Edge detection
    # Edges for Bilateral filtered image
    edges1 = laplacian(vis)
    edges2 = laplacian(nir)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(edges1, cmap='gray')
    plt.title(f'vis edges')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges2, cmap='gray')
    plt.title(f'nir edges')
    plt.axis('off')

    plt.show()

    # Step 2: Feature detection using ORB
    # create ORB feature detector
    orb = cv2.ORB_create(nfeatures=2000)
    # keypoints and binary descriptions
    keypoints1, descriptors1 = orb.detectAndCompute(edges1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(edges2, None)

    # Visualize ORB keypoints
    img1_keypoints = cv2.drawKeypoints(edges1, keypoints1, None, color=(0, 255, 0))
    img2_keypoints = cv2.drawKeypoints(edges2, keypoints2, None, color=(0, 255, 0))


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_keypoints)
    plt.title('VIS Keypoints')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2_keypoints)
    plt.title('NIR Keypoints')
    plt.axis('off')

    plt.suptitle('Oriented FAST and Rotated Brief (ORB) feature detecting')
    plt.show()


    # Step 3: Match features
    # FLANN
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                    table_number=30,  # Number of hash tables
                    key_size=20,     # Size of the key
                    multi_probe_level=2)  # Number of probes
        
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    flann_matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    matches = [m for match in flann_matches for m in match]

    img_matches = cv2.drawMatches(edges1, keypoints1, edges2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    print(f'Matches: {len(matches)}')
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title('Feature Matches ORB')
    plt.axis('off')
    plt.show()

    # Filter the mismatched based on the orientation claculated by ORB
    matches = filter_by_orientation(matches, keypoints1, keypoints2, threshold=10)

    img_matches = cv2.drawMatches(edges1, keypoints1, edges2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title('Feature Matches')
    plt.axis('off')
    plt.show()

    # Step 4: Extract location of good matches and estimate transformation matrix
    # arrays to store x and y coordinates
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    #Extract keypoint coordinates 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Estimate transformation matrix
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width = nir.shape[:2]
    aligned_image = cv2.warpPerspective(vis, H, (width, height), flags=cv2.INTER_CUBIC)

    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    plt.imshow(vis, cmap='gray' )
    plt.title('Original VIS image')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(aligned_image, cmap='gray')
    plt.title('Aligned VIS image')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(nir, cmap='gray')
    plt.title('Original NIR image')
    plt.axis('off')
    plt.show()



    
    #Return the transformation matrix
    return(H)