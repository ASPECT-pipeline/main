import os
import numpy as np
import cv2
from astropy.io import fits

"""
    Function for aligning and resampling vis channel images with nir channel images 
    based on asteroid edge detection, feature extraction, and feature matching.

    Steps of the method:
    1. Edge detection 
        Detecting contours of the asteroid using second-order derivative based Laplacian method
    2. Feature extraction
        Oriented FAST and Rotated BRIEF (ORB) function for detecting FAST keypoints with an orientation
        and BRIEF binary descriptors for keypoints
    3. Feature matching
        FLANN with LSH for matching keypoints from vis image to nir image keypoints 
        based on approximating their descriptor similarity with hash functions.
        The matches are filtered based on the orientation of FAST keypoints to filter out
        mismatches.
    4. Estimating transformation matrix
        Estimating a 3x3 transformation matrix based on the feature matches using RANSAC
    5. Aligning the image
        Applying the transformation matrix to map the visible image to have the same dimensions os nir image
        so that the asteroid appears same sized at the same (x,y) location on the image
    
    Function: filterByOrientation 
        Parameters:
            - matches: keypoint matches made with FLANN of BF-matcher
            - keypoints1: keypoints detected from query image 
            - keypoints2: keypoints detected from train image 
            - threshold: highest allowed angle difference between keypoint orientation. Default is 10.
        Description:
            Iterates over all matches and calculates what is the difference in orientation between the keypoints. 
            If the difference is less than 10 degrees the match is added to filtered_matches, that is returned.

    Function: estimateMatrix
        Parameters: 
            - vis: visible channel image
            - nir: near-infrared channel image
        Description:
            Performs the alignment method described above returning the estimated transformation matrix
    
    Function: alignFitsFiles
        Parameters:
            - fitsPath: path to the FITS file
            - outputFolder: path to the folder where the new file is stored
        Description:
            Given paths to two fitsfiles, one VIS and one NIR, performs the estimateMatrix method
            and uses the trasnformation matrix to align every 2D visible image. Creates one ImageHDU 
            containing the aligned datacube that has all vis and nir files

"""

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

def filterByOrientation(matches, keypoints1, keypoints2, threshold=10):
    filtered_matches = []
    for m in matches:
        angle1 = keypoints1[m.queryIdx].angle
        angle2 = keypoints2[m.trainIdx].angle
        angle_diff = abs(angle1 - angle2)
        if angle_diff < threshold or angle_diff > (360 - threshold):
            filtered_matches.append(m)
    return filtered_matches

def estimateMatrix(vis, nir):

    # Step 1: Edge detection

    # Edges for Bilateral filtered image
    edges1 = laplacian(vis)
    edges2 = laplacian(nir)


    # Step 2: Feature detection using ORB

    # create ORB feature detector
    orb = cv2.ORB_create(nfeatures=2000)
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

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    flannMatches = flann.knnMatch(descriptors1, descriptors2, k=2)
    matches = [m for match in flannMatches for m in match]

    # Filter the mismatched based on the orientation claculated by ORB
    matches = filterByOrientation(matches, keypoints1, keypoints2, threshold=10)

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
    
    #Return the transformation matrix
    return(H)

def alignFitsFiles(vis_file, nir1_file, nir2_file, output):

    # Open both FITS files simultaneously for image alignment
    with fits.open(vis_file) as vis_hdul, fits.open(nir1_file) as nir1_hdul, fits.open(nir2_file) as nir2_hdul:

        #VIS data
        vis_primary_HDU = vis_hdul[0] # Primary HDU
        vis_image_HDU = vis_hdul[1] # Image HDU

        #NIR1 data
        nir1_primary_HDU = nir1_hdul[0] # Primary HDU
        nir1_image_HDU = nir1_hdul[1] # Image HDU
        nir_height = nir1_image_HDU.header.get('NAXIS1') # NIR image height
        nir_width = nir1_image_HDU.header.get('NAXIS2') # NIR image width

        #NIR2 data
        nir2_primary_HDU = nir2_hdul[0] # Primary HDU
        nir2_image_HDU = nir2_hdul[1] # Image HDU

        # Calculate the transformation matrix from the first images of vis and nir cubes

        vis_image = vis_image_HDU.data[0] # First image of vis cube
        nir_image = nir1_image_HDU.data[0]  # First image of nir1 cube

        transforamtionMatrix = estimateMatrix(vis_image, nir_image)

        #Creating and empty primary HDU
        primaryHdu = fits.PrimaryHDU()
        primaryHdu.header['EXTEND'] = True  # This indicates that there are extensions (following image HDU)

        #List that will contain all the 2D images
        imageDataList = []

        # Align all the vis images with the transformation matrix
        # Loop over the 2D images
        for image in enumerate(vis_image_HDU.data):
            imageDataList.append(cv2.warpPerspective(image, transforamtionMatrix, (nir_width, nir_height), flags=cv2.INTER_CUBIC))
        # Add all the 2D images of the nir file
        for image in enumerate(nir1_image_HDU.data):
            imageDataList.append(image)
        
        for image in enumerate(nir2_image_HDU.data):
            imageDataList.append(image)
        
        data_cube = np.stack(imageDataList, axis=0)
        #Create an image HDU
        new_image_HDU = fits.ImageHDU(data_cube)

        # vlaHdu = nirHdul[2]
        # Extract HDUs containing the binary tables
        vlaHdu1 = nir1_hdul[2]
        vlaHdu2 = nir2_hdul[2]

        # Verify both HDUs are variable-length tables
        assert 'VLA' in vlaHdu1.header['TFORM1'], "Table 1 is not a variable-length table"
        assert 'VLA' in vlaHdu2.header['TFORM1'], "Table 2 is not a variable-length table"

        # Combine the columns from both tables
        new_columns = fits.ColDefs(vlaHdu1.columns + vlaHdu2.columns)

        # Create a new binary table HDU
        new_vlaHdu = fits.BinTableHDU.from_columns(new_columns)

        new_hdul = fits.HDUList([primaryHdu, new_image_HDU, new_vlaHdu])

        file_name = 'complete_cube.fits'
        fits_file = os.path.join(output, file_name)

        hdulist = fits.HDUList(new_hdul)
        hdulist.writeto(fits_file, overwrite=True)

        return(fits_file)