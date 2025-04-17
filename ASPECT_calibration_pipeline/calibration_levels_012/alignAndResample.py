import os
import numpy as np
import cv2
from astropy.io import fits
import utilities

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
        The matches are filtered either based on the orientation of FAST keypoints, or by the distance between the first and the second match candidate.
    4. Estimating transformation matrix
        Estimating a 3x3 transformation matrix based on the feature matches using RANSAC
    5. Aligning the image
        Applying the transformation matrix to map the visible image to have the same dimensions os nir image
        so that the asteroid appears same sized at the same (x,y) location on the image
    

    Function: estimate_matrix (utilities.py)
        Parameters: 
            - vis: visible channel image
            - nir: near-infrared channel image
        Description:
            Performs the alignment method described above returning the estimated transformation matrix
    
    Function: align_fits_files
        Parameters:
            - fitsPath: path to the FITS file
            - outputFolder: path to the folder where the new file is stored
        Description:
            Given paths to two fitsfiles, one VIS and one NIR, performs the estimateMatrix method
            and uses the trasnformation matrix to align every 2D visible image. Creates one ImageHDU 
            containing the aligned datacube that has all vis and nir files

"""

def align_fits_files(vis_file: str, nir1_file: str, nir2_file: str, swir_file: str, output: str):

    # Open both FITS files simultaneously for image alignment
    with fits.open(vis_file) as vis_hdul, fits.open(nir1_file) as nir1_hdul, fits.open(nir2_file) as nir2_hdul, fits.open(swir_file) as swir_hdul:

        # VIS data
        vis_img_HDU = vis_hdul[1] # Image HDU
        vis_header = vis_img_HDU.header # Image HDU header

        # NIR1 data
        nir1_img_HDU = nir1_hdul[1] # Image HDU
        nir1_header = nir1_img_HDU.header # Image HDU header
        nir_width = nir1_img_HDU.header.get('NAXIS1') # NIR image height
        nir_height = nir1_img_HDU.header.get('NAXIS2') # NIR image width

        # NIR2 data
        nir2_img_HDU = nir2_hdul[1] # Image HDU
        nir2_header = nir2_img_HDU.header # Image HDU header

        # SWIR data
        swir_img_HDU = swir_hdul[1]
        swir_header = swir_img_HDU.header

        # Creating and empty primary HDU
        primaryHdu = fits.PrimaryHDU()
        primaryHdu.header['EXTEND'] = True  # This indicates that there are extensions (following image HDU)

        # Create new list of HDU's and append the primary HDU, new image HDU and other extensions
        HDUs = []
        HDUs.insert(0, primaryHdu)

        # Combine all header data into one dictionary
        header_dict = utilities.combine_headers(vis_header, nir1_header, nir2_header, swir_header)
        #New image HDU to contain the whole data cube
        new_image_HDU = fits.ImageHDU()
        #Add metadata
        new_image_HDU = utilities.append_header(new_image_HDU, header_dict)

        # Calculate the transformation matrix from the first images of vis and nir cubes
        vis_image = vis_img_HDU.data[0] # First image of vis cube
        nir_image = nir1_img_HDU.data[0]  # First image of nir1 cube

        transforamtionMatrix = utilities.estimate_matrix(vis_image, nir_image)

        #List that will contain all the 2D images
        imageDataList = []

        # Align all the vis images with the transformation matrix
        # Loop over the 2D images
        for image in vis_img_HDU.data:
            imageDataList.append(cv2.warpPerspective(image, transforamtionMatrix, (nir_width, nir_height), flags=cv2.INTER_CUBIC))

        # Add all the 2D images of the nir file
        for image in nir1_img_HDU.data:
            imageDataList.append(image)
        
        for image in nir2_img_HDU.data:
            imageDataList.append(image)
        
        # for i, img in enumerate(imageDataList):
            
            # print(f'frame {i}: {img.shape}')
        data_cube = np.stack(imageDataList, axis=0)

        # Add data to the data cube
        new_image_HDU.data = data_cube

        # Append the new image cube and swir
        HDUs.append(new_image_HDU)
        HDUs.append(swir_img_HDU)

        # vlaHdu = nirHdul[2]
        # Extract HDUs containing the binary tables
        vlaHdu1 = nir1_hdul[2]
        vlaHdu2 = nir2_hdul[2]

        # Verify both HDUs are variable-length tables
        assert vlaHdu1.header['TFORM1'].startswith('P'), "Table 1 is not a variable-length table"
        assert vlaHdu1.header['TFORM2'].startswith('P'), "Table 2 is not a variable-length table"

        # Extract columns from both tables
        columns1 = vlaHdu1.columns
        columns2 = vlaHdu2.columns
        # Create a combined list of columns
        new_columns = []
        for i, col in enumerate(columns1 + columns2):  # First vlaHdu1, then vlaHdu2
            new_col_name = f"Col_{i+1}"  # Renaming columns in increasing order
            new_columns.append(fits.Column(name=new_col_name, format=col.format, array=col.array))

        # Create a new binary table HDU with renamed columns
        new_vlaHdu = fits.BinTableHDU.from_columns(new_columns)
        HDUs.append(new_vlaHdu)

        file_name = 'D1D2v5_simulated_full_datacube.fits'

        fits_file = os.path.join(output, file_name)

        # create the new fits file with combined datacube
        hdulist = fits.HDUList(HDUs)
        hdulist.writeto(fits_file, overwrite=True)

        return(fits_file)