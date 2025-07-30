import os
import numpy as np
import cv2
from astropy.io import fits
import levels_012.modules.utilities as utilities
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path


"""
    Function for aligning vis channel images with nir channel images 
    based on asteroid edge detection, feature extraction, and feature matching.

    Steps of the method:
    1. Edge detection 
        Detecting contours of the asteroid using second-order derivative based Laplacian method.
    2. Feature extraction
        Oriented FAST and Rotated BRIEF (ORB) function for detecting FAST keypoints with an orientation
        and BRIEF binary descriptors for keypoints.
    3. Feature matching
        Fast Library for Approximate Nearest Neighbour (FLANN) with LSH for matching keypoints from vis image to nir image keypoints 
        based on approximating their descriptor similarity with locality-sensitive hashing.
        The matches are filtered based by the distance between the first and the second match candidate.
    4. Estimating transformation matrix
        Estimating a 3x3 transformation matrix based on the feature matches using random sample consensus (RANSAC)
    5. Aligning the image
        Applying the transformation matrix to map the visible images to have the same dimensions as near-infrared images
        so that the asteroid appears same sized at the same (x,y) location on the image

"""
def merge_fits_files(files: List[str | Path], output_dir: str | Path) -> str:
    """
    Combines the FITS files into one single files containing the hyperspectral data cube.
    If the files list contains VIS channel the VIS channel is aligned into same grid with the NIR channel images.
    If the fiels contains the SWIR channel .... 

    Parameters:
        files (List[str | Path]): List of all fits files to be combined.
        output_dir (str | Path): Path to the directory where the new file is stored.

    Returns 
        (str): path to the created FITS file
    """
    if not files:
        print(f"Parameter 'files: List[str | Path]' is empty in combine_fits_files()")
        return None
    
    output_dir = Path(output_dir)
    channel_map = {
        0 : 'VIS',
        1 : 'NIR1',
        2 : 'NIR2',
        3 : 'SWIR'
    }
    channels = [] # List of channels to be combined

    primary_header_list = []
    image_header_list = []
    image_dict = {}
    swir_header = None
    swir_data = None
    cds_dict = {}
    for file in files:
        file = Path(file)

        try:
            file_name = file.name
            char = file.stem[2]
            channel_int = int(char)
            channel_name = channel_map[channel_int]
            if channel_name in channels:
                raise ValueError(f"Duplicate channel '{channel_name}' found in file '{files}'")
            channels.append(channel_name)

            with fits.open(file) as hdul:
                primary_header = hdul[0].header.copy()
                primary_header_list.append(primary_header)
                if channel_name in ('VIS', 'NIR1', 'NIR2'):
                    image_header = hdul[1].header.copy()
                    image_data = hdul[1].data.copy()
                    image_header_list.append(image_header)
                    image_dict[channel_name] = image_data
                else:
                    swir_header = hdul[1].header.copy()
                    swir_data = hdul[1].data.copy()
                if channel_name in ('NIR1', 'NIR2') and len(hdul) > 2:
                    cds_data = fits.BinTableHDU.from_columns(hdul[2].columns, header=hdul[2].header)
                    cds_dict[channel_name] = cds_data

        except (IndexError, ValueError) as e:
            print(f"Error: {e}")
        except KeyError:
            print(f"Error: unknown channel ID '{char}' in '{file.name}' ")
    
    if len(channels) == 1:
        print(f'More than one channel needed to combine.')
        return files[0]
    print(f"Combining channels {channels}")
    # Create new list of HDU's and append the primary HDU, new image HDU and other extensions.
    HDUs = []
    new_primary_header = utilities.combine_primary_headers(primary_header_list)
    new_primary_HDU = fits.PrimaryHDU(header=new_primary_header)
    HDUs.append(new_primary_HDU)

    new_image_header = utilities.combine_image_headers(image_header_list)

    new_image_data = []
    nir1_data = image_dict['NIR1']
    nir2_data = image_dict['NIR2']

    if 'VIS' in channels:
        print('Aligning VIS channel to NIR grid..')
        vis_data = image_dict['VIS']

        # Align vis channel based on first images of vis and nir1
        vis_image = vis_data[0]
        nir_image = nir1_data[0]

        transforamtion_matrix = utilities.estimate_matrix(vis_image, nir_image) # Alignment transformation matrix
        for frame in vis_data:
            # Convert to little-endian float32 for OpenCV
            little_endian = np.ascontiguousarray(frame.astype('<f4'))
            wrapped = cv2.warpPerspective(little_endian, transforamtion_matrix, (640, 512), flags=cv2.INTER_LINEAR )

            # Convert back to big_endian float32
            big_endian = np.ascontiguousarray(wrapped.astype('>f4'))
            new_image_data.append(big_endian)

    for frame in nir1_data:
            new_image_data.append(frame)
        
    for frame in nir2_data:
        new_image_data.append(frame)


    data_cube = np.stack(new_image_data, axis=0)

    # Add data to the data cube
    new_image_HDU = fits.ImageHDU(header=new_image_header, data=data_cube)
    HDUs.append(new_image_HDU)
    if 'SWIR' in channels:
        new_swir_HDU = fits.BinTableHDU.from_columns(columns=swir_data, header=swir_header)
        HDUs.append(new_swir_HDU)
    

    # Extract HDUs containing the binary tables
    if 'NIR1' in cds_dict and 'NIR2' in cds_dict:
        nir1_hdu = cds_dict['NIR1']
        nir2_hdu = cds_dict['NIR2'] 

        # Combine all columns from both tables
        all_columns = nir1_hdu.columns + nir2_hdu.columns
        combined_table = fits.BinTableHDU.from_columns(all_columns)
        
        HDUs.append(combined_table)
    hdu_list = fits.HDUList(HDUs)
    # File name for new fits
    file_path = Path(files[0])
    stem = file_path.stem
    suffix = file_path.suffix
    new_calibration_level = '2B'
    file_name = 'ASP'+ stem[3:25] + new_calibration_level + suffix
    primary_header = hdu_list[0].header
    primary_header['FILENAME'] = file_name
    primary_header['PROCLEVL'] = new_calibration_level
    # Create the new fits file with dark-subtracted images
    fits_file = os.path.join(output_dir, file_name)
    hdu_list.writeto(fits_file, overwrite=True)

    return(fits_file)

