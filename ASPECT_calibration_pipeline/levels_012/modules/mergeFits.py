import os
import numpy as np
import cv2
from astropy.io import fits
from astropy.table import Table, hstack
import levels_012.modules.utilities as utilities
from pathlib import Path
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
    If the files list contains Vis channel the Vis channel is aligned into same grid with the NIR channel images.
    If the fiels contains the SWIR channel .... 

    Parameters:
        files (List[str | Path]): List of all fits files to be combined.
        output_dir (str | Path): Path to the directory where the new file is stored.

    Returns 
        (str): path to the created FITS file
    """
    if not files:
        print(f"[WARNING] Files list empty in merge_fits_files. \nThe input directory should contain atleast two files ending with '_1B.fits'. To create them execute pipeline level 1 before level 2 or rename files.")
        return None
    
    output_dir = Path(output_dir)
    channel_map = {
        0 : 'Vis',
        1 : 'NIR1',
        2 : 'NIR2',
        3 : 'SWIR'
    }
    channels = [] # List of channels to be combined

    primary_header_list = []
    image_dict = {}
    cds_dict = {}
    swir_data = None
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
                primary_hdu = hdul[0]
                primary_header = primary_hdu.header
                primary_data = primary_hdu.data

                new_primary_header = primary_header.copy()
                primary_header_list.append(new_primary_header)

                if channel_name in ('Vis', 'NIR1', 'NIR2'):
                    image_dict[channel_name] = primary_data
                    if channel_name in ('NIR1', 'NIR2'):
                        if 1 < len(hdul):
                            cds_dict[channel_name] = Table(hdul[1].data, copy=True)
                else:
                    swir_data = primary_data
        except (IndexError, ValueError) as e:
            print(f"Error: {e}")
        except KeyError:
            print(f"Error: unknown channel ID '{char}' in '{file.name}' ")
    
    if len(channels) == 1:
        raise ValueError(f'More than one channel needed to combine.')
    
    print(f"Combining channels {channels}")

    # New header
    new_primary_header = utilities.combine_primary_headers(primary_header_list)

    # Combine imaging channels
    image_channels = [ch for ch in channels if ch != "SWIR"]
    if len(image_channels) == 1:
        data = image_dict[image_channels[0]]
        new_primary_hdu = fits.PrimaryHDU(data, new_primary_header)
    else:
        new_image_data = []

        if 'Vis' in channels:
            print('Aligning Vis channel to NIR grid..')
            vis_data = image_dict['Vis']
            nir_data = image_dict.get('NIR1', image_dict.get('NIR2'))
            
            # Align vis channel based on first images of vis and nir
            vis_image = vis_data[0]
            nir_image = nir_data[0]

            transformation_matrix = utilities.estimate_matrix(vis_image, nir_image) # Alignment transformation matrix
            for frame in vis_data:
                # Convert to little-endian float32 for OpenCV
                little_endian = np.ascontiguousarray(frame.astype('<f4'))
                wrapped = cv2.warpPerspective(little_endian, transformation_matrix, (640, 512), flags=cv2.INTER_LINEAR )
                # Convert back to big_endian float32
                big_endian = np.ascontiguousarray(wrapped.astype('>f4'))

                new_image_data.append(big_endian)

        nir1_data = image_dict.get('NIR1')
        nir2_data = image_dict.get('NIR2')
        if nir1_data is not None and nir1_data.size > 0:
            for frame in nir1_data:
                new_image_data.append(frame)
        if nir2_data is not None and nir2_data.size > 0:
            for frame in nir2_data:
                new_image_data.append(frame)

        data_cube = np.stack(new_image_data, axis=0)
        new_primary_hdu = fits.PrimaryHDU(data_cube, new_primary_header)
    
    new_hdul = [new_primary_hdu]
    # Add data to the data cube
    if swir_data is not None:
        swir_hdu = fits.ImageHDU(data=swir_data)
        new_hdul.append(swir_hdu)

    # Extract HDUs containing the binary tables
    if 'NIR1' in cds_dict and 'NIR2' in cds_dict:
        # Combine all columns from both tables
        combined_table_astropy = hstack([cds_dict['NIR1'], cds_dict['NIR2']], join_type='exact')
        combined_table = fits.BinTableHDU(data=combined_table_astropy.as_array())
        new_hdul.append(combined_table)
    elif 'NIR1' in cds_dict:
        combined_table = fits.BinTableHDU(data=cds_dict['NIR2'].as_array())
        new_hdul.append(combined_table)
    elif 'NIR2' in cds_dict:
        combined_table = fits.BinTableHDU(data=cds_dict['NIR2'].as_array())
        new_hdul.append(combined_table)

    hdu_list = fits.HDUList(new_hdul)
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

