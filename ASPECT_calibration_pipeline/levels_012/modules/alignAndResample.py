import os
import numpy as np
import cv2
from astropy.io import fits
import modules.utilities as utilities
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path

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
    
    
    Function: align_fits_files
        Description:
            Given paths to 4 fitsfiles, performs the estimateMatrix method from utilities.py and uses the trasnformation matrix to align every 2D visible image from VIS with NIR1. 
            Creates one ImageHDU containing the aligned datacube that has all vis and nir files

"""
def combine_fits_files(files: List[str | Path], output_dir: str | Path) -> str:
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
                raise ValueError(f"Duplicate channel '{channel_name}' found in file '{file.name}'")
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
                    cds_header = hdul[2].header.copy()
                    cds_data = hdul[2].copy()
                    cds_dict[channel_name] = (cds_header, cds_data)

        except (IndexError, ValueError) as e:
            print(f"Error: {e}")
        except KeyError:
            print(f"Error: unknown channel ID '{char}' in '{file.name}' ")
    
    if len(channels) == 1:
        print(f'More than one channel needed to combine.')
        return files[0]
    
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
            new_image_data.append(cv2.warpPerspective(frame, transforamtion_matrix, (640, 512), flags=cv2.INTER_CUBIC))

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
        nir1_cds_header, nir1_cds_data = cds_dict['NIR1']
        nir2_cds_header, nir2_cds_data = cds_dict['NIR2'] 

        # Extract columns from both tables
        columns1 = fits.ColDefs(nir1_cds_data)
        columns2 = fits.ColDefs(nir2_cds_data)
        # Create a combined list of columns
        new_columns = []
        for i, col in enumerate(columns1 + columns2):  # First vlaHdu1, then vlaHdu2
            new_col_name = f"Col_{i+1}"  # Renaming columns in increasing order
            new_columns.append(fits.Column(name=new_col_name, format=col.format, array=col.array))

        # Create a new binary table HDU with renamed columns
        new_vlaHdu = fits.BinTableHDU.from_columns(new_columns)
        HDUs.append(new_vlaHdu)
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





















def align_fits_files(vis_file: str | Path, nir1_file: str | Path, nir2_file: str | Path, swir_file: str | Path, output_dir: str | Path) -> str:
    """
    Parmeters:
        vis_path, nir1_path, nir2_path, swir_file: Path to a FITS file.
        output: Path to the folder where the new fits file will be stored.
    """

    vis_file = Path(vis_file)
    nir1_file = Path(nir1_file)
    nir2_file = Path(nir2_file)
    swir_file = Path(swir_file)
    output_dir = Path(output_dir)
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
        # header_dict = utilities.combine_headers(vis_header, nir1_header, nir2_header, swir_header)
        #New image HDU to contain the whole data cube
        new_image_HDU = fits.ImageHDU()
        #Add metadata
        # new_image_HDU = utilities.append_header(new_image_HDU, header_dict)

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

        hdu_list = fits.HDUList(HDUs)
        # File name for new fits
        stem = vis_file.stem
        suffix = vis_file.suffix
        new_calibration_level = '2B'
        file_name = 'ASP'+ stem[3:25] + new_calibration_level + suffix
        primary_header = hdu_list[0].header
        primary_header['FILENAME'] = file_name
        primary_header['PROCLEVL'] = new_calibration_level
        # Create the new fits file with dark-subtracted images
        fits_file = os.path.join(output_dir, file_name)
        hdu_list.writeto(fits_file, overwrite=True)

        return(fits_file)