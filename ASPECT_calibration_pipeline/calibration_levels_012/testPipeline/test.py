import sys
import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

"""
Test file to run the pipeline on test data and read the created files 

to run from main directory:

python3 ASPECT_calibration_pipeline/calibration_levels_012/testPipeline/test.py

"""

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_dir)

import calibrationMain
import utilities
import alignAndResample

#Path to VIS, NIR1 and NIR2
vis = os.path.join(main_dir, "test_data/levels_012_test/test_data/vis_lo_600w_2500")
nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_data/nir1_lo_600w_2500")
nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_data/nir2_lo_600w_2500")
swir = os.path.join(main_dir, "test_data/levels_012_test/test_data/swir_test22")

output = os.path.join(main_dir, "test_data/levels_012_test/test_output")
outputPath = os.path.join(output, "test_1")

# calibrationMain.pipeline(vis,nir1,nir2,swir,outputPath)

def read_fits_file(path):
    with fits.open(path) as hdul:
        print(f'Fitsfile from path:')
        print({path})

        for i, hdu in enumerate(hdul):

            h = hdu.header
            print(f'Header for HDU {i}')
            print(repr(h))

        for frame in hdul[1].data:
            plt.imshow(frame, cmap='gray')
            plt.show()
        total_size = hdul._file.size
            
        print(f'Total Size: {total_size}')
        print()

def read_output_files():
    vis = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/VIS/VIS_cropped.fits")
    nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR1/NIR1_1B_Rc.fits")
    nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR2/NIR2_1B_Rc.fits")
    swir = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/SWIR/SWIR_1B_Rc.fits")
    combined = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/ASPECT_full_datacube.fits")

    # alignAndResample.alignFitsFiles(vis,nir1,nir2,swir, outputPath, False)
    # read_fits_file(vis)
    # read_fits_file(nir1)
    # read_fits_file(nir2)
    # read_fits_file(swir)
    read_fits_file(combined)

read_output_files()

"""
Function to create manually an example fits file 
"""



def crop_vis(vis_path, output_folder):
    new_height, new_width = 512, 640 #NIR dimensions

    with fits.open(vis_path) as hdul:
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)
        data_cube = img_HDU.data
        num_slices, height, width = data_cube.shape


        center_y, center_x = height // 2, width // 2
        start_y = center_y - new_height // 2
        end_y = center_y + new_height // 2
        start_x = center_x - new_width // 2
        end_x = center_x + new_width // 2

        # Create new list of HDU's and append the cube to it
        HDUs = []
        HDUs.insert(0, hdul[0])

        #To store the calibrated datacube
        cropped_cube = np.empty((num_slices, new_height, new_width), dtype=data_cube.dtype)

        #loop over the 2D images inside the extension
        for i in range(num_slices):
            cropped_cube[i] = data_cube[i, start_y:end_y, start_x:end_x]
        
       
        ImageHDU = fits.ImageHDU(data=cropped_cube, header=img_header)
        HDUs.append(ImageHDU)

        #File name for new fits
        file_name = f'VIS/{channel}_cropped.fits'
        fits_file = os.path.join(output_folder, file_name)

        # create the new fits file with cropped vis 
        hdulist = fits.HDUList(HDUs)
        hdulist.writeto(fits_file, overwrite=True)

# crop_vis(os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/VIS/VIS_1B_Rc.fits"), outputPath)



