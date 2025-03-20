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
import extractAsteroid

#Path to VIS, NIR1 and NIR2
vis = os.path.join(main_dir, "test_data/levels_012_test/test_data/vis_lo_600w_2500")
nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_data/nir1_lo_600w_2500")
nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_data/nir2_lo_600w_2500")
swir = os.path.join(main_dir, "test_data/levels_012_test/test_data/swir_test22")

simulated_vis = os.path.join(main_dir, "test_data/levels_012_test/test_data/simulated_VIS.fits")
simulated_nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_data/simulated_NIR1.fits")
simulated_nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_data/simulated_NIR2.fits")
swir_fits = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/SWIR/SWIR_1B_Rc.fits")

output = os.path.join(main_dir, "test_data/levels_012_test/test_output")
outputPath = os.path.join(output, "test_1")


# calibrationMain.pipeline(simulated_vis, simulated_nir1, simulated_nir2, swir_fits, outputPath)

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
    vis = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/VIS/VIS_1B_Rc.fits")
    nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR1/NIR1_1B_Rc.fits")
    nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR2/NIR2_1B_Rc.fits")
    swir = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/SWIR/SWIR_1B_Rc.fits")
    combined = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/ASPECT_full_datacube.fits")

    simulated_vis = os.path.join(main_dir, "test_data/levels_012_test/test_data/simulated_VIS.fits")
    simulated_nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_data/simulated_NIR1.fits")
    simulated_nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_data/simulated_NIR2.fits")

    simulated_full = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/simulated_full_datacube.fits")

    extractAsteroid.extract_asteroid(simulated_full)
    # read_fits_file(vis)
    # read_fits_file(nir1)
    # read_fits_file(nir2)
    # read_fits_file(swir)
    # read_fits_file(combined)
    # read_fits_file(simulated_vis)
    # read_fits_file(simulated_nir1)
    # read_fits_file(simulated_nir2)
    # read_fits_file(simulated_full)

read_output_files()

"""
Function to create manually an example fits file 
"""

def add_metadata(simulated_path, data_path):
    with fits.open(simulated_path, mode='update') as simulated_hdul, fits.open(data_path) as data_hdul:
        simulated_header = simulated_hdul[1].header
        metadata_header = data_hdul[1].header

        simulated_header['CHANNEL'] = metadata_header.get('CHANNEL') #Channel
        simulated_header['ORDER'] = metadata_header.get('ORDER') # Higher or lower order simply h or l
        simulated_header['PIEZO1'] = metadata_header.get('PIEZO1') # set point 1 values
        simulated_header['PIEZO2'] = metadata_header.get('PIEZO2') # set point 2 values
        simulated_header['PIEZO3'] = metadata_header.get('PIEZO3') # set point 3 values
        simulated_header['EXPOS'] = metadata_header.get('EXPOS') #All exposure times as a string

        simulated_hdul.flush() # writes the changes to the file 
# add_metadata(os.path.join(main_dir, "test_data/levels_012_test/test_data/simulated_VIS.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/VIS/VIS_1B_Rc.fits"))
# add_metadata(os.path.join(main_dir, "test_data/levels_012_test/test_data/simulated_NIR1.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR1/NIR1_1B_Rc.fits"))
# add_metadata(os.path.join(main_dir, "test_data/levels_012_test/test_data/simulated_NIR2.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR2/NIR2_1B_Rc.fits"))

def add_diagnostics(simulated_path, data_path):
    with fits.open(simulated_path, mode='update') as simulated_hdul, fits.open(data_path) as data_hdul:
        simulated_hdul.append(data_hdul[2]) # append the diagnosti pixels
        simulated_hdul.flush()# writes the changes to the file 

# add_diagnostics(os.path.join(main_dir, "test_data/levels_012_test/test_data/simulated_NIR2.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR2/NIR2_1B_Rc.fits"))
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



