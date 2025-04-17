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
import dataFiltering
import convertToFits
import convertWavelengths
import removeDiagnostic
import badPixels
import darkSubtraction
import flatField
import radiometric
import alignAndResample

#Path to VIS, NIR1 and NIR2
vis = os.path.join(main_dir, "test_data/levels_012_test/test_data/vis_lo_600w_2500")
nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_data/nir1_lo_600w_2500")
nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_data/nir2_lo_600w_2500")
swir = os.path.join(main_dir, "test_data/levels_012_test/test_data/swir_test22")

simulated_vis = os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-0.05ms_simulated_VIS.fits")
simulated_nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-40ms_simulated_NIR1.fits")
simulated_nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-40ms_simulated_NIR2.fits")
swir_fits = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/SWIR/SWIR_1B_Rc.fits")

output = os.path.join(main_dir, "test_data/levels_012_test/test_output")
outputPath = os.path.join(output, "test_2")


# calibrationMain.pipeline(simulated_vis, simulated_nir1, simulated_nir2, swir_fits, output)



"""
Function to read fits files
"""

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

    simulated_vis = os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-0.05ms_simulated_VIS.fits")
    simulated_nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-40ms_simulated_NIR1.fits")
    simulated_nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-40ms_simulated_NIR2.fits")

    simulated_full = os.path.join(main_dir, "test_data/levels_012_test/test_output/D1D2v5_simulated_full_datacube.fits")

    # dataFiltering.extract_asteroid(simulated_full)
    # read_fits_file(simulated_vis)
    # read_fits_file(nir1)
    # read_fits_file(nir2)
    # read_fits_file(swir)
    # read_fits_file(combined)
    # read_fits_file(simulated_vis)
    # read_fits_file(simulated_nir1)
    # read_fits_file(simulated_nir2)
    read_fits_file(simulated_full)

# read_output_files()

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
# add_metadata(os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-0.05ms_simulated_VIS.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/VIS/VIS_1B_Rc.fits"))
# add_metadata(os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-40ms_simulated_NIR1.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR1/NIR1_1B_Rc.fits"))
# add_metadata(os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-40ms_simulated_NIR2.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR2/NIR2_1B_Rc.fits"))

def add_diagnostics(simulated_path, data_path):
    with fits.open(simulated_path, mode='update') as simulated_hdul, fits.open(data_path) as data_hdul:
        simulated_hdul.append(data_hdul[2]) # append the diagnosti pixels
        simulated_hdul.flush()# writes the changes to the file 

# add_diagnostics(os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-40ms_simulated_NIR1.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR1/NIR1_1B_Rc.fits"))
# add_diagnostics(os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-40ms_simulated_NIR2.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR2/NIR2_1B_Rc.fits"))



"""

Testing function for the whole pipeline

"""

def test_level_01(path: str, output: str):
    #Test converting binary files into Fits and calibrating the data
    # Use convert_to_fits function to convert the data in the directory into a FITS file
    
    print(f'Testing level 0 - 1')

    calibrationMain.pipeline(vis, nir1, nir2, swir, output)


    return

test_level_01(vis, outputPath)