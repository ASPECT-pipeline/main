import sys
import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd

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
import dataFiltering

#Path to VIS, NIR1 and NIR2
vis = os.path.join(main_dir, "test_data/levels_012_test/test_data/vis_ho_600w_1875")
nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_data/nir1_ho_600w_7500")
nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_data/nir2_ho_600w_7500")
swir = os.path.join(main_dir, "test_data/levels_012_test/test_data/swir_test22")

simulated_vis = os.path.join(main_dir, "test_data/levels_012_test/test_data/D1v6-10km-vis-noiseless-20ms_simulated_VIS.fits")
simulated_nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_data/D1v5-10km-noiseless-40ms_simulated_NIR1.fits")
simulated_nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_data/D1v5-10km-noiseless-40ms_simulated_NIR2.fits")
swir_fits = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/SWIR/SWIR_1B_Rc.fits")

output = os.path.join(main_dir, "test_data/levels_012_test/test_output")
outputPath = os.path.join(output, "simulated_test_3")


# calibrationMain.pipeline(vis, nir1, nir2, swir, outputPath)
# calibrationMain.pipeline(simulated_vis, simulated_nir1, simulated_nir2, swir_fits, outputPath)



"""
Function to read fits files
"""

def visulize_alignment(img1, img2):
    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)  # leave space for slider

    # Show base image
    ax.imshow(img1, cmap='gray', interpolation='none')

    # Overlay image with initial opacity
    overlay = ax.imshow(img2, cmap='jet', alpha=0.5, interpolation='none')

    ax.set_title("Adjust overlay opacity")
    ax.axis("off")

    # Slider axis: [left, bottom, width, height]
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider = Slider(ax_slider, 'Opacity', 0.0, 1.0, valinit=0.5)

    # Update function
    def update(val):
        overlay.set_alpha(slider.val)
        fig.canvas.draw_idle()

    # Connect the slider to update function
    slider.on_changed(update)

    plt.show()

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

            
        # print(f'Total Size: {total_size}')
        print()

def read_output_files():
    vis = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_3/VIS/VIS_1B_Rc.fits")
    nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_3/NIR1/NIR1_1B_Rc.fits")
    nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_3/NIR2/NIR2_1B_Rc.fits")
    swir = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_3/SWIR/SWIR_1B_Rc.fits")
    combined = os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/ASPECT_full_datacube.fits")

    simulated_vis = os.path.join(main_dir, "test_data/levels_012_test/test_data/D2v6-10km-vis-noiseless-20ms_simulated_VIS.fits")
    simulated_nir1 = os.path.join(main_dir, "test_data/levels_012_test/test_data/D2v5-10km-noiseless-40ms_simulated_NIR1.fits")
    simulated_nir2 = os.path.join(main_dir, "test_data/levels_012_test/test_data/D1D2v5-10km-40ms_simulated_NIR2.fits")

    simulated_full = os.path.join(main_dir, "test_data/levels_012_test/test_output/simulated_test_3/D1v6v5_simulated_full_datacube.fits")

    # dataFiltering.extract_asteroid(simulated_full)
    # read_fits_file(vis)
    # read_fits_file(nir1)
    # read_fits_file(nir2)
    # read_fits_file(swir)
    # read_fits_file(combined)
    # read_fits_file(simulated_vis)
    # read_fits_file(simulated_nir1)
    # read_fits_file(simulated_nir2)
    read_fits_file(simulated_full)

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
# add_metadata(os.path.join(main_dir, "test_data/levels_012_test/test_data/D2v6-10km-vis-noiseless-20ms_simulated_VIS.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/VIS/VIS_1B_Rc.fits"))
# add_metadata(os.path.join(main_dir, "test_data/levels_012_test/test_data/D2v5-10km-noiseless-40ms_simulated_NIR1.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR1/NIR1_1B_Rc.fits"))
# add_metadata(os.path.join(main_dir, "test_data/levels_012_test/test_data/D2v5-10km-noiseless-40ms_simulated_NIR2.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR2/NIR2_1B_Rc.fits"))

def add_diagnostics(simulated_path, data_path):
    with fits.open(simulated_path, mode='update') as simulated_hdul, fits.open(data_path) as data_hdul:
        simulated_hdul.append(data_hdul[2]) # append the diagnosti pixels
        simulated_hdul.flush()# writes the changes to the file 

# add_diagnostics(os.path.join(main_dir, "test_data/levels_012_test/test_data/D2v5-10km-noiseless-40ms_simulated_NIR1.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR1/NIR1_1B_Rc.fits"))
# add_diagnostics(os.path.join(main_dir, "test_data/levels_012_test/test_data/D2v5-10km-noiseless-40ms_simulated_NIR2.fits"), os.path.join(main_dir, "test_data/levels_012_test/test_output/test_1/NIR2/NIR2_1B_Rc.fits"))



"""

Testing function for 3A

"""

def test_level_3A(path: str):

    #create spectra that represents the output of filter_asteroid_spectra
    excel_path = path
    list_of_spectras= [
        pd.read_excel(excel_path, sheet_name='600W, 10000|2500, HO', usecols='D', skiprows=4, nrows=33),
        pd.read_excel(excel_path, sheet_name='600W, 7500|1875, HO', usecols='I', skiprows=4, nrows=33),
        pd.read_excel(excel_path, sheet_name='200W, 10000|2500, HO', usecols='D', skiprows=4, nrows=33),
        pd.read_excel(excel_path, sheet_name='400W, 10000|2500, HO', usecols='D', skiprows=4, nrows=33),
        pd.read_excel(excel_path, sheet_name='400W, 10000|2500, LO', usecols='C', skiprows=4, nrows=33),
    ]
    df_wl = pd.read_excel(excel_path, sheet_name='600W, 10000|2500, HO', usecols='M', skiprows=4, nrows=33)
    df_wl = df_wl.apply(pd.to_numeric, errors="coerce") 
    df_wl.dropna(inplace=True)
    wl = df_wl.iloc[:,0].to_numpy()

    spectras = []
    coords = [0,1,2,3,4]
    
    for i, spectra in enumerate(list_of_spectras):
        df = spectra.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        points = df.iloc[:,0].to_numpy()
        spectras.append(points)
    
    print(f'spectras: {len(spectras)}')


    combined = list(zip(coords, spectras))
    test_data = combined, wl

    result = dataFiltering.filter_spectra(path, test=True, test_data=test_data)

    coords, smooth_spectras = zip(*result)
    coords = np.array(coords) 
    smooth_spectras = np.array(smooth_spectras)

    print(coords)
    print(smooth_spectras)

    fig, axs = plt.subplots(2, 2, figsize=(12,5))
    axs = axs.flatten()

    #first spectra
    axs[0].plot(wl, spectras[0], 'ro-', label="Original spectra")
    axs[0].plot(wl, smooth_spectras[0], 'bo-', label="Smoothed")
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Intensity")
    axs[0].set_title(f"Spectra 1")

    #second spectra
    axs[1].plot(wl, spectras[1], 'ro-', label="Original spectra")
    axs[1].plot(wl, smooth_spectras[1], 'bo-', label="Smoothed")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Intensity")
    axs[1].set_title(f"Spectra 2")

    #third spectra
    axs[2].plot(wl, spectras[2], 'ro-', label="Original spectra")
    axs[2].plot(wl, smooth_spectras[2], 'bo-', label="Smoothed")
    axs[2].set_xlabel("Wavelength (nm)")
    axs[2].set_ylabel("Intensity")
    axs[2].set_title(f"Spectra 3")

    #fourth spectra
    axs[3].plot(wl, spectras[3], 'ro-', label="Original spectra")
    axs[3].plot(wl, smooth_spectras[3], 'bo-', label="Smoothed")
    axs[3].set_xlabel("Wavelength (nm)")
    axs[3].set_ylabel("Intensity")
    axs[3].set_title(f"Spectra 3")

    plt.tight_layout()
    plt.show()

    return 

test_level_3A_path = os.path.join(main_dir, "test_data/REF_MEAS_upd_wl.xlsx")

# print(f'test path:')
# print(test_level_3A_path)

# test_level_3A(test_level_3A_path)

"""

Testing function for the whole pipeline

"""

def test_level_01(path: str, output: str):
    #Test converting binary files into Fits and calibrating the data
    # Use convert_to_fits function to convert the data in the directory into a FITS file
    
    print(f'Testing level 0 - 1')

    calibrationMain.pipeline(vis, nir1, nir2, swir, output)


    return

# test_level_01(vis, outputPath)