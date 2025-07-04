import os
from pathlib import Path
import modules.convertToFits as convertToFits
import modules.utilities as utilities
import modules.convertWavelengths as convertWavelengths
import modules.removeDiagnostic as removeDiagnostic
import modules.

"""
    The main program to execute the data processing pipeline.

    The pipeline consist of following steps:
    1. Convert data into FITS file (Level 0)
        - A directory containing folder for metadata and another folder for acquisition are converted into a single FITS file.
    2. Wavelength calibration (Level 1A)
        - Converts the piezo setpoint values from config files ino corresponding wavelengths.
    3. Extract diagnostic pixels from NIR (Level 1A)
        - Diagnostic pixels are extracted and added as an extension to NIR FITS files.
    4. Removing bad pixels (Level 1A)
        - Remove bad pixels of all 2D images.
    5. Dark background subtraction (Level 1A)
        - Subtract a dark background of all 2D images.
    6. Flat field calibration (Level 1A)
        - Apply flatfield correction to all 2D images.
    7. Radiometric correction (Level 1B)
        - Correct the pixel values into scientific units.
    8. Align & resample to uniform grid (Level 2B)
        - Perform the alignment algorithm to combine all channels into one FITS file with one uniform grid.
    9. Data filtering (Level 3A)
        - Extract the asteroid spectra across all wavelengths for each pixel and perform data filtering for the spectra.
    
    Read more about the pipeline from README.
    Also, more info about the specific steps on their corresponding files.

    This file consist of 2 functions:
    - calibration_pipeline, which performs the calibration procedures for individual sensors.
    - pipeline function, which calls the calibrationPipeline for each sensor and aligns and combines to form the complete file.
"""

def calibration_pipeline(
        input_dir: str | Path, 
        output_dir: str | Path,
        software: str = 'ASPECTCAL v1.0',
        missphase: str = '',
        observph: str = '',
        target: str = 'DIDYMOS',
        object: str = 'Didymos',
    ) -> str:
    """
    These functions perform the level 0 and 1 of the pipeline
    Parmeters:
        path: Path to a folder containing data of an acquisition from a single sensor.
        output: Path to the folder where the fits files will be stored.
    """

    # Verify the existance of directory paths and convert them into Path objects
    input_dir = utilities.verify_directory_path(input_dir)
    output_dir = utilities.verify_directory_path(output_dir)
    # Convert the input directory into FITS file(s)
    fits_files = convertToFits.convert_to_fits(
            dir_path=input_dir, 
            output_dir=output_dir,
            software=software,
            missphase=missphase,
            observph=observph, 
            target=target, 
            object=object 
        )
    
    print(f'New file saved: {fits_files}')

    # Process each fits file separately

    for fits_file in fits_files:

        # Convert the wavelengths
        fits_file = convertWavelengths.convert_wl(fits_file, output_dir)
        print(f'New file created: {fits_file}')

        #Extract diagnostic pixels. The function will only extract pixels on NIR1 and NIR2 channels
        # path = os.path.join(output, fits_file)
        fits_file = removeDiagnostic.extract_diagnostic_pixels(fits_file, output)
        print(f'New file created: {fits_file}')

        # Use removeBadPixels to change the value to the mean of the neighbours
        # path = os.path.join(output, fits_file)
        fits_file = badPixels.remove_bad_pixels(fits_file, output)
        print(f'New file created: {fits_file}')

        # Use darSubstraction to substract the dark frame from fits file and
        # create a new fits file
        # path = os.path.join(output, fits_file)
        fits_file = darkSubtraction.dark_subtraction(fits_file, output)
        print(f'New file created: {fits_file}')

        # Use flatFieldCalibration function
        # path = os.path.join(output, fits_file) 
        fits_file = flatField.flat_field_calibration(fits_file, output)
        print(f'New file created: {fits_file}')

        # # Use radiometricCalbration function
        # path = os.path.join(output, fits_file)
        fits_file = radiometric.radiometric_calibration(fits_file, output)
        print(f'New file created: {fits_file}')

    #Return radiometrically calibrated FITS file (end of level 1)
    return fits_file

def pipeline(vis, nir1, nir2, swir, output):
    """
    levels 0 to 1 to each cahnnel and combine all channels into one FITS
    
    Parameters:
        vis: path to folder containing visible cahnnel acquisition data
        nir1: path to folder containing near-infrared 1 cahnnel acquisition data
        nir2: path to folder containing near-infrared 2 cahnnel acquisition data
        swir: path to folder containing swir cahnnel acquisition data
    """
    #create a folder for fits file
    os.makedirs(output, exist_ok=True)


    # vis = calibration_pipeline(vis, os.path.join(output, "VIS"))
    # nir1 = calibration_pipeline(nir1, os.path.join(output, "NIR1"))
    # nir2 = calibration_pipeline(nir2, os.path.join(output, "NIR2"))
    # swir = calibration_pipeline(swir, os.path.join(output, "SWIR"))

    # # print(f'Successfully calibrated all channels')

    # aligned_fits = alignAndResample.align_fits_files(vis, nir1, nir2, swir, output)
    # print(f"New file created: {aligned_fits}")

acq_path = os.path.join(os.getcwd(), 'test_data/ASPECT_fly_images/acqseq_101')
fits_output_dir = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_fly')

print(calibration_pipeline(acq_path, fits_output_dir))

