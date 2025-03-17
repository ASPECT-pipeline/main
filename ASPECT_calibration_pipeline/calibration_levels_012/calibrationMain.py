import os
import convertToFits
import convertWavelengths
import removeDiagnostic
import badPixels
import darkSubtraction
import flatField
import radiometric
import alignAndResample

"""
    The main program to execute the data processing pipeline levels 0, 1 and 2.

    The pipeline consist of following steps:
    1. Convert data into FITS file (Level 0)
        - A directory containing folder for metadata and another folder for acquisition are converted into a single FITS file.
    2. Wavelength calibration (Level 1A)
        - Converts the piezo setpoint values from config files ino corresponding wavelengths
    3. Extract diagnostic pixels from NIR (Level 1A)
        - Diagnostic pixels are extracted and added as FITS extension from NIR images
    4. Removing bad pixels (Level 1A)
        - Remove bad pixels of all 2D images
    5. Dark background subtraction (Level 1A)
        - Subtract a dark background of all 2D images
    6. Flat field calibration (Level 1A)
        - Apply flatfield correction to all 2D images
    7. Radiometric correction (Level 1B)
        - Correct the pixel values into scientific units
    8. Align & resample to uniform grid (Level 2B)
        - perform the alignment algorithm
    
    More info about the steps on their corresponding files.

    The file consist of 2 functions:
    - calibration_pipeline, which performs the calibration procedures for individual sensors
    - pipeline function, which calls the calibrationPipeline for each sensor and aligns and combines to form the complete file.
"""

def calibration_pipeline(path, output):
    """
    These functions perform the level 0 and 1 of the pipeline
    Parmeters:
        path: Path to a folder containing data of an acquisition from a single sensor
        output: Path to the folder where the fits files will be stored
    """

    # Use convert_to_fits function to convert the data in the directory into a FITS file
    fits_file = convertToFits.convert_to_fits(path, output)
    print(f"Fits file created: {fits_file}")

    # Convert the wavelengths
    path = os.path.join(output, fits_file)
    fits_file = convertWavelengths.convert_wl(path, output)
    print(f'New file created: {fits_file}')

    #Extract diagnostic pixels. The function will only extract pixels on NIR1 and NIR2 channels
    path = os.path.join(output, fits_file)
    fits_file = removeDiagnostic.extract_diagnostic_pixels(path, output)
    print(f'New file created: {fits_file}')

    # Use removeBadPixels to change the value to the mean of the neighbours
    path = os.path.join(output, fits_file)
    fits_file = badPixels.remove_bad_pixels(path, output)
    print(f'New file created: {fits_file}')

    # Use darSubstraction to substract the dark frame from fits file and
    # create a new fits file
    path = os.path.join(output, fits_file)
    fits_file = darkSubtraction.dark_subtraction(path, output)
    print(f'New file created: {fits_file}')

    # Use flatFieldCalibration function
    path = os.path.join(output, fits_file) 
    fits_file = flatField.flat_field_calibration(path, output)
    print(f'New file created: {fits_file}')

    # Use radiometricCalbration function
    path = os.path.join(output, fits_file)
    fits_file = radiometric.radiometric_calibration(path, output)
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



    # vis = calibration_pipeline(vis, os.path.join(output, "VIS"))
    # nir1 = calibration_pipeline(nir1, os.path.join(output, "NIR1"))
    # nir2 = calibration_pipeline(nir2, os.path.join(output, "NIR2"))
    # swir = calibration_pipeline(swir, os.path.join(output, "SWIR"))

    aligned_fits = alignAndResample.align_fits_files(vis, nir1, nir2, swir, output, True)
    # print(f"New file created: {aligned_fits}")
