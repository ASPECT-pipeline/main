import os
from pathlib import Path
from astropy.io import fits
from typing import Tuple, List
import levels_012.modules.convertToFits as convertToFits
import levels_012.modules.utilities as utilities
import levels_012.modules.calibrateHeader as calibrateHeader
import levels_012.modules.extractCDS as extractCDS
import levels_012.modules.darkSubtraction as darkSubtraction
import levels_012.modules.flatField as flatField
import levels_012.modules.badPixels as badPixels
import levels_012.modules.radiometric as radiometric
import levels_012.modules.mergeFits as mergeFits
from config import observph, missphase
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
        channel: str,
        channel_info: Tuple[str, List[str]],
        differential: bool
    ) -> str:

    """
    These functions perform the level 0 and 1 of the pipeline
    Parmeters:
        path: Path to a folder containing data of an acquisition from a single sensor.
        output: Path to the folder where the fits files will be stored.
    """

    diff = None
    # Search for the differential encoding offsets, if not found the differential decoding is not executed
    if differential:
        acq_dir = Path(input_dir) / 'acq_000'
        for subdir in acq_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / 'diff_encoding.json'
                if candidate.is_file():
                    diff= Path(candidate)
    # Convert the input directory into FITS file(s)
    fits_file = convertToFits.convert_to_fits(
            dir_path=input_dir, 
            output_dir=output_dir,
            channel=channel,
            channel_info=channel_info,
            diff= diff
        )
    
    print(f'New fits file created: {fits_file}')

    # Process the calibration steps

    fits_file = calibrateHeader.calibrate_header(fits_file, output_dir)
    print(f'New fits file created: {fits_file}')

    fits_file = Path(fits_file)

    with fits.open(fits_file, memmap=False) as hdul:


        # Convert the data to float64 for calibration
        hdul = utilities.convert_to_float64(hdul)

        # Extract diagnostic pixels from NIR2 and NIR2. Convert the values to float64
        hdul = extractCDS.extract_cds_pixels(hdul)
        print(f'CDS extracted')

        # Subtrack the dark frame from each image
        hdul = darkSubtraction.dark_subtraction(hdul)
        print(f'Dark frame subtracted')

        # Apply flatfield correction
        hdul = flatField.flat_field_calibration(hdul)
        print(f'Flat field calibrated')


        # Replace bad pixels with neigbours
        hdul = badPixels.replace_bad_pixels(hdul)
        print(f'Bad pixels replaced')


        # Apply radiometric calibration
        hdul = radiometric.radiometric_calibration(hdul)
        print(f'Radiometric calibrated')

        stem = fits_file.stem
        suffix = fits_file.suffix
        new_calibration_level = '1B'
        file_name = stem[:25] + new_calibration_level + suffix
        primary_header = hdul[0].header
        primary_header['FILENAME'] = file_name
        primary_header['PROCLEVL'] = new_calibration_level

        hdul = utilities.convert_to_float32(hdul)

    # create the new fits file with dark-subtracted images
    fits_file = Path(output_dir) / file_name
    hdul.writeto(fits_file, overwrite=True)
    print(f'New fits file created: {fits_file}')

    #Return radiometrically calibrated FITS file (end of level 1)
    return fits_file

def pipeline_levels_01(
        input_dir: str | Path, 
        output_dir: str | Path,
        differential: bool
    ) -> str:
    """
    Executes the calibration levels 0, 1, 2 of the ASPECT calibation pipeline. 
    The pipeline consist of converting the raw binary data into FITS files,
    performing clibration steps to each channel individually, and lastly 
    combining all cahnnels into one FITS file containing one calibrated 
    hyperspectral data cube.
    
    Parameters:
        input_dir (str | Path):     Directory containing the acquisition files.
        output_dir (str | Path):    Directory where a new directory containg all new files will be sotred.
        differential (bool):        True if differnetial encoding is used for the input images.
    
    Return: Path to the single combined FITS file as the result of level 2B
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Verify the existance of directory paths and convert them into Path objects
    input_dir = utilities.verify_directory_path(input_dir)
    output_dir = utilities.verify_directory_path(output_dir)

    acq_dir, meta_dir, telemetry_path, config_path = utilities.verify_acquisition_directory(input_dir)

    channel_acq = utilities.channel_files(acq_dir) # Dict[channel, (original_channel_name, [files names belongs to this channel])]
    channel_names = list(channel_acq.keys()) # List of all channels in acquisition folder

    for channel in channel_names:
        print(f'Calibrating channel: {channel}')
        channel_info = channel_acq[channel] # Tuple[original_filename, List[filenames_belongs_this_channels]]
        calibrated_fits_file = calibration_pipeline(input_dir=input_dir, 
                                                    output_dir=output_dir, 
                                                    channel=channel,
                                                    channel_info=channel_info,
                                                    differential=differential
                                                    )
        if missphase == 'SIMULATE':
            utilities.update_fits_exposure(calibrated_fits_file, None)
            utilities.update_fits_wl(calibrated_fits_file, None)

    print(f'Successfully calibrated all channels')

    return (output_dir)


def pipeline_level_02(input_dir: str | Path, output_dir: str | Path, instrument: str = 'vis-nir1-nir2') -> str:
    """
    Combines fits files into one single fits file.

    Parameters:
        fits_dir (str | Path): Directory path to fits files
        output_dir (str | Path): Path to the directory where the new file is stored.
        instrument (str): Defines which channels are combined in level 2

    Returns 
        (str): path to the created FITS file
    """
    fits_dir = Path(input_dir)

    channel_map = {
        'vis' : 0,
        'nir1' : 1,
        'nir2' : 2,
        'swir' : 3
    }

    channels = instrument.split('-')
    valid_channel_ids = {str(channel_map[ch]) for ch in channels} # a list of channel IDs that match the instrument parameter

    files_to_be_combined = []
    for file in fits_dir.iterdir():
        try:
            stem = file.stem
            if not stem.endswith('1B'):
                continue
            if file.suffix.lower() != '.fits':
                continue
            if stem[2] in valid_channel_ids:
                files_to_be_combined.append(file)
        except IndexError:
            print(f'Skipping malformed filename: {file.name}')

    combined_fits_file = mergeFits.merge_fits_files(files=files_to_be_combined, output_dir=output_dir)

    return combined_fits_file

# ASPECT FLY
acq_path = os.path.join(os.getcwd(), 'test_data/ASPECT_fly_images/acqseq_104')
fits_output_dir = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_fly')
fits_output_dir_ = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_fly/104')

# ASPECT simulated
acq_path_sim = os.path.join(os.getcwd(), 'test_data/ASPECT_simulated_images/2027-03-23_06_00_00-McEwen')
fits_output_dir_sim = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_simulated')
fits_output_dir_sim_ = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_simulated/2027-03-23_06_00_00')

# ASPECT Diffetential encoded
autoseq_dir = os.path.join(os.getcwd(), 'test_data/ASPECT_Autoseq_20240809/acqseq_505')
autoseq_output_dir = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_DIFF')

# pipeline_levels_01(autoseq_dir, autoseq_output_dir, differential=True)
# pipeline_level_02(fits_output_dir_sim_, fits_output_dir_sim_)

# Python3 ASPECT_calibration_pipeline/levels_012/main_calibration.py