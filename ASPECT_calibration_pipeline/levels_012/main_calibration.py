import os
from pathlib import Path
from typing import Tuple, List
import modules.convertToFits as convertToFits
import modules.utilities as utilities
import modules.calibrateHeader as calibrateHeader
import modules.extractCDS as extractCDS
import modules.darkSubtraction as darkSubtraction
import modules.flatField as flatField
import modules.badPixels as badPixels
import modules.radiometric as radiometric
import modules.mergeFits as mergeFits

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

    # Convert the input directory into FITS file(s)
    fits_file = convertToFits.convert_to_fits(
            dir_path=input_dir, 
            output_dir=output_dir,
            channel=channel,
            channel_info=channel_info,
            software=software,
            missphase=missphase,
            observph=observph, 
            target=target, 
            object=object 
        )
    
    print(f'New file saved: {fits_file}')

    # Process the calibration steps

    fits_file = calibrateHeader.calibrate_header(fits_file, output_dir)
    print(f'New file created: {fits_file}')

    # Extract diagnostic pixels from NIR2 and NIR2. Convert the values to float64
    fits_file = extractCDS.extract_cds_pixels(fits_file, output_dir)
    print(f'New file created: {fits_file}')

    # Use darSubstraction to substract the dark frame from fits file and
    fits_file = darkSubtraction.dark_subtraction(fits_file, output_dir)
    print(f'New file created: {fits_file}')

    # Correct the flatfield image 
    fits_file = flatField.flat_field_calibration(fits_file, output_dir)
    print(f'New file created: {fits_file}')

    # Change the value of bad pixels to the mean of the neighbours
    fits_file = badPixels.remove_bad_pixels(fits_file, output_dir)
    print(f'New file created: {fits_file}')

    # Apply radiometric calibration
    fits_file = radiometric.radiometric_calibration(fits_file, output_dir)
    print(f'New file created: {fits_file}')

    #Return radiometrically calibrated FITS file (end of level 1)
    return fits_file

def pipeline_levels_01(
        input_dir: str | Path, 
        output_dir: str | Path,
        software: str = 'ASPECTCAL v1.0',
        missphase: str = '',
        observph: str = '',
        target: str = 'DIDYMOS',
        object: str = 'Didymos',
    ) -> str:
    """
    Executes the calibration levels 0, 1, 2 of the ASPECT calibation pipeline. 
    The pipeline consist of converting the raw binary data into FITS files,
    performing clibration steps to each channel individually, and lastly 
    combining all cahnnels into one FITS file containing one calibrated 
    hyperspectral data cube.
    
    Parameters:
        input_dir (str | Path):    Directory containing the acquisition files.
        output_dir (str | Path):  Directory where a new directory containg all new files will be sotred.
        software (str): Pipeline software identification.
        missphase (str): Identification of the mission phase.
        observph (str): Identification of the observation ID
        target (str): Taret in SPICE format
        object (str): Unique name for target
    
    Return: Path to the single combined FITS file as the result of level 2B
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir = Path(output_dir) / observph # output directory for this acquisition
    output_dir.mkdir(parents=True, exist_ok=True) # create the directory for this acquisition

    # Verify the existance of directory paths and convert them into Path objects
    input_dir = utilities.verify_directory_path(input_dir)
    output_dir = utilities.verify_directory_path(output_dir)

    acq_dir, meta_dir, telemetry_path, config_path = utilities.verify_acquisition_directory(input_dir)

    channel_acq = utilities.channel_files(acq_dir) # Dict[channel, (original_channel_name, [files names belongs to this channel])]
    channel_names = list(channel_acq.keys()) # List of all channels in acquisition folder

    level_1B_files = []
    for channel in channel_names:
        channel_info = channel_acq[channel] # Tuple[original_filename, List[filenames_belongs_this_channels]]

        calibrated_fits_file = calibration_pipeline(input_dir=input_dir, 
                                                    output_dir=output_dir, 
                                                    channel=channel,
                                                    channel_info=channel_info,
                                                    software=software,
                                                    missphase=missphase,
                                                    observph=observph,
                                                    target=target,
                                                    object=object)

        level_1B_files.append(calibrated_fits_file)
    print(f'Successfully calibrated all channels')

    return (output_dir, level_1B_files)


def pipeline_level_02(fits_dir: str | Path, output_dir: str | Path, instrument: str = 'vis-nir1-nir2') -> str:
    """
    Combines fits files into one single fits file.

    Parameters:
        fits_dir (str | Path): Directory path to fits files ()
        output_dir (str | Path): Path to the directory where the new file is stored.
        instrument (str): Defines which channels are combined in level 2

    Returns 
        (str): path to the created FITS file
    """
    fits_dir = Path(fits_dir)

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
    print(f'Fits files combined. New file: {combined_fits_file}')

    return combined_fits_file

acq_path = os.path.join(os.getcwd(), 'test_data/ASPECT_fly_images/acqseq_104')
fits_output_dir = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_fly')
fits_output_dir_ = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_fly/104')

acq_path_sim = os.path.join(os.getcwd(), 'test_data/ASPECT_simulated_images/2027-03-23_06_00_00-McEwen')
fits_output_dir_sim = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_simulated')
fits_output_dir_sim_ = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/ASPECT_simulated/2027-03-23_06_00_00-McEwen')


# pipeline_levels_01(acq_path, fits_output_dir, missphase='FLY_TEST', observph='104', object='DARK')
pipeline_level_02(fits_output_dir_, fits_output_dir_)

# Python3 ASPECT_calibration_pipeline/levels_012/main_calibration.py