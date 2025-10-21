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
import levels_012.modules.reflectance as reflectance
from config import MISSPHAS, instrument
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
        input_dir (str | Path): Path to a folder containing data of an acquisition from a single sensor.
        output_dir (str | Path): Path to the folder where the fits files will be stored.
        channel (str): VIS, NIR1, NIR2, or SWIR
        channel_info (Tuple[str, List[str]]): [original filename, List[files that belongs to the channel]]
        differential (bool):  Differential encoding is used for the files. 
    """

    diff = None
    # Search for the differential encoding offsets, if not found the differential decoding is not executed
    if differential:
        matches = list(Path(input_dir).rglob('diff_encoding.json'))

        if not matches:
            raise FileNotFoundError(f"'differential' is True but no 'diff_encoding.json' file was found under {input_dir}")
        elif len(matches) > 1:
            raise RuntimeError(f"Multiple 'diff_encoding.json' files found under {input_dir}. Expected only one.")
        else:
            diff = matches[0]


    # Convert the input directory into FITS file(s)
    fits_file = convertToFits.convert_to_fits(
            dir_path=input_dir, 
            output_dir=output_dir,
            channel=channel,
            channel_info=channel_info,
            diff=diff
        )
    
    print(f'New fits file created: {fits_file}')
    print(f'---------- LEVEL 0A COMPLETED ----------')

    # Process the calibration steps

    fits_file = calibrateHeader.calibrate_header(fits_file, output_dir)
    print(f'New fits file created: {fits_file}')
    print(f'---------- LEVEL 1A COMPLETED ----------')

    fits_file = Path(fits_file)

    with fits.open(fits_file, memmap=False) as hdul:


        # Convert the data to float64 for calibration
        hdul = utilities.convert_to_float64(hdul)

        # Extract diagnostic pixels from NIR2 and NIR2. Convert the values to float64
        hdul = extractCDS.extract_cds_pixels(hdul)

        # Subtrack the dark frame from each image
        hdul = darkSubtraction.dark_subtraction(hdul)

        # Apply flatfield correction
        hdul = flatField.flat_field_calibration(hdul)

        # Replace bad pixels with neigbours
        hdul = badPixels.replace_bad_pixels(hdul)

        # Apply radiometric calibration
        hdul = radiometric.radiometric_calibration(hdul)

        stem = fits_file.stem
        suffix = fits_file.suffix
        new_calibration_level = '1B'
        file_name = stem[:25] + new_calibration_level + suffix
        primary_header = hdul[0].header
        primary_header['FILENAME'] = file_name
        primary_header['PROCLEVL'] = new_calibration_level

        hdul = utilities.convert_to_float32(hdul)

    # create the new fits
    fits_file = Path(output_dir) / file_name
    hdul.writeto(fits_file, overwrite=True)
    print(f'New fits file created: {fits_file}')
    print(f'---------- LEVEL 1B COMPLETED ----------')

    """
    1C implementation here
    """
    with fits.open(fits_file, memmap=False) as hdul:
     # Convert the data to float64 for calibration
        hdul = utilities.convert_to_float64(hdul)

        hdul = reflectance.reflectance_calibration(hdul, fwhm_nm=30)

        hdul = utilities.convert_to_float32(hdul)

        stem = fits_file.stem
        suffix = fits_file.suffix
        new_calibration_level = '1C'
        file_name = stem[:25] + new_calibration_level + suffix
        primary_header = hdul[0].header
        primary_header['FILENAME'] = file_name
        primary_header['PROCLEVL'] = new_calibration_level

    fits_file = Path(output_dir) / file_name
    hdul.writeto(fits_file, overwrite=True)
    print(f'New fits file created: {fits_file}')
    print(f'---------- LEVEL 1C COMPLETED ----------')

    #Return radiometrically calibrated FITS file (end of level 1)
    return fits_file

def pipeline_levels_01(
        input_dir: str | Path, 
        output_dir: str | Path,
        differential: bool
    ) -> str:
    """
    Executes the calibration levels 0, 1 of the ASPECT calibation pipeline. 
    The pipeline consist of converting the raw binary data into FITS files,
    performing clibration steps to each channel individually. The calibrated 
    FITS files are saved to output_dir.
    
    Parameters:
        input_dir (str | Path):     Directory containing the acquisition files.
        output_dir (str | Path):    Directory where a new directory containg all new files will be sotred.
        differential (bool):        True if differnetial encoding is used for the input images.
    
    Return: 
        (str): path to the created FITS file
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Verify the existance of directory paths and convert them into Path objects
    input_dir = utilities.verify_directory_path(input_dir)
    output_dir = utilities.verify_directory_path(output_dir)

    acq_dir = utilities.verify_acquisition_directory(input_dir)
    output_dir = Path(output_dir) / acq_dir.name # output directory for this acquisition
    output_dir.mkdir(parents=True, exist_ok=True) # create the directory for this acquisition

    print('Separating acquisition directory into channel specific files.')
    channel_acq = utilities.channel_files(acq_dir) # Dict[channel, (original_channel_name, [files names belongs to this channel])]
    channel_names = list(channel_acq.keys()) # List of all channels in acquisition folder
    print(f'Channels found: {channel_names}.')
    channels_to_calibrate = [x for x in channel_names if x in instrument]
    print(f'Calibrating channels: {channels_to_calibrate}')
    for channel in channels_to_calibrate:
        print()
        print(f'Calibrating channel: {channel}')
        channel_info = channel_acq[channel] # Tuple[original_filename, List[filenames from this channel]]
        calibrated_fits_file = calibration_pipeline(input_dir=input_dir, 
                                                    output_dir=output_dir, 
                                                    channel=channel,
                                                    channel_info=channel_info,
                                                    differential=differential
                                                    )

        # # Modify these for simulated images
        # if MISSPHAS == 'SIMULATE':
        #     utilities.update_fits_exposure(calibrated_fits_file, None)
        #     utilities.update_fits_wl(calibrated_fits_file, None)

    print(f'Successfully calibrated all channels')

    return (output_dir)


def pipeline_level_02(input_dir: str | Path, output_dir: str | Path, instrument: str = 'vis-nir1-nir2') -> str:
    """
    Combines all channels into one FITS file containing one calibrated 
    hyperspectral data cube.

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
            if not stem.endswith('1C'):
                continue
            if file.suffix.lower() != '.fits':
                continue
            if stem[2] in valid_channel_ids:
                files_to_be_combined.append(file)
        except IndexError:
            print(f'Skipping malformed filename: {file.name}')

    combined_fits_file = mergeFits.merge_fits_files(files=files_to_be_combined, output_dir=output_dir)
    print(f'New fits file created: {combined_fits_file}')
    print(f'---------- LEVEL 2B COMPLETED ----------')
    return combined_fits_file
