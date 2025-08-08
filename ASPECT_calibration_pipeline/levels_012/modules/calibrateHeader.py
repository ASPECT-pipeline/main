import os
from pathlib import Path
import numpy as np
from astropy.io import fits
import levels_012.modules.utilities as utilities
from config import reverse_channel_map, channel_map

"""

Helper functions

"""

def convert_detector_temp(header, channel, channel_id):
        try:
            original = header.get(f'{channel_id}_CCDTMP')
            if original and original != 'UNK':
                c, k = utilities.det_temp_conversion(float(original),channel)
                return(f'{channel_id}_CCDTMP', f'{k}', f'Detector temp [K] ({c} [C])')
        except Exception as e:
            print(f"[WARNING] {channel_id} CCDTMP conversion failed: {e}")
        return (f'{channel_id}_CCDTMP', 'UNK', "Detector temp [K] ('UNK' [C])") 


def convert_fpi_temp(header, channel, channel_id, suffix):
    try:
        original = header.get(f'{channel_id}_{suffix}')
        if original and original != 'UNK':
            if suffix == 'FPI1':
                c, k = utilities.fpi_temp_conversion(float(original), channel, 1)
            else:
                c, k = utilities.fpi_temp_conversion(float(original), channel, 2)
            return(f'{channel_id}_{suffix}', f'{k}', f'{suffix} temp [K] ({c} [C])')
    except Exception as e:
        print(f"[WARNING] {channel} {suffix} conversion failed: {e}")
    return (f'{channel_id}_{suffix}', 'UNK', f"{suffix} temp [K] ('UNK' [C])") 

def convert_exposure_times(header, channel, channel_id):
    try:
        original = header.get(f'{channel_id}_EXPOS')
        if original and original != 'UNK':
            exposure_list = list(map(int, original.split(',')))
            exposures_in_s = [utilities.exposure_conversion(x, channel) for x in exposure_list]
            exposures_str = ','.join(f"{x:.6f}" for x in exposures_in_s)
            return(f'{channel_id}_EXPOS', exposures_str, 'Exposure time(s) [s]')
    except Exception as e:
        print(f'[WARNING] {channel} exposure time conversion failed: {e}')
    return (f'{channel_id}_EXPOS', 'UNK', 'Exposure time(s) [s]')

def convert_waverlengths(header, channel, channel_id, order):
    try:
        setpoint1 = header.get(f'{channel_id}_SP1')
        if setpoint1 and setpoint1 != 'UNK':
            values = setpoint1.split(",") # capasitance values of setpoint 1
            values = [float(value) for value in values] # convert the values to numbers
            wavelengths = utilities.wavelength_conversion(channel, order, values)
            return(f'{channel_id}_WL', str(wavelengths), f'[nm]')
    except Exception as e:
        print(f'[WARNING] {channel} wavelenght conversion failed: {e}')
    return(f'{channel_id}_WL', 'UNK', f'[nm]')

def calibrate_header(fits_path: str | Path, output_dir: str | Path) -> str:
    """
    Parmeters:
        fits_path: Path to the FITS file.
        output: Path to the folder where the new fits file will be stored.
    
    Returns:
        path to the created fits file.
    """

    fits_path = Path(fits_path)
    output_dir = Path(output_dir)

    with fits.open(fits_path) as hdul:

        # Data from fits file
        new_hdul = fits.HDUList([hdu.copy() for hdu in hdul])
        hdu = new_hdul[0]
        header = hdu.header

        try:
            channel_ids = list(channel_map.keys())

            for channel_id in channel_ids:  
                channel = channel_map[channel_id] # Channel (VIS, NIR1, NIR2, SWIR)
                order = header.get(f'{channel_id}_ORDER')

                # To run conversions
                keys = [
                    convert_detector_temp(header, channel, channel_id),
                    convert_fpi_temp(header, channel, channel_id, 'FPI1'),
                    convert_fpi_temp(header, channel, channel_id, 'FPI2'),
                ]

                for key, value, comment in keys:
                    card_length = len(key) + len(value) + len(comment) + 4
                    if card_length <= 80:
                        header[key] = (value, comment)
                    else:
                        header[key] = (value, '')
            
            channel = header.get('CHANNELS')
            channel_id = reverse_channel_map.get(channel)
            # To run conversions
            keys = [
                convert_exposure_times(header, channel, channel_id),
                convert_waverlengths(header, channel, channel_id, order)
            ]

            for key, value, comment in keys:
                card_length = len(key) + len(value) + len(comment) + 4
                if card_length <= 80:
                    header[key] = (value, comment)
                else:
                    header[key] = (value, '')

        except Exception as e:
            print(f"[WARNING] Error while calibrating IMAGE HDU header: {e}")
            

        stem = fits_path.stem
        suffix = fits_path.suffix
        new_calibration_level = '1A'
        file_name = stem[:25] + new_calibration_level + suffix
        primary_header = header
        primary_header['FILENAME'] = file_name
        primary_header['PROCLEVL'] = new_calibration_level
        fits_file = os.path.join(output_dir, file_name)
        new_hdul.writeto(fits_file, overwrite=True)

    return(fits_file)