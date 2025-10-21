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
        missphas = header.get('MISSPHAS')
        if missphas == 'SIMULATED':
            return (f'AS{channel_id}_CCDTEMP', 'N/A', f"{channel} detector temp [K] ('N/A' [C])") 
        original = header.get(f'AS{channel_id}_CCDTEMP')
        if original and original != 'UNK':
            c, k = utilities.det_temp_conversion(float(original),channel)
            return(f'AS{channel_id}_CCDTEMP', f'{k}', f'{channel} detector temp [K] ({c} [C])')
    except Exception as e:
        print(f"[WARNING] {channel_id} CCDTEMP conversion failed: {e}")
    return (f'AS{channel_id}_CCDTEMP', 'UNK', f"{channel} detector temp [K] ('UNK' [C])") 


def convert_fpi_temp(header, channel, channel_id, suffix):
    try:
        missphas = header.get('MISSPHAS')
        if missphas == 'SIMULATED':
            return (f'AS{channel_id}_{suffix}', 'N/A', f"{channel} {suffix} [K] ('N/A' [C])") 
        original = header.get(f'AS{channel_id}_{suffix}')
        if original and original != 'UNK':
            if suffix == 'FPI_TEMP1':
                c, k = utilities.fpi_temp_conversion(float(original), channel, 1)
            else:
                c, k = utilities.fpi_temp_conversion(float(original), channel, 2)
            return(f'AS{channel_id}_{suffix}', f'{k}', f'{channel} {suffix} [K] ({c} [C])')
    except Exception as e:
        print(f"[WARNING] {channel} {suffix} conversion failed: {e}")
    return (f'AS{channel_id}_{suffix}', 'UNK', f"{channel} {suffix} [K] ('UNK' [C])") 

def convert_exposure_times(header, channel, channel_id):
    try:
        missphas = header.get('MISSPHAS')
        if missphas == 'SIMULATED':
            raise Exception('Simulated')
        
        task_number = int(header.get(f'AS{channel_id}_TASK_NUMBER'))
        exposures = []

        for i in range(0, task_number):
            num = f'{i:03d}'
            exposures.append(header.get(f'AS{channel_id}_TASK_{num}').split(' ')[3])
        
        values = [float(value) for value in exposures] # convert the values to numbers
        exp_dict = utilities.exposure_conversion(values,channel,task_number)

    except Exception as e:
        print(f'[WARNING] {channel} exposure conversion failed: {e}')
        unk_dict = {}
        for i in range(0, task_number):
            num = f'{i:03d}' # e.g. 1 -> 001
            unk_dict[f'AS{channel_id}_WL_{num}'] = ('UNK', '')
            return unk_dict

    return exp_dict

def convert_wavelengths(header, channel, channel_id, order):
    try:
        missphas = header.get('MISSPHAS')
        simulated = missphas == 'SIMULATED'

        task_number = int(header.get(f'AS{channel_id}_TASK_NUMBER'))

        setpoint1 = []

        for i in range(0, task_number):
            num = f'{i:03d}'
            setpoint1.append(header.get(f'AS{channel_id}_TASK_{num}').split(' ')[0])

        values = [float(value) for value in setpoint1] # convert the values to numbers

        # if len(values) != len(frames):
        #     raise Exception('The number of piezo setpoint 1 values missmatch with the number of frames.')
        wl_dict = utilities.wavelength_conversion(channel, order, values, task_number, simulated) # get correct wl by frame

    except Exception as e:
        print(f'[WARNING] {channel} wavelenght conversion failed: {e}')
        unk_dict = {}
        for i in range(0, task_number):
            num = f'{i:03d}' # e.g. 1 -> 001
            unk_dict[f'AS{channel_id}_WL_{num}'] = ('UNK', '')
            return unk_dict

    return wl_dict

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
                channel = channel_map[channel_id] # Channel (Vis, NIR1, NIR2, SWIR)
                order = header.get(f'AS{channel_id}_ORDER')

                # To run conversions
                keys = [
                    convert_detector_temp(header, channel, channel_id),
                    convert_fpi_temp(header, channel, channel_id, 'FPI_TEMP1'),
                    convert_fpi_temp(header, channel, channel_id, 'FPI_TEMP2'),
                ]

                for key, value, comment in keys:
                    card_length = 8 + len(value) + len(comment) + 7
                    if card_length <= 80:
                        header[key] = (value, comment)
                    else:
                        header[key] = (value, '')
            
            channel = header.get(f'ASP_CHANNELS')
            channel_id = reverse_channel_map.get(channel)
            order = header.get(f'AS{channel_id}_ORDER')

            # Convert wavelengths
            wl_dict = convert_wavelengths(header, channel, channel_id, order)
            sorted_wl_dict = dict(sorted(wl_dict.items(), key=lambda x: int(x[0][-3:]), reverse=True))
            for key, (value, comment) in sorted_wl_dict.items():
                task_idx = header.index(f'AS{channel_id}_TASK_NUMBER')
                task_number = int(header.get(f'AS{channel_id}_TASK_NUMBER'))
                task_idx += task_number
                header.insert(task_idx, (f'HIERARCH {key}', value, comment), after=True)

            # Conver exposure
            exp_dict = convert_exposure_times(header, channel, channel_id)
            sorted_exp_dict = dict(sorted(exp_dict.items(), key=lambda x: int(x[0][-9:-6]), reverse=True))

            for key, (value, comment) in sorted_exp_dict.items():
                task_idx = header.index(f'AS{channel_id}_TASK_NUMBER')
                task_number = int(header.get(f'AS{channel_id}_TASK_NUMBER'))
                task_idx += task_number
                header.insert(task_idx, (f'HIERARCH {key}', value, comment), after=True)


        except Exception as e:
            print(f"[WARNING] Error while calibrating primary HDU header: {e}")
            

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