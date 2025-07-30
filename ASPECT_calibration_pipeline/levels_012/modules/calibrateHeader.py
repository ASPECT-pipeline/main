import os
from pathlib import Path
import numpy as np
from astropy.io import fits
import levels_012.modules.utilities as utilities

"""

This function is to convert the piezo1 setpoints into corresponding wavelength values

"""

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
        primary_hdu = hdul[0]
        primary_header = primary_hdu.header
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)
        order = img_header.get('ORDER') # Order used for capturing light
        try:
            piezo1_values = img_header.get('SP1').split(",") # capasitance values of setpoint 1
            piezo1_values = [float(value) for value in piezo1_values] # convert the values to numbers
        except Exception as e:
            print(f"[WARNING] No valid piezo setpoint values {e}")
            piezo1_values = None

        # Convert exposure time from DNs to seconds
        exposure = primary_header['EXPOSURE']
        if exposure and exposure != 'UNK':
            exposure_list = list(map(int, exposure.split(',')))
            exposures_in_s = [utilities.exposure_conversion(x, channel) for x in exposure_list]
            exposures_str = ','.join(f"{x:.6f}" for x in exposures_in_s)
            primary_header['EXPOSURE'] = (exposures_str, "Exposure time [s]")
        else:
            print(f"[WARNING] no valid exposure value '{exposure}' found in primary header")
            exposures_str = 'UNK'
        # Convert detector reading from DN to Kelvins and Celsius
        det_temp = primary_header['CCDTEMP']
        if det_temp and det_temp != 'UNK':
            det_temp = float(det_temp)
            c, k = utilities.det_temp_conversion(det_temp, channel)
            c = round(c, 2)
            k = round(k, 2)
            if channel in ('NIR1','NIR2'):
                primary_header['CCDTEMP'] = ('UNK', f"Detector temp [K] ('UNK' [C])")
            else:
                primary_header['CCDTEMP'] = (f"{k:.2f}", f'Detector temp [K] ({f"{c:.2f}"} [C])')
        else:
            print(f"[WARNING] not a valid CCDTEMP '{det_temp}' found in primary header")
        

        # Create new list of HDU's and append the cube to it
        HDUs = []
        HDUs.insert(0, hdul[0])

        wavelengths = utilities.wavelength_conversion(channel, order, piezo1_values)

        #Create a new fits file with same primary header and new imageHDU 
        #Stroe the file into the same folder with the input image
        data_copy = img_HDU.data.copy()
        
        if channel == 'SWIR':
            data_HDU = fits.BinTableHDU(data=data_copy, header=img_header)
        else:
            data_HDU = fits.ImageHDU(data=data_copy, header=img_header)
        
        data_HDU.header['WAVELEN'] = wavelengths
        data_HDU.header['EXPOSURE'] = (exposures_str, "Exposure time [s]")
        HDUs.append(data_HDU)

        hdu_list = fits.HDUList(HDUs)
        stem = fits_path.stem
        suffix = fits_path.suffix
        new_calibration_level = '1A'
        file_name = stem[:25] + new_calibration_level + suffix
        primary_header = hdu_list[0].header
        primary_header['FILENAME'] = file_name
        primary_header['PROCLEVL'] = new_calibration_level
        fits_file = os.path.join(output_dir, file_name)
        hdu_list.writeto(fits_file, overwrite=True)

    return(fits_file)