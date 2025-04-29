import os
import numpy as np
from astropy.io import fits

"""

This function is to convert the piezo1 setpoints into corresponding wavelength values

"""

def convert_wl(fitsPath: str, output: str) -> str:

    with fits.open(fitsPath) as hdul:

         # Data from fits file
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)
        order = img_header.get('ORDER') # Order used for capturing light
        piezo1_values = img_header.get('PIEZO1').split(",") # capasitance values of setpoint 1
        piezo1_values = [float(value) for value in piezo1_values] # convert the values to numbers

        # Create new list of HDU's and append the cube to it
        HDUs = []
        HDUs.insert(0, hdul[0])

        wavelengths = []
    
        match (channel, order):
            # The correct values for the corretion needed
            case 'VIS', 'h':
                for i in range(0, len(piezo1_values)):
                    wavelength = round(0.0749 * piezo1_values[i] - 786.9)
                    wavelengths.append(wavelength)
            case 'VIS', 'l':
                for i in range(0, len(piezo1_values)):
                    wavelength = round(0.1244 * piezo1_values[i] - 1498.2)
                    wavelengths.append(wavelength)
            case 'NIR1', 'h':
                for i in range(0, len(piezo1_values)):
                    wavelength = round(0.1331 * piezo1_values[i] - 1823.1)
                    wavelengths.append(wavelength)
            case 'NIR1', 'l':
                for i in range(0, len(piezo1_values)):
                    wavelength = round(0.2379 * piezo1_values[i] - 3190.5)
                    wavelengths.append(wavelength)
            case 'NIR2', 'h':
                for i in range(0, len(piezo1_values)):
                    wavelength = round(0.1293 * piezo1_values[i] - 1619.4)
                    wavelengths.append(wavelength)
            case 'NIR2', 'l':
                for i in range(0, len(piezo1_values)):
                    wavelength = round(0.2366 * piezo1_values[i] - 2925.8)
                    wavelengths.append(wavelength)
            case 'SWIR', '':
                for i in range(0, len(piezo1_values)):
                    wavelength = round(0.2869 * piezo1_values[i] - 3847.2)
                    wavelengths.append(wavelength)

        #Create a new fits file with same primary header and new imageHDU 
        #Stroe the file into the same folder with the input image
        data_copy = img_HDU.data.copy()
        
        if channel == 'SWIR':
            data_HDU = fits.BinTableHDU(data=data_copy, header=img_header)
        else:
            data_HDU = fits.ImageHDU(data=data_copy, header=img_header)
        
        data_HDU.header['WAVELEN'] = ','.join(map(str, wavelengths)) # Add the wavelengths to the header

        HDUs.append(data_HDU)
        file_name = f'{channel}_1A_wl.fits' # Adjust the file name here
        fits_file = os.path.join(output, file_name)
        hdu_list = fits.HDUList(HDUs)
        hdu_list.writeto(fits_file, overwrite=True)

    return(fits_file)