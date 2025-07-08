import os
import numpy as np
from astropy.io import fits
from pathlib import Path

"""
Function for converting the pixel values into scientific units.

    Description:
        - Iterated over all 2D images inside the data cube multiplying it with a coefficient.
        - Creates a new FITS file with the calibrated data
"""


def radiometric_calibration(fits_path: str, output_dir: str) -> str:
    """
    Function for converting the pixel values into scientific units.

    Parmeters:
        fits_path: Path to the FITS file.
        output_dir: Path to the folder where the new fits file will be stored.

    Returns:
        path to the created fits file.
    """
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)

    # Open the fits file
    with fits.open(fits_path) as hdul:

        # Data from fits file
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)

        # Create new list of HDU's and append the primary HDU, new image HDU and other extensions
        HDUs = []
        HDUs.insert(0, hdul[0])

    
        coefficient = 1 # Temporary coefficient for radiometric calibration

        if channel == 'SWIR':
            bin_table_data = hdul[1].data.copy() #To store the calibrated BinTable
            for col_name in bin_table_data.names:
                bin_table_data[col_name] = (bin_table_data[col_name] * coefficient).astype(np.float64)

            new_bin_table_hdu = fits.BinTableHDU(data=bin_table_data, header=hdul[1].header)
            HDUs.append(new_bin_table_hdu)

        else:
            newDataCube = hdul[1].data.copy()#To store the calibrated datacube
            #loop over the 2D images inside the extension
            for i, image in enumerate(newDataCube):
                newDataCube[i] = (image * coefficient).astype(np.float64) # multiply the image with the coefficient 
            
            ImageHDU = fits.ImageHDU(data=newDataCube, header=img_header)
            HDUs.append(ImageHDU)

        # Add all other extensions except for the original Image HDU
        for i in range(1, len(hdul)):
            if not isinstance(hdul[i], fits.ImageHDU):  # Skip the original Image HDU
                HDUs.append(hdul[i])

        hdu_list = fits.HDUList(HDUs)
        # File name for new fits
        stem = fits_path.stem
        suffix = fits_path.suffix
        new_calibration_level = '1B'
        file_name = stem[:25] + new_calibration_level + suffix
        primary_header = hdu_list[0].header
        primary_header['FILENAME'] = file_name
        primary_header['PROCLEVL'] = new_calibration_level
        # Create the new fits file with dark-subtracted images
        fits_file = Path(output_dir) / file_name
        hdu_list.writeto(fits_file, overwrite=True)

    return(fits_file)