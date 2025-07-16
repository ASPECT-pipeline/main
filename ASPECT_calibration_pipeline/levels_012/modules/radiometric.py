import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList

"""
Function for converting the pixel values into scientific units.

    Description:
        - Iterated over all 2D images inside the data cube multiplying it with a coefficient.
        - Creates a new FITS file with the calibrated data
"""


def radiometric_calibration(hdul: HDUList) -> HDUList:
    """
    Function for converting the pixel values into scientific units.

     Parmeters:
        hdul (HDUList): The HDU list of the FITS file that is modified
    
    Returns:
        The modified HDU list of the FITS file
    """

    # Data from fits file
    img_HDU = hdul[1] # Contains the image cube (or swir readings)
    img_header = img_HDU.header # Image HDU header
    img_data = img_HDU.data
    channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)

    coefficient = 1 # Temporary coefficient for radiometric calibration

    if channel == 'SWIR':
        bin_table_data = img_data.copy() #To store the calibrated BinTable
        for col_name in bin_table_data.names:
            bin_table_data[col_name] = (bin_table_data[col_name] * coefficient).astype(np.float64)

        new_bin_table_hdu = fits.BinTableHDU(data=bin_table_data, header=hdul[1].header)
        hdul[1] = new_bin_table_hdu

    else:
        new_data_cube = img_data.astype(np.float64, copy=True)
        #loop over the 2D images inside the extension
        for i, image in enumerate(img_data):
            new_data_cube[i] = (image * coefficient).astype(np.float64) # multiply the image with the coefficient 
        
        image_hdu = fits.ImageHDU(data=new_data_cube, header=img_header)
        hdul[1] = image_hdu

    
    return hdul