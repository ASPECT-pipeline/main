import os
import numpy as np
from astropy.io import fits

"""
Function for applying the flat field calibration to each 2D image

Function: flatFieldCalibration
    Parameters:
        - fitsPath: path to the FITS file
        - outputFolder: path to the folder where the new file is stored
    Description:
        - Iterated over all 2D images inside the data cube dividing by flatfield.
        - Creates a new FITS file with the flat field calibrated cube
"""


def flat_field_calibration(fits_path: str, output: str):

     # Open the fits file
    with fits.open(fits_path) as hdul:

        # Data from fits file
        primary_HDU = hdul[0]
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)
        
        # This step is not done to SWIR images
        if channel == 'SWIR':
            return fits_path
        
        width = img_header.get('NAXIS1')
        height = img_header.get('NAXIS2')

        # Create new list of HDU's and append the primary HDU, new image HDU and other extensions
        HDUs = []
        HDUs.insert(0, hdul[0])

        #Place holder flatfield array
        flatField = np.ones((height, width), dtype=hdul[1].data.dtype)

        #To store the calibrated datacube
        newDataCube = img_HDU.data.copy()

        #loop over the 2D images inside the extension
        for i, image in enumerate(newDataCube):
            #Devide the image with the flatfield 
            newDataCube[i] = image / flatField
        
       
        ImageHDU = fits.ImageHDU(data=newDataCube, header=img_header)
        HDUs.append(ImageHDU)
        # Add all other extensions except for the original Image HDU
        for i in range(1, len(hdul)):
            if not isinstance(hdul[i], fits.ImageHDU):  # Skip the original Image HDU
                HDUs.append(hdul[i])
        
        #File name for new fits
        file_name = f'{channel}_1A_Ff.fits'

        # create the new fits file with dark-subtracted images
        hdu_list = fits.HDUList(HDUs)
        fits_file = os.path.join(output, file_name)
        hdu_list = fits.HDUList(HDUs)
        hdu_list.writeto(fits_file, overwrite=True)

    return(fits_file)