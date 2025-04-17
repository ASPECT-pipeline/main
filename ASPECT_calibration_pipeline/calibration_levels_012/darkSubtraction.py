import os
import numpy as np
from astropy.io import fits

"""
Function for subtracting a dark frame from each 2D image.

Function: darkSubtraction
    Parameters:
        - fitsPath: path to the FITS file
        - outputFolder: path to the folder where the new file is stored
    Description:
        - Iterated over all 2D images inside the data cube subtracting a dark frame from them.
        - Creates a new FITS file with the dark subtracted cube

"""



def dark_subtraction(fits_path: str, output: str):

    # darkFramePath =  os.path.join(os.getcwd(), "outputFiles/dark_VIS_l_1250/dark_VIS_l_1250.fits")

    #Read the FITS file
    with fits.open(fits_path) as hdul:

        # Data from fits file
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)
       
        if channel == 'SWIR':
            return(fits_path)
        
        width = img_header.get('NAXIS1')
        height = img_header.get('NAXIS2')

        # Create new list of HDU's and append the cube to it
        HDUs = []
        HDUs.insert(0, hdul[0])

        # Place holder for the darkframe
        darkFrame = np.zeros((height, width), dtype=hdul[1].data.dtype)

        #To store the calibrated datacube
        new_data_cube = img_HDU.data.copy()

        #loop over the 2D images inside the extension
        for i, image in enumerate(new_data_cube):
            
            #Subtract the dark frame from image
            new_data_cube[i] = image - darkFrame

        # Add the modified image_HDU to the new HDU list
        ImageHDU = fits.ImageHDU(data=new_data_cube, header=img_header)
        HDUs.append(ImageHDU)
        # Add all other extensions except for the original Image HDU
        for i in range(1, len(hdul)):
            if not isinstance(hdul[i], fits.ImageHDU):  # Skip the original Image HDU
                HDUs.append(hdul[i])
        #File name for new fits
        file_name = f'{channel}_1A_Ds.fits'

        # create the new fits file with dark-subtracted images
        hdu_list = fits.HDUList(HDUs)
        fits_file = os.path.join(output, file_name)
        hdu_list = fits.HDUList(HDUs)
        hdu_list.writeto(fits_file, overwrite=True)

    return(fits_file)