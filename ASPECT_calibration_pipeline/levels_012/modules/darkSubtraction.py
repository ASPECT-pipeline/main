import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList


def dark_subtraction(hdul: HDUList) -> HDUList:

    """
    Function for subtracting a dark frame from each 2D image.

    Parmeters:
        hdul (HDUList): The HDU list of the FITS file that is modified
    
    Returns:
        The modified HDU list of the FITS file
    """

    # Data from fits file
    print(f'hdul length: {len(hdul)}')
    img_HDU = hdul[1] # Contains the image cube (or swir readings)
    img_data = img_HDU.data
    img_header = img_HDU.header # Image HDU header
    channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)

    if channel == 'SWIR':
        return hdul
    else:

        width = img_header.get('NAXIS1')
        height = img_header.get('NAXIS2')

        # Place holder for the darkframe
        darkFrame = np.zeros((height, width), dtype=hdul[1].data.dtype)

        # To store the calibrated datacube
        new_data_cube = img_HDU.data.astype(np.float64, copy=True)

        # Loop over the 2D images inside the extension
        for i, image in enumerate(img_data):
            # Subtract the dark frame from image
            new_data_cube[i] = image - darkFrame

        # Add the modified image_HDU to the new HDU list
        image_hdu = fits.ImageHDU(data=new_data_cube, header=img_header)
        hdul[1] = image_hdu

        return hdul