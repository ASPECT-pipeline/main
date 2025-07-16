import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList



def flat_field_calibration(hdul: HDUList) -> HDUList:
    """
    Applies flatfield correction to each 2D frame on the channel
    
    Parmeters:
        hdul (HDUList): The HDU list of the FITS file that is modified
    
    Returns:
        The modified HDU list of the FITS file
    """


    # Data from fits file
    primary_hdu = hdul[0]
    primary_header = primary_hdu.header
    img_HDU = hdul[1] # Contains the image cube (or swir readings)
    img_data = img_HDU.data
    img_header = img_HDU.header # Image HDU header
    channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)
    
    # This step is not done to SWIR images
    if channel == 'SWIR':
        return hdul
    else:
        width = img_header.get('NAXIS1')
        height = img_header.get('NAXIS2')

        # Place holder flatfield array
        # This should be replaced wiht the correct flatfield used for the imager.
        flatField = np.ones((height, width), dtype=hdul[1].data.dtype)

        # To store the calibrated datacube
        new_data_cube = img_HDU.data.astype(np.float64, copy=True)

        # loop over the 2D images inside the extension
        for i, image in enumerate(img_data):
            # Divide the image with the flatfield 
            new_data_cube[i] = image / flatField
        
    
        image_HDU = fits.ImageHDU(data=new_data_cube, header=img_header)
        hdul[1] = image_HDU

        return hdul