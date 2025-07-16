import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList


 
def replace_bad_pixels(hdul: HDUList) -> HDUList:
    """
    Function for removing bad pixels from each 2D image inside a cube given a mask.

    Description:
        - Iterate over each 2D slice of the cube
            - iterate over each pixel and identify bad pixels with the mask
            - change the value of bad pixels into mean of neigbours
            - if no 'good' neighbours put the pixel on hold
        - Iterate over the pictures on hold and determine valeus for them
        - Create a new FITS file and return the path to it

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

    #Skip This for SWIR
    if channel == 'SWIR':
        return hdul
    else:
        
        width = img_header.get('NAXIS1')
        height = img_header.get('NAXIS2')

        # Place holder mask array
        mask = np.zeros((height, width))

        # Update the mask to remove the already corrected values
        updatingMask = mask.copy()

        # To store the bad pixels that did not have 'good' neighbours
        onHold = []

        # To store the calibrated datacube
        new_data_cube = img_data.astype(np.float64, copy=True)

        # Iterate over all 2D images inside the cube
        for im, image in enumerate(img_data):
            data = image.copy()
            # Loop over all pixels indetifying bad ones
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                        if mask[i, j] == 1:
                            #get neighbours from data (3-8 neighbours)
                            neighbours = data[max(0, i-1):min(data.shape[0], i+2),
                                            max(0, j-1):min(data.shape[1], j+2)]
                            #get corresponding neighbours from mask
                            neighboursMask = updatingMask[max(0, i-1):min(data.shape[0], i+2),
                                                max(0, j-1):min(data.shape[1], j+2)]
                            #Store 'good' neighbours
                            validNeighbours = []
                            #loop over neighbours and corresponding mask values
                            for neighbourVal, maskVal in zip(neighbours.flatten(), neighboursMask.flatten()):
                                if maskVal == 0:
                                    validNeighbours.append(neighbourVal)
                            #calculate the mean of 'good' neighbors
                            if validNeighbours:
                                #calculate the mean and change value in updatingmask
                                data[i,j] = np.mean(validNeighbours)
                                updatingMask[i][j] = 0
                            else:
                                #put this pixel on hold to be processed later
                                onHold.append((i,j))
            #Loop over onhold
            for i, j in reversed(onHold):
                #Loopin in reverse to ensure that the remaining bad pixels have good neighbours
                neighbours = data[max(0, i-1):min(data.shape[0], i+2),
                                    max(0, j-1):min(data.shape[1], j+2)]
                neighboursMask = updatingMask[max(0, i-1):min(data.shape[0], i+2),
                                          max(0, j-1):min(data.shape[1], j+2)]
                validNeighbours = []

                for neighbourVal, maskVal in zip(neighbours.flatten(), neighboursMask.flatten()):
                    if maskVal == 0:
                        validNeighbours.append(neighbourVal)
                #calculate the mean of 'good' neighbors
                if validNeighbours:
                    data[i,j] = np.mean(validNeighbours)
                    updatingMask[i][j] = 0
                else:
                    #This should not happen
                    print(f"WARNING: STILL No valid neighbors for data[{i}][{j}]")
            new_data_cube[im] = data

       
        image_hdu = fits.ImageHDU(data=new_data_cube, header=img_header)
        hdul[1] = image_hdu

        return hdul