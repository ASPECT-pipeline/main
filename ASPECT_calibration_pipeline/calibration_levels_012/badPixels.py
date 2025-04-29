import os
import numpy as np
from astropy.io import fits


"""
Function for removing bad pixels from each 2D image inside a cube given a mask.

Function removeBadPixels:
    Parameters:
        - fitsPath: path to the FITS file
        - outputFolder: path to the folder where the new file is stored
    Description:
        - Open the FITS file and extract important headers
        - Iterate over each 2D slice of the cube
            - iterate over each pixel and identify bad pixels with the mask
            - change the value of bad pixels into mean of neigbours
            - if no 'good' neighbours put the pixel on hold
        - Iterate over the pictures on hold and determine valeus for them
        - Create a new FITS file and return the path to it
"""

#################################################
#  This function assumes that the fits file has
#  one primary HDU and one Imge HDU. Image HDU 
#  contains the data cube.
################################################## 
 
def remove_bad_pixels(fits_path: str, output: str) -> str:

    #Open the FITS file
    with fits.open(fits_path) as hdul:

        # Data from fits file
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_header = img_HDU.header # Image HDU header
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)

        #Skip This for SWIR
        if channel == 'SWIR':
            return(fits_path)
        
        width = img_header.get('NAXIS1')
        height = img_header.get('NAXIS2')

        # Create new list of HDU's and append the primary HDU, new image HDU and other extensions
        HDUs = []
        HDUs.insert(0, hdul[0])

        #Place holder mask array
        mask = np.zeros((height, width), dtype=hdul[1].data.dtype)

        #Update the mask to remove the already corrected values
        updatingMask = mask.copy()

        #To store the bad pixels that did not have 'good' neighbours
        onHold = []

        #To store the calibrated datacube
        newDataCube = hdul[1].data.copy()

        #Accessing the data cube
        dataCube = hdul[1].data

        #Iterate over all 2D images inside the cube
        for im, image in enumerate(dataCube):
            data = image.copy()
            #Loop over all pixels indetifying bad ones
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
                    #This should not happen under any circumstanses
                    print(f"WARNING: STILL No valid neighbors for data[{i}][{j}]")
            newDataCube[im] = data

       
        ImageHDU = fits.ImageHDU(data=newDataCube, header=img_header)
        HDUs.append(ImageHDU)
        # Add all other extensions except for the original Image HDU
        for i in range(1, len(hdul)):
            if not isinstance(hdul[i], fits.ImageHDU):  # Skip the original Image HDU
                HDUs.append(hdul[i])
        file_name = f'{channel}_1A_Bp.fits'

        hdu_list = fits.HDUList(HDUs)
        fits_file = os.path.join(output, file_name)
        hdu_list = fits.HDUList(HDUs)
        hdu_list.writeto(fits_file, overwrite=True)

    return(fits_file)