import os
import numpy as np
from astropy.io import fits
from pathlib import Path


 
def remove_bad_pixels(fits_path: str | Path, output_dir: str | Path) -> str:
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
        fits_path: Path to the FITS file.
        output_dir: Path to the folder where the new fits file will be stored.

    Returns:
        path to the created fits file.
    """
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)

    # Open the FITS file
    with fits.open(fits_path) as hdul:

        # Data from fits file
        primary_hdu = hdul[0]
        primary_header = primary_hdu.header
        img_HDU = hdul[1] # Contains the image cube (or swir readings)
        img_header = img_HDU.header # Image HDU header
        img_data = img_HDU.data
        channel = img_header.get('CHANNEL') # Channel (VIS, NIR1, NIR2, SWIR)

        #Skip This for SWIR
        if channel == 'SWIR':
            return(fits_path)
        
        width = img_header.get('NAXIS1')
        height = img_header.get('NAXIS2')

        # Create new list of HDU's and append the primary HDU, new image HDU and other extensions
        HDUs = []
        HDUs.insert(0, primary_hdu)

        # Place holder mask array
        mask = np.zeros((height, width), dtype=hdul[1].data.dtype)

        # Update the mask to remove the already corrected values
        updatingMask = mask.copy()

        # To store the bad pixels that did not have 'good' neighbours
        onHold = []

        # To store the calibrated datacube
        newDataCube = img_data.copy()

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
            newDataCube[im] = data

       
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
        fits_file = os.path.join(output_dir, file_name)
        hdu_list.writeto(fits_file, overwrite=True)

    return(fits_file)