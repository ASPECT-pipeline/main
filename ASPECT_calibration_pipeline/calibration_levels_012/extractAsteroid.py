import numpy as np
from astropy.io import fits
import utilities

"""
Function for extracting the asteroid spectra from the aligned and resampled asteroid images.

Function extract_asteroid:
    Parameters:
        - path: path to the FITS file
        - mask_index: index of the frame that the mask is made of
    Description:
        - Open the FITS file and extracts the mask that tells which pixels contains information about the asteroid.
        - Extract the coordinates from the mask (coords) and corresponding pixel values from each frame (spectra)
        - zips coords with the spectra
        - to unzip the coords and the spectra to separate lists:
            coords, spectra = zip(*combined)
            coords = np.array(coords) 
            spectra = np.array(spectra)

"""

def extract_asteroid(path, mask_index=0):
    with fits.open(path) as hdul:
        image_cube = hdul[1].data
        image = image_cube[mask_index]

        asteroid_mask = utilities.asteroid_mask(image)

        #Store the coordinates of the image where mask has value of non 0
        coords = np.argwhere(asteroid_mask != 0)

        #Extract the corresponding spectra for coords
        spectra = np.array([image_cube[:, y, x] for y, x in coords])

        #Combine coords and the spectra
        combined = list(zip(coords, spectra))

        return combined


