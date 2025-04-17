import numpy as np
from astropy.io import fits
import utilities

"""
Function for extracting the asteroid spectra from the aligned and resampled asteroid images.

Function extract_asteroid:
    Parameters:
        - path: path to the FITS file
    Description:
        - Open the FITS file and extract the asteroid spectrum.
        - Remove outliers from the spectrum
        - Apply Gaussian smoothing to the spectrum
        - zips coords with the spectra
        - to unzip the coords and the spectra to separate lists:
            coords, spectra = zip(*combined)
            coords = np.array(coords) 
            spectra = np.array(spectra)

"""

def filter_asteroid_spectra(path: str):
    with fits.open(path) as hdul:
        image_hdu = hdul[1]
        image_header = image_hdu.header
        image_cube = image_hdu.data
    
    # Wavelengths 
    vis_wl = [int(x) for x in image_header['V_WL'].split(',')]
    nir_str = image_header['N1_WL'] + ',' + image_header['N2_WL']
    nir_wl = [int(x) for x in nir_str.split(',')]
    all_wl = vis_wl + nir_wl

    # Extract asteroid from data cube (coords and spectra)
    asteroid = utilities.extract_asteroid(image_cube, mask_index=0)
    coords, spectras = zip(*asteroid)
    coords = np.array(coords) 
    spectras = np.array(spectras)

    # Remove outliers for each spectra (pixel coordinate)
    cleaned_spectras = spectras.copy()
    for i, spectra in enumerate(spectras):
        spectra_2d = spectra.reshape(1, -1)
        cleaned_spectras[i] = utilities.remove_outliers_2d(spectra_2d, kernel_size=5, n_std=1.6)
        cleaned_spectras[i] = utilities.denoise_spectra(cleaned_spectras[i], all_wl)

    #Combine coords and the spectra
    combined = list(zip(coords, cleaned_spectras))

    return combined




