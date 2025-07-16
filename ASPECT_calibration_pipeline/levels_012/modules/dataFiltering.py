import numpy as np
from astropy.io import fits
import levels_012.modules.utilities

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

    return combined, all_wl

# numerical eps
_num_eps = 1e-5  # num_eps of float32 is 1e-7

def remove_outliers(y: np.ndarray, x: np.ndarray | None = None,
                    z_thresh: float = 1.5, num_eps: float = _num_eps) -> np.ndarray | tuple[np.ndarray, ...]:
    inds_to_remove = utilities.find_outliers(y=y, x=x, z_thresh=z_thresh, num_eps=num_eps)

    if x is None:
        return np.delete(y, inds_to_remove)
    
    # Create a mask of where outliers are
    mask = np.zeros_like(y, dtype=bool)
    mask[inds_to_remove] = True

    # Interpolate the outliers
    interpolated_y = utilities.interpolate_mask_1d(y.copy(), mask=mask)

    # return np.delete(y, inds_to_remove), np.delete(x, inds_to_remove)
    return interpolated_y.flatten(), x

def denoise_spectra(data: np.ndarray, wavelength: np.ndarray, sigma_nm: float | None = 7.) -> np.ndarray:
    if sigma_nm is None:
        return data

    if sigma_nm <= 0.:
        raise ValueError(f'"sigma_nm" must be positive float but equals {sigma_nm}')

    if np.ndim(data) == 1:
        data = np.reshape(data, (1, len(data)))

    return utilities.denoise_array(data, sigma=sigma_nm, x=wavelength)


def filter_spectra(path: str, test: bool, test_data, ):

    if test:
        combined_spectra, wavelengths = test_data
    else:
        combined_spectra, wavelengths = filter_asteroid_spectra(path)

    coords, spectras = zip(*combined_spectra)
    coords = np.array(coords) 
    spectras = np.array(spectras)
    smooth_spectra = spectras.copy()

    print(f'spectras: {len(spectras)}')

    for i, spectra in enumerate(spectras):
        nir1_wavelengths = wavelengths[10:20]
        nir2_wavelengths = wavelengths[20:]
        nir1_spectra = spectra[10:20]
        nir2_spectra = spectra[20:]
        nir2_spectra_corrected, offset = utilities.nir2_offset_correction(nir1_wavelengths, nir1_spectra, nir2_wavelengths, nir2_spectra, test=True)
        print(f'nir2_spectra_corrected: ')
        print(nir2_spectra_corrected)
        connected = np.concatenate((spectra[:20], nir2_spectra_corrected))
        outliers_removed = remove_outliers(y=connected, x=wavelengths, z_thresh=1.1)
        denoised = denoise_spectra(data=outliers_removed[0], wavelength=wavelengths).flatten()
        smooth_spectra[i] = denoised

    #Combine coords and the spectra
    combined = list(zip(coords, smooth_spectra))

    return combined
