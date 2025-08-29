import numpy as np
from pathlib import Path
from astropy.io.fits import HDUList
from scipy.ndimage import gaussian_filter1d
from config import reverse_channel_map, calibration_directory

def load_ssi_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (wavelength, Fsun 1AU Wm2 per nm) from SSI csv file.
    """
    data = np.genfromtxt(csv_path, delimiter=",", comments="#", dtype=float)
    if data.ndim == 1:  # single data row
        data = data[None, :]
    if data.shape[1] < 2:
        raise ValueError("Expected at least two columns: wavelength, value.")
    wl = data[:, 0].astype(float)
    vals = data[:, 1].astype(float)
    return wl, vals

def gaussian_convolution(
        array: np.ndarray,
        x: np.ndarray,
        fwhm: int,
    ) -> np.ndarray:
    step = x[1] - x[0]
    fwhm_to_sigma = 1. / np.sqrt(8. * np.log(2.))
    array_averaged = gaussian_filter1d(array, sigma=(fwhm_to_sigma * fwhm) / step, mode="nearest")
    return array_averaged

def reflectance_calibration(
        hdul: HDUList,
        fwhm_nm: float = 30.0,
    ) -> HDUList:
    """
    The image is calibrated by pixel linear transformations. The pixels represent I/f reflectance units.

    Parameters: 
        hdul (HDUList) : hdu list containing the Primary HDU data (image) to be converted to I/f 
    
    Returns: 
        Reflectance unit converted data inside the same hdul 
    """

    primary_hdu = hdul[0]
    primary_header = primary_hdu.header
    data = primary_hdu.data

    if primary_header.get('MISSPHAS') == 'SIMULATED':
        return hdul

    channel = primary_header.get('CHANNELS')
    sun_dist = primary_header.get('SOLAR_D')

    if sun_dist in (None, 'UNK'):
        print(f"[WARNING] Solar distance missing from '{channel}' header. Skipping I/F.'")
        return hdul
    else:
        sun_dist = float(sun_dist)
    
    if channel == 'SWIR':
        fwhm_nm = 40.0
    
    channel_id = reverse_channel_map.get(channel)
    wavelengths = primary_header.get(f'{channel_id}_WL').split(',')

    if wavelengths[0] in (None, 'UNK', 'N/A'):
        print(f"[WARNING] Wavelengths is not valid for '{channel}' header '{wavelengths}'. Skipping I/F.'")
        return hdul
    
    calib_dir = Path(calibration_directory)
    ssi_csv = calib_dir / 'ssi_yearly_avg_e2024_c20250221.csv'
    wl_nm, ssi_vals = load_ssi_csv(ssi_csv)

    ssi_gaussian = gaussian_convolution(ssi_vals, wl_nm, fwhm_nm)

    for i, frame in enumerate(data):
        wl = float(wavelengths[i])
        ssi_index = int(np.searchsorted(wl_nm, wl))
        f_au = ssi_gaussian[ssi_index]
        IF_frame = np.pi * frame * (sun_dist**2) / f_au
    
        data[i] = IF_frame

    primary_hdu.data = data
    return hdul





