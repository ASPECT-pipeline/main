from astropy.io import fits
import numpy as np

nir1_path = "test_data/spectra_extraction/NIR1_h.fits"
nir2_path = "test_data/spectra_extraction/NIR2_h.fits"
output_path = "test_data/test_outputs/extracted_fits_spectra.npz"

create_test_spectra = False
test_combined_pixel_spectra = True

def print_npz(file_path):
    try:
        with np.load(file_path) as data:
            print(f"Contents of '{file_path}':")
            for key in data.files:
                print(f"\nKey: {key}")
                print(f"Shape: {data[key].shape}")
                print(f"Data:\n{data[key]}")
    except Exception as e:
        print(f"Error: {e}")

def extract_spectra(file_path):
    try:
        with fits.open(file_path) as hdul:
            if len(hdul) < 2 or not isinstance(hdul[1], fits.ImageHDU):
                raise ValueError("The FITS file does not contain a valid ImageHDU at index 1.")
            
            hdu = hdul[1]
            data = hdu.data
            header = hdu.header

            if data is None or not isinstance(data, np.ndarray):
                raise ValueError("No valid data found in the ImageHDU.")

            if 'WAVELEN' not in header:
                raise KeyError("The FITS file does not contain 'WAVELEN' information in the header.")
            
            wavelengths = np.array([float(w) for w in header['WAVELEN'].split(',')])

            if data.shape[0] != len(wavelengths):
                raise ValueError("The number of wavelength entries does not match the data's first dimension.")

            structured_data = np.zeros((data.shape[1], data.shape[2]), dtype=[('spectra', 'O'), ('wavelengths', 'O')])

            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    structured_data[i, j] = (data[:, i, j], wavelengths)

            return structured_data

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def save_combined_pixel_spectra(nir1_path, nir2_path, output_path, coordinates: tuple = (233, 333)):
    """
    Currently only loads NIR1 and NIR2 spectras from a given coordinate.
    """
    try:
        nir1_data = extract_spectra(nir1_path)
        nir2_data = extract_spectra(nir2_path)

        i, j = coordinates

        if i >= nir1_data.shape[0] or j >= nir1_data.shape[1]:
            raise IndexError("Pixel coordinates out of bounds.")

        nir1_pixel = nir1_data[i, j]
        nir2_pixel = nir2_data[i, j]

        combined_wavelengths = np.concatenate((nir1_pixel['wavelengths'], nir2_pixel['wavelengths']))
        combined_spectra = np.concatenate((nir1_pixel['spectra'], nir2_pixel['spectra'])).astype(float)

        np.savez_compressed(output_path, spectra=combined_spectra, wavelengths=combined_wavelengths)
        print(f"Combined spectral data for pixel {coordinates} saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except IndexError as e:
        print(f"Index Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_test_spectra_npz(wavelengths, spectra, output_path):
    """
    Creates a test NPZ file with given wavelength and spectra lists.
    
    Parameters:
    wavelengths (list or np.ndarray): List or array of wavelengths.
    spectra (list or np.ndarray): List or array of corresponding spectral values.
    output_path (str): Path to save the NPZ file.
    """
    try:
        wavelengths = np.array(wavelengths, dtype=float)
        spectra = np.array(spectra, dtype=float)
        
        if wavelengths.shape[0] != spectra.shape[0]:
            raise ValueError("Wavelengths and spectra must have the same length.")
        
        np.savez_compressed(output_path, wavelengths=wavelengths, spectra=spectra)
        print(f"Test spectra NPZ file saved to: {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")

if create_test_spectra:
    wavelengths = [675, 690, 705, 720, 735, 750, 765, 780, 795, 810, 825, 875, 904, 933, 963,
                992, 1021, 1050, 1079, 1108, 1138, 1167, 1196, 1225, 1254, 1283, 1313, 1342,
                1371, 1400, 1429, 1458, 1488, 1517, 1546, 1575]
    spectra = [8091.0, 8056.91, 8022.83, 7988.74, 7954.66, 7920.57, 7886.49, 7852.4, 
    7818.31, 7784.23, 7750.14, 7716.06, 7681.97, 7647.89, 7613.8, 7579.71, 
    7545.63, 7511.54, 7477.46, 7443.37, 7409.29, 7375.2, 7341.11, 7307.03, 
    7272.94, 7238.86, 7204.77, 7170.69, 7136.6, 7102.51, 7068.43, 7034.34, 
    7000.26, 6966.17, 6932.09, 6898.0]
    output_path = "test_data/test_outputs/test_spectra.npz"

    create_test_spectra_npz(wavelengths, spectra, output_path)
    print_npz(output_path)

if test_combined_pixel_spectra:
    save_combined_pixel_spectra(nir1_path, nir2_path, output_path)
    print_npz(output_path)
