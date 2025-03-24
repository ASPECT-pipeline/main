import numpy as np

"""
NIR2 offset correction calculates an offset for all NIR2 spectra values which
is used to allign NIR1 and NIR2, forming a continuous spectrum and correcting
a possible error gap in measurements caused by the hardware and its physical environment.

The offset is calculated by fitting a linear regression to the last 3 values of NIR1 and
the first 3 values of NIR2, and then calculating the difference between the linear regressions
at the wavelength 1225. The offset is then added to all NIR2 spectra values.
"""

test_spectra = "test_data/test_outputs/test_spectra_with_duplicate_1225.npz"

test_nir2_correction = False

def nir2_offset_correction(
		nir1_wavelengths: np.ndarray,
		nir1_spectra: np.ndarray,
		nir2_wavelengths: np.ndarray,
		nir2_spectra: np.ndarray,
		overlap_wavelength: int = 1225,
		test: bool = False
	):
	"""
	Corrects NIR2 spectra by aligning the overlap region with NIR1 spectra using linear regression.

	Parameters:
		nir1_wavelengths (np.ndarray): Wavelengths of NIR1 wavelengths.
		nir1_spectra (np.ndarray): NIR1 spectra.
		nir2_wavelengths (np.ndarray): Wavelengths of NIR2 wavelengths.
		nir2_spectra (np.ndarray): NIR2 spectra.
		overlap_wavelength (int): Wavelength at which to align the spectra.
		test (bool): If True, prints the calculated values.
	
	Returns:
		corrected_nir2 (np.ndarray): Corrected NIR2 spectra.
		offset (float): Offset value.
	"""

	if test:
		print(f'Initial:\n{nir1_wavelengths}\n{nir2_wavelengths}\n{nir1_spectra}\n{nir2_spectra}')

	coeffs_nir1 = np.polyfit(nir1_wavelengths[-3:], nir1_spectra[-3:], 1)
	coeffs_nir2 = np.polyfit(nir2_wavelengths[:3], nir2_spectra[:3], 1)

	f_nir1 = np.polyval(coeffs_nir1, overlap_wavelength)
	f_nir2 = np.polyval(coeffs_nir2, overlap_wavelength)
	
	if test:
		print(f'f_nir1: {f_nir1}, f_nir2: {f_nir2}')

	offset = f_nir1 - f_nir2
	corrected_nir2 = nir2_spectra + offset

	if test:
		print("Offset:", offset)
		print("Corrected NIR2 spectra:\n", corrected_nir2)
		
	return corrected_nir2, offset

if test_nir2_correction:
	with np.load(test_spectra) as data:
		spectra = data['spectra']
		wavelengths = data['wavelengths']

	nir1_wavelengths = wavelengths[:24]
	nir2_wavelengths = wavelengths[24:]
	nir1_spectra = spectra[:24]
	nir2_spectra = spectra[24:]

	nir2_spectra_corrected, offset = nir2_offset_correction(nir1_wavelengths, nir1_spectra, nir2_wavelengths, nir2_spectra, test=True)

