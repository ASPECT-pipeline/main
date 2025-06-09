from astropy.io import fits
import numpy as np
from modules.utilities_spectra import (extract_asteroid, nir2_offset_correction, remove_outliers,
                                        denoise_spectra, normalise_spectra, collect_all_models)
from modules.NN_evaluate import evaluate
import os

"""
The level 3 of the ASPECT data processing pipeline consist of 3 sublevels:
    - Level 3A: Data filtering; Connecting segments, removing outliers, smoothing
    - Level 3B: Spectral parameters; Albedo, slope, band depths
    - Level 3C Composition analysis; Asteroid taxonomy, mineral content 

Parameters:
    instrument:int
        1 = ASPECT vis nir1 nir2
        2 = ASPECT vis nir1 nir2 swir
        3 = ASPECT nir1 nir2
        4 = ASPECT nir1 nir2 swir
    
        models:str
        C = composition
        T = taxonomy
        CT = composition and taxonomy

"""

def level3(fits_file:str, instrument:int = 1, models:str = 'C', nir_overlap:int = 1231, z_thresh:int = 1):

    """
    Execute the steps 3A, 3B, 3C
    """
    # Start by extracting the asteroid spectras from the data cube

    # Extract relevant metadata
    with fits.open(fits_file) as hdul:
        img_HDU = hdul[1]
        img_header = img_HDU.header
        img_cube = img_HDU.data

        # Number of frames
        vis_num = img_header.get('V_NUM')
        nir1_num = img_header.get('N1_NUM')
        nir2_num = img_header.get('N2_NUM')
        swir_num = img_header.get('S_NUM')

        # Wavelenghts
        vis_wl = np.array([int(x.strip()) for x in img_header.get('V_WL').split(",") if x.strip()], dtype=int)
        nir1_wl = np.array([int(x.strip()) for x in img_header.get('N1_WL').split(",") if x.strip()], dtype=int)
        nir2_wl = np.array([int(x.strip()) for x in img_header.get('N2_WL').split(",") if x.strip()], dtype=int)
        swir_wl = np.array([int(x.strip()) for x in img_header.get('S_WL').split(",") if x.strip()], dtype=int)
        all_wl = np.concatenate([vis_wl, nir1_wl, nir2_wl, swir_wl])

        match instrument:
            case 1: 
                instrument_wl = np.concatenate([vis_wl, nir1_wl, nir2_wl])
                start_idx = 0
                end_idx = vis_num + nir1_num + nir2_num
                first_nir = vis_num
                norm_wl = 1539
                stem = 'ASPECT-vis-nir1-nir2-1539'
            case 2:
                instrument_wl = np.concatenate([vis_wl, nir1_wl, nir2_wl, swir_wl])
                start_idx = 0
                end_idx = None
                first_nir = vis_num
                norm_wl = 2348
                stem = 'ASPECT-vis-nir1-nir2-swir-2348'
            case 3: 
                instrument_wl = np.concatenate([nir1_wl, nir2_wl])
                start_idx = vis_num
                end_idx = vis_num + nir1_num + nir2_num
                first_nir = 0
                norm_wl = 1539
                stem = 'ASPECT-nir1-nir2-1539'
            case 4:
                instrument_wl = np.concatenate([nir1_wl, nir2_wl, swir_wl])
                start_idx = vis_num
                end_idx = None
                first_nir = 0
                norm_wl = 2348
                stem = 'ASPECT-vis-nir1-nir2-swir-2348'

        print(f'vis: {vis_wl}')
        print(f'nir1: {nir1_wl}')
        print(f'nir2: {nir2_wl}')
        print(f'swir: {swir_wl}')
        print(f'instrument {instrument}: {instrument_wl}')

    combined = extract_asteroid(img_cube, mask_index=0, start_idx=start_idx, end_idx=end_idx)
    
    coords, spectras = zip(*combined)
    coords = np.array(coords) 
    spectras = np.array(spectras)

    # new array for filttered spectras 
    filtered_spectras = np.empty_like(spectras)

    # Filter each spectra extracted and append it to the filtered_spectras array

    for i, spectra in enumerate(spectras):

        # #3A
        nir1_spectra = spectra[first_nir : first_nir + nir1_num]
        nir2_spectra = spectra[first_nir + nir1_num : first_nir + nir1_num + nir2_num]

        nir2_offset_correction_result = nir2_offset_correction(
            nir1_wavelengths=nir1_wl,
            nir1_spectra=nir1_spectra,
            nir2_wavelengths=nir2_wl,
            nir2_spectra=nir2_spectra,
            overlap_wavelength=nir_overlap
        )

        # print(nir2_offset_correction_result)
        connected = np.concatenate(
            [spectra[:first_nir + nir1_num], nir2_offset_correction_result[0]] +
            ([spectra[first_nir + nir1_num + nir2_num:]] if spectra[first_nir + nir1_num + nir2_num:].size > 0 else [])
        )

        # remove outliers
        cleaned = remove_outliers(connected, instrument_wl, z_thresh=z_thresh)[0]

        # denoise spectra 
        denoised = denoise_spectra(cleaned, instrument_wl).flatten()

        filtered_spectras[i] = denoised
    
    spectra_normalized = normalise_spectra(
        data=filtered_spectras,
        wavelength=instrument_wl,
        wvl_norm_nm=norm_wl
    )

    if models in ('C', 'CT'):
        model_subdir = os.path.join('composition', stem)
        model_name = ""
        model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=True)

        composition = evaluate(model_name, spectra_normalized)

    if models in ('T', 'CT'):
        model_subdir = os.path.join('taxonomy', stem)
        model_name = ""
        model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=True)

        taxonomy = evaluate(model_name, spectra_normalized)


level3(os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/simulated_test_3/D1v6v5_simulated_full_datacube.fits'))

# python3 ASPECT_calibration_pipeline/level_3/main_level_3.py