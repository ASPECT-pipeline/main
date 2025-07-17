from astropy.io import fits
import numpy as np
from level_3.modules.utilities_spectra import ( denoise_spectra, normalise_spectra, collect_all_models)
from level_3.level_3_utilities import (extract_asteroid, nir2_offset_correction, remove_outliers, get_wavelengths, validate_wl, validate_instrument)
from level_3.modules.NN_evaluate import evaluate
from level_3.test_utilities import get_reflectances, plot_4_spectra, plot_spectra
import matplotlib
matplotlib.use('MacOSX')

import os

"""
The level 3 of the ASPECT data processing pipeline consist of 3 sublevels:
    - Level 3A: Data filtering; Connecting segments, removing outliers, smoothing
    - Level 3B: Spectral parameters; Albedo, slope, band depths
    - Level 3C Composition analysis; Asteroid taxonomy, mineral content 

Parameters:
    instrument:int
        'vis-nir1-nir2' = ASPECT vis nir1 nir2
        'vis-nir1-nir2'-swir = ASPECT vis nir1 nir2 swir
        'nir1-nir2' = ASPECT nir1 nir2
        'nir1-nir2-swir' = ASPECT nir1 nir2 swir
    
    models:str
        C = composition
        T = taxonomy
        CT = composition and taxonomy
    
    nir_overlap:int
        1231 nm is the default wavelength used to align nir1 and nir2 segments
    
    z_thresh:int
        Threshold for determining outliers from the spectra. Low values gives more outliers. Default 1

"""

def level3( fits_file:str, instrument:str = 'vis-nir1-nir2', data_filtering:bool = True, models:str = 'C', nir_overlap:int = 1231, z_thresh:int = 1, test_with_simulated:bool = False):

    """
    Execute the steps 3A, 3B, 3C
    """
    # Start by extracting the asteroid spectras from the data cube

    # Extract relevant metadata
    with fits.open(fits_file) as hdul:
        primary_hdu = hdul[0]
        img_HDU = hdul[1]
        img_header = img_HDU.header
        img_cube = img_HDU.data

        # print(repr(primary_hdu.header))
        # print(repr(img_header))


        # Wavelenghts
        # vis_wl = np.array([int(x.strip()) for x in img_header.get('V_WL').split(",") if x.strip()], dtype=int)
        # nir1_wl = np.array([int(x.strip()) for x in img_header.get('N1_WL').split(",") if x.strip()], dtype=int)
        # nir2_wl = np.array([int(x.strip()) for x in img_header.get('N2_WL').split(",") if x.strip()], dtype=int)
        # swir_wl = np.array([int(x.strip()) for x in img_header.get('S_WL').split(",") if x.strip()], dtype=int)
        # all_wl = np.concatenate([vis_wl, nir1_wl, nir2_wl, swir_wl])

    validate_instrument(instrument)
        
    wavelengths = get_wavelengths(img_header)
    validate_wl(wavelengths, instrument)
    
    all_wl = np.sort(np.concatenate([wavelengths[ch] for ch in wavelengths.keys()]))

    match instrument:
        case 'vis-nir1-nir2': 
            instrument_wl = np.concatenate([wavelengths['VIS'], wavelengths['NIR1'], wavelengths['NIR2']])
            norm_wl = 1539
            start_idx = 0
            end_idx = int(np.where(all_wl == wavelengths['NIR2'][-1])[0]) + 1
            stem = 'ASPECT-vis-nir1-nir2-1539'
        case 'vis-nir1-nir2-swir':
            instrument_wl = np.concatenate([wavelengths['VIS'], wavelengths['NIR1'], wavelengths['NIR2'], wavelengths['SWIR'] ])
            norm_wl = 2348
            start_idx = 0
            end_idx = len(all_wl)
            stem = 'ASPECT-vis-nir1-nir2-swir-2348'
        case 'nir1-nir2': 
            instrument_wl = np.concatenate([wavelengths['NIR1'], wavelengths['NIR2']])
            norm_wl = 1539
            start_idx = int(np.where(all_wl == wavelengths['NIR1'][0])[0])
            end_idx = int(np.where(all_wl == wavelengths['NIR2'][-1])[0]) + 1
            stem = 'ASPECT-nir1-nir2-1539'
        case 'nir1-nir2-swir':
            instrument_wl = np.concatenate([wavelengths['NIR1'], wavelengths['NIR2'], wavelengths['SWIR']])
            norm_wl = 2348
            start_idx = int(np.where(all_wl == wavelengths['NIR1'][0])[0])
            end_idx = len(all_wl)
            stem = 'ASPECT-nir1-nir2-swir-2348'

    # print(f'Instrument wl: {instrument_wl}, lenght: {len(instrument_wl)}')
    # print(f'stem: {stem}')

    print(f'Extracting asteroid spectras')
    combined = extract_asteroid(img_cube, mask_index=0)

    coords, spectras = zip(*combined)
    coords = np.array(coords) 
    spectras = np.array(spectras)
    print(f'{len(spectras)} spectras extracted')

    spectra_0 = spectras[0]
    spectra_1 = spectras[50000]
    spectra_0[:11] /= 4096
    spectra_0[11:] /= 16384
    spectra_1[:11] /= 4096
    spectra_1[11:] /= 16384

    first_nir1_idx = int(np.where(all_wl == wavelengths['NIR1'][0])[0])
    nir1_len = len(wavelengths['NIR1'])
    nir2_len = len(wavelengths['NIR2'])

    if data_filtering:
        cleaned_spectras = []
        print(f'Applying data filtering')
        for i, spectra in enumerate(spectras):
            #3A
            spectra[:11] /= 4096
            spectra[11:] /= 16384
            nir1_spectra = spectra[first_nir1_idx : first_nir1_idx + nir1_len]
            nir2_spectra = spectra[first_nir1_idx + nir1_len : first_nir1_idx + nir1_len + nir2_len]
            nir2_offset_correction_result = nir2_offset_correction(
                nir1_wavelengths=wavelengths['NIR1'],
                nir1_spectra=nir1_spectra,
                nir2_wavelengths=wavelengths['NIR2'],
                nir2_spectra=nir2_spectra,
                overlap_wavelength=nir_overlap
            )

            connected = np.concatenate(
                [spectra[:first_nir1_idx + nir1_len - 1], nir2_offset_correction_result[0]] +
                ([spectra[first_nir1_idx + nir1_len + nir2_len:]] if spectra[first_nir1_idx + nir1_len + nir2_len:].size > 0 else [])
            )
            unique = np.array(list(dict.fromkeys(all_wl)))
            print(len(unique))
            print(unique)
            # plot_spectra(connected, unique)
            # remove outliers
            cleaned = remove_outliers(connected, unique, z_thresh=z_thresh)[0]

            # denoise spectra 
            denoised = denoise_spectra(cleaned, unique).flatten()

            # filtered_spectras[i] = denoised
            try:
                plot_4_spectra(spectra, connected, cleaned, denoised, instrument_wl, ['original', 'connected', 'outliers', 'denoised', 'level 3A'])
            except KeyboardInterrupt:
                print('Stopped')
            cleaned_spectras[i] = denoised

    return
    if test_with_simulated:
        spectras = spectras[50000:50005]
        coords = coords[50000:50005]
        for i, s in enumerate(spectras):
            spectras[i] = get_reflectances(s, all_wl[:-swir_num])
        spectras = np.atleast_2d(spectras)
    # new array for filttered spectras 
    # filtered_spectras = np.zeros_like(spectras)

    # If data_filtering = True, perform 3 pre filteting steps for each spectra: connecting nir1 and nir2 segments, remove outliers, and denoise the spectra.
    if data_filtering:
        for i, spectra in enumerate(spectras):
            #3A
            nir1_spectra = spectra[first_nir : first_nir + nir1_num]
            nir2_spectra = spectra[first_nir + nir1_num : first_nir + nir1_num + nir2_num]
            nir2_offset_correction_result = nir2_offset_correction(
                nir1_wavelengths=nir1_wl,
                nir1_spectra=nir1_spectra,
                nir2_wavelengths=nir2_wl,
                nir2_spectra=nir2_spectra,
                overlap_wavelength=nir_overlap
            )

            connected = np.concatenate(
                [spectra[:first_nir + nir1_num], nir2_offset_correction_result[0]] +
                ([spectra[first_nir + nir1_num + nir2_num:]] if spectra[first_nir + nir1_num + nir2_num:].size > 0 else [])
            )

            # remove outliers
            cleaned = remove_outliers(connected, instrument_wl, z_thresh=z_thresh)[0]

            # denoise spectra 
            denoised = denoise_spectra(cleaned, instrument_wl).flatten()

            # filtered_spectras[i] = denoised
            plot_4_spectra(spectra, connected, cleaned, denoised, instrument_wl, ['original', 'connected', 'outliers', 'denoised', 'level 3A'])
            spectras[i] = denoised

    
    spectra_normalized = normalise_spectra(
        data=spectras,
        wavelength=instrument_wl,
        wvl_norm_nm=norm_wl
    )

    if models in ('C', 'CT'):
        model_subdir = os.path.join('composition', stem)
        model_name = ""
        model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=True)

        composition = evaluate(model_names, spectra_normalized)

    if models in ('T', 'CT'):
        model_subdir = os.path.join('taxonomy', stem)
        model_name = ""
        model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=True)

        taxonomy = evaluate(model_names, spectra_normalized)


# level3(os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/simulated_test_3/D1v6v5_simulated_full_datacube.fits'), test_with_simulated=True)

# python3 ASPECT_calibration_pipeline/level_3/main_level_3.py