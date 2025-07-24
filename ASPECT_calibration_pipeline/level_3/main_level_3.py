from astropy.io import fits
import numpy as np
from level_3.modules.utilities_spectra import ( denoise_spectra, normalise_spectra, collect_all_models)
from level_3.level_3_utilities import (extract_asteroid, nir2_offset_correction, remove_outliers, get_wavelengths, validate_wl, validate_instrument)
from level_3.modules.NN_evaluate import evaluate
from level_3.test_utilities import get_reflectances, plot_4_spectra, plot_spectra
from level_3.mgm import fit, plot
from level_3.test_utilities import show_mgm_figures, visualise_composition
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
        C = Composition
        T = Taxonomy
        M = MGM
    
    nir_overlap:int
        1231 nm is the default wavelength used to align nir1 and nir2 segments
    
    z_thresh:int
        Threshold for determining outliers from the spectra. Low values gives more outliers. Default 1

"""

def level3( fits_file:str, instrument:str = 'vis-nir1-nir2', data_filtering:bool = False, models:str = 'C', nir_overlap:int = 1231, z_thresh:int = 1, initGuess: list[list[float]] = [[0.1, 950, 150], [0.01, 1250, 50]], test_with_simulated:bool = False):

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

    # Extract the spectrums from data cube
    print(f'Extracting asteroid spectras')
    combined = extract_asteroid(img_cube, mask_index=0)

    coords, spectras = zip(*combined)
    coords = np.array(coords) 
    spectras = np.array(spectras)
    print(f'{len(spectras)} spectras extracted')

    # Select the range based on instrument selection
    selected_wl = all_wl[start_idx:end_idx]
    first_nir1_idx = int(np.where(selected_wl == wavelengths['NIR1'][0])[0])
    nir1_len = len(wavelengths['NIR1'])
    nir2_len = len(wavelengths['NIR2'])

    denoised_spectras = []
    if data_filtering:
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
                [spectra[start_idx:first_nir1_idx + nir1_len], nir2_offset_correction_result[0][1:]] +
                ([spectra[first_nir1_idx + nir1_len + nir2_len:end_idx]])
            )

            selected_wl = np.array(list(dict.fromkeys(selected_wl)))

            # Remove outliers
            cleaned = remove_outliers(connected, selected_wl, z_thresh=z_thresh)[0]

            # denoise spectra 
            denoised = denoise_spectra(cleaned, selected_wl).flatten()

            denoised_spectras.append(denoised)
            try:
                if i % 10000 == 0:
                    plot_4_spectra(np.delete(spectra[start_idx:end_idx], first_nir1_idx + nir1_len), connected, cleaned, denoised, selected_wl, ['original', 'connected', 'outliers', 'denoised', 'level 3A'])
            except KeyboardInterrupt:
                print('Stopped')
    else:
        # Only select the range remove the nir1-nir2 ovelap
        for i, spectra in enumerate(spectras):
            #3A
            spectra[:11] /= 4096
            spectra[11:] /= 16384
            selected_wl = np.array(list(dict.fromkeys(selected_wl)))
            denoised = np.delete(spectra[start_idx:end_idx], first_nir1_idx + nir1_len)
            denoised_spectras.append(denoised)

    denoised_spectras = np.array(denoised_spectras)
    # denoised_spectras = denoised_spectras[45000:45005]

    print(f'Normalising spectras at {norm_wl}nm')
    spectra_normalized = normalise_spectra(
        data=denoised_spectras,
        wavelength=selected_wl,
        wvl_norm_nm=norm_wl
    )

    model = [str(m).upper() for m in models.split('-')] # Analysis to be executed

    if 'M' in model:
        print('MGM analysis')
        mgm_results = []
        for i, spectra in enumerate(denoised_spectras):
            combined = np.column_stack((selected_wl, spectra))
            result = fit(combined, initGuess, contLinDeg=0, eps=0.01)
            mgm_results.append(result)
            if i % 10 == 0:
                print(f'{i} / {len(denoised_spectras)}')
            #     print(i)
            #     result = fit(combined, initGuess, contLinDeg=0, eps=0.01)
            #     mgm_results.append(result)
            #     figs = plot(combined, result)
            #     show_mgm_figures(figs)

        print('printing from results')
        print(len(mgm_results))
        print(mgm_results[0])
        # combined = np.column_stack((selected_wl, denoised_spectras[50000]))
        # figs = plot(combined,mgm_results[0])
        # show_mgm_figures(figs)
    
    if 'C' in model:
        print('Composition analysis with Neural Network')
        model_subdir = os.path.join('composition', stem)
        model_name = ""
        model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=True)
        composition = evaluate(model_names, spectra_normalized)

        print('Composition analysis:')
        print(f'Length of compositions {len(composition)}')
        print(composition[45000:45005])

        visualise_composition(img_cube[0],composition, coords)
    if 'T' in model:
        model_subdir = os.path.join('taxonomy', stem)
        model_name = ""
        model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=True)

        taxonomy = evaluate(model_names, spectra_normalized)
        print('Taxonomy analysis:')
        print(f'Length of taxonomy {len(taxonomy)}')
        print(composition[0])
    print()

# level3(os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/simulated_test_3/D1v6v5_simulated_full_datacube.fits'), test_with_simulated=True)

# python3 ASPECT_calibration_pipeline/level_3/main_level_3.py