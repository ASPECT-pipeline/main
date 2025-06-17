import os
import numpy as np
import pandas as pd
from modules._constants import _project_data, _project_dir, _path_model
from modules.NN_evaluate import evaluate
from modules.utilities_spectra import normalise_spectra, collect_all_models, denoise_spectra, normalise_spectra
from main_level_3 import level3
from mgm import fit, plot
import ast
import h5py
from test_utilities import plot_spectra, show_mgm_figures, test_and_plot_nir_connection, test_and_plot_remove_outliers, test_and_plot_denoise_spectra
from level_3_utilities import nir2_offset_correction, remove_outliers
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt



# 1. Load spectra file
csv_path = os.path.join(_project_data, "600w_exposures_2500-10000-10000_pixel_reflectances(4-pixel_binning).csv")
df = pd.read_csv(csv_path, sep=" ", header=None)

# 2. Extract wavelengths and spectra
wavelengths = df.iloc[:, 0].to_numpy()         # shape (N,)
spectra = df.iloc[:, 1:].to_numpy().T          # shape (16, N)


fits_file = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/simulated_test_3/D1v6v5_simulated_full_datacube.fits')
simulated_cube = os.path.join(os.getcwd(), 'test_data/test_outputs/simulated_full_datacube.fits')

def test_NN():
    
    spectra_normalized = normalise_spectra(
        data=spectra,
        wavelength=wavelengths,
        wvl_norm_nm=1539.0
    )

    model_subdir = "composition/ASPECT-vis-nir1-nir2-1539"   # Composition 
    model_subdir = "taxonomy/ASPECT-vis-nir1-nir2-1539"   # Taxonomy

    model_name = ""
    model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=True)
    predictions = evaluate(model_names, spectra_normalized)

    print(predictions)

test_NN()

def test_preprocessing_and_NN():
    print('inside test_preprocessing')
    level3(fits_file=simulated_cube, instrument=3, test_with_simulated=True)

# test_preprocessing_and_NN()

# python3 ASPECT_calibration_pipeline/level_3/test_level_3.py

def read_model():
    from modules.utilities_spectra import load_keras_model, _path_model
    model_filename = os.path.join(_path_model, 'composition/ASPECT-vis-nir1-nir2-1539-new',  'CNN_ASPECT-vis-nir1-nir2-1539_1110-11-110-111-000_20250612082728.h5')

    model = load_keras_model(model_filename)  # to read the model

        # Read metadata
    with h5py.File(model_filename, "r") as f:
        parameters = ast.literal_eval(f.attrs["params"])  # to convert string "{key: value}" to dict {key: value}
        layer_names = f.attrs["layer_names"]  # names of layers; now only general info visible also using model.summary(). I use this for the ongoing project because I only save weights, not the whole model to keep it more general; you can ignore the layer names here
        md_wavelengths = parameters["wavelengths"] # and many others are there (insttument, normalisation wavelength, hyperparameters)
        print(f'flayer_names: {layer_names}')
        print(f'model wl: {md_wavelengths}')
        print(f'num wl: {len(md_wavelengths)}')

# read_model()

def test_mgm():
    combined = np.column_stack((wavelengths, spectra[0]))
    cleaned = remove_outliers(spectra[0], wavelengths, z_thresh=1)[0]

    # denoise spectra 
    denoised = denoise_spectra(cleaned, wavelengths).flatten()
    combined = np.column_stack((wavelengths, denoised))
    # plot_spectra(spectra[0], wavelengths)
    initGuess = [[0.1, 950, 150], [0.01, 1250, 50]]
    result = fit(combined, initGuess, contLinDeg=0, eps=0.01)
    print(f'result:\n{result}')

    print('plotting...')
    figs = plot(combined, result)
    show_mgm_figures(figs)


# test_mgm()



# test_and_plot_nir_connection(spectra[0], wavelengths)
# test_and_plot_remove_outliers(spectra[0], wavelengths)
# test_and_plot_denoise_spectra(spectra[0], wavelengths)