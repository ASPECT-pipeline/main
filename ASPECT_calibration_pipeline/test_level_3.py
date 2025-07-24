import os
import numpy as np
import pandas as pd
from level_3.modules.collect_data import resave_ASPECT_transmission
import level_3.main_composition as main_composition
import level_3.main_taxonomy as main_taxonomy
from tqdm import tqdm
from level_3.modules.utilities_spectra import normalise_spectra, collect_all_models
from level_3.modules._constants import _project_data
from level_3.modules.NN_evaluate import evaluate
import level_3.mgm as mgm
from level_3.test_utilities import show_mgm_figures
import level_3.level_3_utilities as level_3_utilities

main_c = os.path.join(os.getcwd(), 'ASPECT_calibration_pipeline/level_3/main_composition.py')

mgm_test = os.path.join(os.getcwd(), 'test_data/mgm_test_spectra/DataSet4_nm.txt')
mgm_didymos = os.path.join(os.getcwd(), 'test_data/mgm_test_spectra/didymos_spectra.txt')

"""
python3 ASPECT_calibration_pipeline/test_level_3.py
"""

def generate_aspect_transmission():
    resave_ASPECT_transmission()


def train_composition_models():
    for _ in tqdm(range(10)):
        y_pred = main_composition.pipeline()


def train_taxonomy_models():
    for _ in tqdm(range(10)):
        y_pred = main_taxonomy.pipeline()

# train_taxonomy_models()

def test_NN(model):
    # 1. Load spectra file
    csv_path = os.path.join(_project_data, "600w_exposures_2500-10000-10000_pixel_reflectances(4-pixel_binning).csv")
    df = pd.read_csv(csv_path, sep=" ", header=None)

    # 2. Extract wavelengths and spectra
    wavelengths = df.iloc[7:, 0].to_numpy()         # shape (N,)
    spectra = df.iloc[7:, 1:].to_numpy().T          # shape (16, N)
    spectra_normalized = normalise_spectra(
        data=spectra,
        wavelength=wavelengths,
        wvl_norm_nm=1539
    )
    
    if model == 'T':
        model_subdir = os.path.join('taxonomy', 'ASPECT-vis-nir1-nir2-1539')
    else:
        model_subdir = os.path.join('composition', 'ASPECT-nir1-nir2-1539')
    model_name = ""
    model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=True)

    result = evaluate(model_names, spectra_normalized)
    print(f'{model} analysis:')
    print(f'Length of result {len(result)}')
    print(f'first instance:')
    print(result)

def test_mgm(data):
    # strength, center, std
    initGuess = [[0.3, 0.94, 0.11], [0.5, 1.12, 0.11], [0.3, 1.32, 0.09], [0.2, 2.14, 0.19]]
    with open(data, 'r') as f:
        dat = np.loadtxt(f)
    # RMS, band parameters, continuum parameters, continuum parameters P-values
    result = mgm.fit(dat, initGuess, contLinDeg=0, eps=0.1)
    print(f'mgm results')
    print(result)
    figure = mgm.plot(dat, result)
    show_mgm_figures(figure)

# generate_aspect_transmission()

# train_composition_models()

# test_NN('C')

test_mgm(mgm_test)