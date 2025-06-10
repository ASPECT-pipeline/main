import os
import numpy as np
import pandas as pd
from modules._constants import _project_data, _project_dir
from modules.NN_evaluate import evaluate
from modules.utilities_spectra import normalise_spectra, collect_all_models
from main_level_3 import level3



# 1. Load spectra file
csv_path = os.path.join(_project_data, "600w_exposures_2500-10000-10000_pixel_reflectances(4-pixel_binning).csv")
df = pd.read_csv(csv_path, sep=" ", header=None)

# 2. Extract wavelengths and spectra
wavelengths = df.iloc[:, 0].to_numpy()         # shape (N,)
spectra = df.iloc[1:, 1:].to_numpy().T          # shape (16, N)

fits_file = os.path.join(os.getcwd(), 'test_data/levels_012_test/test_output/simulated_test_3/D1v6v5_simulated_full_datacube.fits')

def test_NN():
    spectra_normalized = normalise_spectra(
        data=spectra,
        wavelength=wavelengths,
        wvl_norm_nm=1539.0
    )

    model_subdir = "composition/ASPECT-vis-nir1-nir2-1539"   # Composition
    # model_subdir = "taxonomy/ASPECT-vis-nir1-nir2-1539"   # Taxonomy

    model_name = ""
    model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=True)

    predictions = evaluate(model_names, spectra_normalized)

    print(predictions)


def test_preprocessing_and_NN():
    print('inside test_preprocessing')
    level3(spectra=spectra, fits_file=fits_file, test=True)

test_preprocessing_and_NN()

# python3 ASPECT_calibration_pipeline/level_3/test_level_3.py