import numpy as np
from pathlib import Path
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Level 3 imports
from level_3.modules.utilities_spectra import (normalise_spectra, collect_all_models, load_xlsx)
from level_3.modules.NN_evaluate import evaluate
from level_3.modules.NN_data import load_transmission
from level_3.modules.NN_config_taxonomy import classes
from level_3.level_3_utilities import spectra_filtering

def get_aspect_wl():
    wl_dict = {
        'AS0' : [675., 690., 705., 720., 735., 750., 765., 780., 795., 810., 825.,],
        'AS1' : [875., 904.20738725, 933.40538359, 962.41926832, 991.59052354, 1020.78790557, 1050., 1079.21475545, 1108.41876944, 1137.40366510, 1166.57594038, 1195.77918273, 1225.],
        'AS2' : [1225., 1254.22427930, 1283.43620514, 1312.38308545, 1341.55680112, 1370.76774434, 1400., 1429.23686478, 1458.45946423, 1487.35519223, 1516.53104350, 1545.75237632, 1575.],
        'AS3' : None
    }
    return wl_dict


def test_filtering_and_nn(npz_path: str | Path, filtering: bool = True, analysis: bool = False):
    data = np.load(npz_path, allow_pickle=True)
    spectra, coords = data["spectra"], data["coords"]
    data.close()
    # _, _, wvl_central = load_transmission("ASPECT-vis-nir1-nir2")
    all_wl = get_aspect_wl()
    # matplotlib.use('MacOSX')
    # mean = np.mean(spectra, axis=0)
    # plt.figure()
    # plt.plot(all_wl, mean, linewidth=0.8)
    # plt.xlabel('Wavelength (nm)', fontsize=12)
    # plt.ylabel('Reflectance', fontsize=12)
    # plt.title('Mean before filtering')
    # plt.tight_layout()
    # plt.show()
    if filtering:
        denoised_spectra = spectra_filtering(spectras=spectra, wavelengths=all_wl)

    if analysis:
        _, _, wvl_central = load_transmission("ASPECT-vis-nir1-nir2")
        model_subdir, model_name = "taxonomy/ASPECT-vis-nir1-nir2-1546", "CNN_ASPECT-vis-nir1-nir2-1546_9-1"
        spectra = normalise_spectra(denoised_spectra, wavelength=wvl_central, wvl_norm_nm=float(model_name.split("_")[1].split("-")[-1]))

        model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=False)
        predictions = evaluate(model_names, spectra, proportiontocut=0.2, subfolder_model=model_subdir)
        taxonomy = {k: predictions[:, index] for k, index in classes.items()}

        model_subdir, model_name = "composition/ASPECT-vis-nir1-nir2-1546", "CNN_ASPECT-vis-nir1-nir2-1546_1110-11-110-111-000"
        model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=False)
        predictions = evaluate(model_names, spectra, proportiontocut=0.2, subfolder_model=model_subdir)
        quantities = {"OL": 0, "OPX": 1, "CPX": 2, "Fa": 3, "Fo": 4, "Fs (OPX)": 5, "En (OPX)": 6, "Fs (CPX)": 7, "En (CPX)": 8, "Wo (CPX)": 9}
        composition = {k: predictions[:, index] for k, index in quantities.items()}

        df = pd.DataFrame(taxonomy | composition)

        print(df.mean())


def test_training_nn():
    tax_output_setup = {
        "use_unknown_class": True  # Add extra "unknown" class for weird spectra
    }
    # Re-interpolate input data to different resolutions (see reinterpolate_data in load_data.py)
    tax_grid_setup = {
        "instrument": "ASPECT-vis-nir1-nir2",  # "HS-H", "ASPECT-vis-nir1-nir2",
        "interpolate_to": None,  # "full", "Itokawa", "Eros"; only if "instrument" is None
        # used when "instrument" is None and interpolate_to is unknown
        "wvl_grid": safe_arange(820., 1600., 20., endpoint=True),
        # used when interpolate_to is unknown
        "wvl_norm": "adaptive"  # float or "adaptive" (there are no non-normalised data for training)
    }
    comp_output_setup = {
        "minerals": np.array([True,  # olivine
                            True,  # orthopyroxene
                            True,  # clinopyroxene
                            False]),  # plagioclase
        "endmembers": [[True, True],  # Fa, Fo; OL
                    [True, True, False],  # Fs, En, Wo; OPX
                    [True, True, True],  # Fs, En, Wo; CPX
                    [False, False, False]]  # An, Ab, Or; PLG
    }
    # Re-interpolate input data to different resolutions (see reinterpolate_data in load_data.py)
    comp_grid_setup = {
        "instrument": "ASPECT-vis-nir1-nir2",  # "HS-H", "ASPECT-vis-nir1-nir2",
        "interpolate_to": None,  # "full", "Itokawa", "Eros"; only if "instrument" is None
        # used when "instrument" is None and interpolate_to is unknown
        "wvl_grid": safe_arange(820., 1600., 20., endpoint=True),
        # used when interpolate_to is unknown
        "wvl_norm": "adaptive"  # float, None, or "adaptive" (don't use None unless you have a good reason for that)
    }

def resave_ASPECT_transmission() -> None:
    print("Re-saving ASPECT's transmission...")
    """
    filename = path.join(_path_data, "ASPECT", "REF_MEAS_upd_wl.xlsx")
    wavelengths = load_xlsx(filename, sheet_name="600W, 10000|2500, LO", skiprows=2)["wl"].to_numpy()
    wvl_vis = wavelengths[borders[0] + 1:borders[1]]
    wvl_nir1 = wavelengths[borders[1] + 1:borders[2]]
    wvl_nir2 = wavelengths[borders[2] + 1:]
    wvl_swir = safe_arange(1650., 2500., step=30., endpoint=True)
    """
    _path_data = os.path.join(os.getcwd(), 'ASPECT_calibration_pipeline/level_3/datasets')
    filename = os.path.join(_path_data, "ASPECT", "ASPECT_Default_wl.xlsx")
    wavelengths = load_xlsx(filename, sheet_name="ASPECT Default wl", skiprows=0)["wl"].to_numpy()
    borders = np.where(~np.isfinite(wavelengths))[0]
    wvl_vis = wavelengths[:borders[0]]
    wvl_nir1 = wavelengths[borders[0] + 1:borders[1]]
    wvl_nir2 = wavelengths[borders[1] + 1:borders[2]]
    wvl_nir2 = wvl_nir2[1:]  # the first point is used to remove jump in spectrum
    wvl_swir = wavelengths[borders[2] + 1:]
    print(wvl_vis)
    print(wvl_nir1)
    print(wvl_nir2)
    print(wvl_swir)

    fwhm_to_sigma = 1. / np.sqrt(8. * np.log(2.))
    sigma_vis = np.polyval(np.polyfit([np.min(wvl_vis), np.max(wvl_vis)], (20., 20.), 1), wvl_vis) * fwhm_to_sigma
    sigma_nir1 = np.polyval(np.polyfit([np.min(wvl_nir1), np.max(wvl_nir2)], (40., 27.), 1), wvl_nir1) * fwhm_to_sigma
    sigma_nir2 = np.polyval(np.polyfit([np.min(wvl_nir1), np.max(wvl_nir2)], (40., 27.), 1), wvl_nir2) * fwhm_to_sigma
    sigma_swir = np.polyval(np.polyfit([np.min(wvl_swir), np.max(wvl_swir)], (45., 45.), 1), wvl_swir) * fwhm_to_sigma
    
    wvl_transmission = safe_arange(450., 2600., 5., endpoint=True).reshape(-1, 1)
    vis = np.transpose(norm.pdf(wvl_transmission, loc=wvl_vis, scale=sigma_vis))
    wvl_central_vis = np.array([my_argmax(wvl_transmission.ravel(), transm, n_points=2, fit_method="ransac") for transm in vis])
    nir1 = np.transpose(norm.pdf(wvl_transmission, loc=wvl_nir1, scale=sigma_nir1))
    wvl_central_nir1 = np.array([my_argmax(wvl_transmission.ravel(), transm, n_points=2, fit_method="ransac") for transm in nir1])
    nir2 = np.transpose(norm.pdf(wvl_transmission, loc=wvl_nir2, scale=sigma_nir2))
    wvl_central_nir2 = np.array([my_argmax(wvl_transmission.ravel(), transm, n_points=2, fit_method="ransac") for transm in nir2])
    swir = np.transpose(norm.pdf(wvl_transmission, loc=wvl_swir, scale=sigma_swir))
    wvl_central_swir = np.array([my_argmax(wvl_transmission.ravel(), transm, n_points=2, fit_method="ransac") for transm in swir])
    transmissions = {"wavelengths": wvl_transmission.ravel(),
                     "vis": {"transmissions": vis, "central_wavelengths": wvl_central_vis.ravel()},
                     "nir1": {"transmissions": nir1, "central_wavelengths": wvl_central_nir1.ravel()},
                     "nir2": {"transmissions": nir2, "central_wavelengths": wvl_central_nir2.ravel()},
                     "swir": {"transmissions": swir, "central_wavelengths": wvl_central_swir.ravel()}
                     }
    # filename = path.join(_path_data, "ASPECT", f"ASPECT{_sep_out}transmission.npz")
    # check_dir(filename)
    # with open(filename, "wb") as f:
    #     np.savez_compressed(f, **transmissions)

""" 
Python3 ASPECT_calibration_pipeline/test_functions.py
"""

### Path library

_results = (Path(__file__).parent.parent / 'pipeline_results').resolve()
_test_data = (Path(__file__).parent.parent / 'test_data').resolve()


# Spectra
spectra_npz = _results / 'ASPECT_simulated' / 'D1D2_10km' / 'acq_000' / 'coords_spectra.npz'


### Function calls
# test_filtering_and_nn(npz_path=spectra_npz)
resave_ASPECT_transmission()