import numpy as np
from scipy.stats import trim_mean

from utilities_spectra import (normalise_in_rows, is_taxonomical, gimme_indices, gimme_model_specification, 
                               to_list, load_npz, load_txt, gimme_bin_code_from_name, 
                               gimme_custom_objects, load_keras_model)

from NN_config_parse import bin_to_used

from _constants import _wp, _spectra_name, _quiet

# defaults only
from NN_config_composition import comp_model_setup
from NN_config_taxonomy import tax_model_setup


def average_and_normalise(predictions: np.ndarray, bin_code: str, proportiontocut: float) -> np.ndarray:
    taxonomical = is_taxonomical(bin_code=bin_code)

    # Trimmed mean
    predictions = trim_mean(predictions, proportiontocut, axis=2)

    # Normalisations to 1
    if taxonomical:
        predictions = normalise_in_rows(predictions)

    else:
        used_minerals, used_endmembers = bin_to_used(bin_code=bin_code)

        for start, stop in gimme_indices(used_minerals=used_minerals, used_endmembers=used_endmembers):
            predictions[:, start:stop] = normalise_in_rows(predictions[:, start:stop])


    return np.array(predictions, dtype=_wp)


def check_models(model_names: list[str]) -> None:
    specification_models = [gimme_model_specification(model_name) for model_name in to_list(model_names)]

    # must all be the same
    if not np.all([x == specification_models[0] for x in specification_models]):
        raise ValueError("Not all models have the same specification (grid and output labels).")


def filename_data_to_data(filename_or_data: str | np.ndarray, transpose: bool = False,
                          sep: str = "\t", quiet: bool = False) -> np.ndarray:
    if isinstance(filename_or_data, str):
        # Import the test dataset
        if not quiet:
            print("Loading dataset")

        if ".npz" in filename_or_data:
            filename_or_data = load_npz(filename_or_data, subfolder="")
            filename_or_data = np.array(filename_or_data[_spectra_name], dtype=_wp)

        else:
            filename_or_data = np.array(load_txt(filename_or_data, subfolder="", sep=sep, header=None), dtype=_wp)

    elif isinstance(filename_or_data, np.lib.npyio.NpzFile):
            filename_or_data = np.array(filename_or_data[_spectra_name], dtype=_wp)

    else:
        filename_or_data = np.array(filename_or_data, dtype=_wp)

    if np.ndim(filename_or_data) == 1:
        filename_or_data = np.reshape(filename_or_data, (1, len(filename_or_data)))

    if transpose:
        filename_or_data = np.transpose(filename_or_data)

    # convert data to working precision
    return np.array(filename_or_data, dtype=_wp)


def evaluate(model_names: list[str], filename_or_data: str | np.ndarray,
             proportiontocut: float | None = None,
             subfolder_model: str = "") -> np.ndarray:
    # This function evaluates the mean model on new a dataset

    if not model_names:
        raise ValueError('"model_names" is empty')

    check_models(model_names=model_names)
    bin_code = gimme_bin_code_from_name(model_name=model_names[0])

    # loading needed values
    if is_taxonomical(bin_code=bin_code):
        if proportiontocut is None: proportiontocut = tax_model_setup["trim_mean_cut"]

    else:
        if proportiontocut is None: proportiontocut = comp_model_setup["trim_mean_cut"]

    custom_objects = gimme_custom_objects(model_name=model_names[0])

    filename_or_data = filename_data_to_data(filename_or_data, quiet=_quiet)
    data = filename_or_data

    if not _quiet:
        print("Evaluating the neural network")

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model = load_keras_model(model_name, subfolder=subfolder_model, custom_objects=custom_objects)

        if idx == 0:
            predictions = np.zeros((len(data), model.output_shape[1], len(model_names)), dtype=_wp)

        # Evaluate the model on test data
        predictions[:, :, idx] = model.predict(data, verbose=0)  # model.predict(data, verbose=0)

    # Trimmed means and normalisations to 1
    predictions = average_and_normalise(predictions, bin_code=bin_code, proportiontocut=proportiontocut)
    print("-----------------------------------------------------")

    return predictions
