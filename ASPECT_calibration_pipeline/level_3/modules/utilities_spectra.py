import numpy as np
import cv2
import os
import pandas as pd
from typing import List, Tuple, Literal, Callable, Iterable
from modules._constants import (_path_model, _model_suffix, _num_eps, _sep_out, _sep_in, _path_data)
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view
import inspect
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
import warnings
from scipy.stats import norm
from glob import glob
from copy import deepcopy
from pandas.core.common import flatten
from tensorflow.keras.models import load_model, Model
from modules.NN_config_parse import (gimme_endmember_counts, gimme_minerals_all, gimme_num_minerals, bin_to_used)
from modules.NN_losses_metrics_activations import create_custom_objects
from modules.NN_config_composition import minerals_used, endmembers_used

def normalize_to_8bit(img: np.ndarray) -> np.ndarray:
    # Compute min and max values in the image
    min_val = np.min(img)
    max_val = np.max(img)

    # Avoid division by zero in case all values are the same
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.uint8)

    # Normalize image to range 0-255
    normalized = (img - min_val) / (max_val - min_val) * 255.0

    # Convert to 8-bit integer
    return normalized.astype(np.uint8)

def laplacian(img: np.ndarray) -> np.ndarray:

    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not found or unable to open")
    
    # Normalize the image to 8 bit integers
    img = normalize_to_8bit(img)

    # Apply gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3)

    return laplacian

def asteroid_mask(image: np.ndarray) -> np.ndarray:
    edges = laplacian(image) # Detect asteroid edges

    _, binary_mask = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY) # convert to binary mask

    kernel = np.ones((5,5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel) # Apply morphological closing

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the outermost contours

    asteroid_mask = np.zeros_like(image, dtype=np.uint16)

    cv2.drawContours(asteroid_mask, contours, -1, 65535, thickness=cv2.FILLED) # Draw the asteroid mask

    return asteroid_mask

def extract_asteroid(image_cube: np.ndarray, mask_index: int = 0, start_idx: int = 0, end_idx: int = None) -> List[Tuple[np.ndarray, np.ndarray]]:

    image = image_cube[mask_index]

    mask = asteroid_mask(image)

    #Store the coordinates of the image where mask has value of non 0
    coords = np.argwhere(mask != 0)

    #Extract the corresponding spectra for coords
    spectra = np.array([image_cube[start_idx:end_idx, y, x] for y, x in coords])

    #Combine coords and the spectra
    combined = list(zip(coords, spectra))

    return combined


"""

Connecting segments

"""

def nir2_offset_correction(
		nir1_wavelengths: np.ndarray,
		nir1_spectra: np.ndarray,
		nir2_wavelengths: np.ndarray,
		nir2_spectra: np.ndarray,
		overlap_wavelength: int = 1225,
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

    coeffs_nir1 = np.polyfit(nir1_wavelengths[-3:], nir1_spectra[-3:], 1)
    coeffs_nir2 = np.polyfit(nir2_wavelengths[:3], nir2_spectra[:3], 1)

    f_nir1 = np.polyval(coeffs_nir1, overlap_wavelength)
    f_nir2 = np.polyval(coeffs_nir2, overlap_wavelength)


    offset = f_nir1 - f_nir2
    corrected_nir2 = nir2_spectra + offset
        
    return corrected_nir2, offset


"""
Find outliers
"""

def is_constant(array: np.ndarray | list | float, constant: float | None = None, axis: int | None = None,
                atol: float = _num_eps) -> bool | np.ndarray:
    if atol < 0.:
        raise ValueError('"atol" must be a non-negative number')

    array = np.array(array, dtype=float)

    if np.ndim(array) == 0:
        array = array[np.newaxis]

    if constant is None:  # return True if the array is constant along the axis
        ddof = return_ddof(array, axis=axis)

        return np.std(array, axis=axis, ddof=ddof) < atol

    else:  # return True if the array is equal to "constant" along the axis
        return np.all(np.abs(array - constant) < atol, axis=axis)

def stack(arrays: tuple | list, axis: int | None = None, reduce: bool = False) -> np.ndarray:
    """
    concatenate arrays along the specific axis

    if reduce=True, the "arrays" tuple is processed in this way
    arrays = (A, B, C, D)
    stack((stack((stack((A, B), axis=axis), C), axis=axis), D), axis=axis)
    This is potentially slower but allows for concatenating e.g.
    A.shape = (2, 4, 4)
    B.shape = (3, 4)
    C.shape = (4,)
    res = stack((C, B, A), axis=0, reduce=True)
    res.shape = (3, 4, 4)
    res[0] == stack((C, B), axis=0)
    res[1:] == A
    """

    # @reduce_like
    def _stack(arrays: tuple | list, axis: int | None = None) -> np.ndarray:
        ndim = np.array([np.ndim(array) for array in arrays])
        _check_dims(ndim, reduce)

        if np.all(ndim == 1):  # vector + vector + ...
            if axis is None:  # -> vector
                return np.concatenate(arrays, axis=axis)
            else:  # -> 2-D array
                return np.stack(arrays, axis=axis)

        elif np.var(ndim) != 0:  # N-D array + (N-1)-D array + ... -> N-D array
            max_dim = np.max(ndim)

            # longest array
            shape = np.array(np.shape(arrays[np.argmax(ndim)]))
            shape[axis] = -1

            # reshape is dangerous; you can potentially stack e.g. 10x1 with 2x5x2 along axis=0 that is confusing
            # possible dimension difference is one; omit the -1 shape. The rest should be equal.
            shapes = [np.array(np.shape(array)) for array in arrays if np.ndim(array) < max_dim]
            if not np.all([sh in shape[shape > 0] for sh in shapes]):
                raise ValueError("Arrays of these dimensions cannot be stacked.")

            arrays = [np.reshape(array, shape) if np.ndim(array) < max_dim else array for array in arrays]

            return np.concatenate(arrays, axis=axis)

        elif is_constant(ndim):  # N-D array + N-D array + ... -> N-D array or (N+1)-D array
            ndim = ndim[0]
            if axis < ndim:  # along existing dimensions
                return np.concatenate(arrays, axis=axis)
            else:  # along a new dimension
                return np.stack(arrays, axis=axis)

    def _check_dims(ndim: np.ndarray, reduce: bool = False) -> None:
        error_msg = ("Maximum allowed difference in dimension of concatenated arrays is one. "
                     "If you want to stack along higher dimensions, use a combination of stack and np.reshape.")

        if np.max(ndim) - np.min(ndim) > 1:
            if reduce:
                raise ValueError(error_msg)
            else:
                raise ValueError(f'{error_msg}\nUse "reduce=True" to unlock more general (but slower) stacking.')

    # 0-D arrays to 1-D arrays (e.g. add a number to a vector)
    arrays = [np.reshape(array, (1,)) if np.ndim(array) == 0 else np.array(array) for array in arrays]
    arrays = tuple([array for array in arrays if np.size(array) > 0])
    if len(arrays) == 0: arrays = (np.array([], dtype=int),)  # enable to stack(np.array([]))

    if reduce:
        return _stack.reduce(arrays, axis)
    else:
        return _stack(arrays, axis)

def accepts_n_params(func, nparams: int):
    if not callable(func):
        return False
    sig = inspect.signature(func)
    return len(sig.parameters) == nparams

def sliding_window(image: np.ndarray, kernel: np.ndarray,
                   func: Literal["conv2", "min", "max", "median", "mean", "std", "sum"] | callable,
                   mode: Literal["full", "same", "valid"] = "same") -> np.ndarray:
    """
    Optimized sliding window function using NumPy's stride tricks.
    !!!!! func must be computed along axis=(2, 3) !!!!!
    e.g. func=lambda im, k: np.var(im, axis=(2, 3))
    """
    if not (accepts_n_params(func, nparams=2) or func in ["conv2", "min", "max", "median",  "mean", "std", "sum"]):
        raise ValueError('"func" must be func(image, kernel) or be in ["conv2", "min", "max", "median",  "mean", "std", "sum"]')
    if mode not in ["full", "same", "valid"]:
        raise ValueError('"mode" must be in ["full", "same", "valid"]')
    kernel = np.array(kernel, dtype=float)
    image = np.array(image, dtype=float)
    kern_h, kern_w = kernel.shape
    if mode == "full":
        pad_h = (kern_h - 1, kern_h - 1)
        pad_w = (kern_w - 1, kern_w - 1)
    elif mode == "same":
        pad_h = (kern_h // 2, (kern_h - 1) // 2)  # Adjust for even-sized kernels
        pad_w = (kern_w // 2, (kern_w - 1) // 2)  # Adjust for even-sized kernels
    else:  # valid
        pad_h = (0, 0)
        pad_w = (0, 0)
    # Pad the image
    image_padded = np.pad(image, (pad_h, pad_w), mode="constant", constant_values=np.nan)
    # Create strided view of the image
    windows = sliding_window_view(image_padded, window_shape=(kern_h, kern_w))
    if callable(func):
        result = func(windows, kernel)
    elif func == "conv2":
        result = np.nansum(windows * np.rot90(kernel, 2), axis=(2, 3))
    elif func == "sum":
        result = np.nansum(windows, axis=(2, 3))
    elif func == "median":
        result = np.nanmedian(windows, axis=(2, 3))
    elif func == "mean":
        result = np.nanmean(windows, axis=(2, 3))
    elif func == "std":
        result = np.nanstd(windows, ddof=return_ddof(kernel), axis=(2, 3))
    elif func == "max":
        result = np.nanmax(windows, axis=(2, 3))
    elif func == "min":
        result = np.nanmin(windows, axis=(2, 3))
    else:
        raise ValueError("Invalid function")
    return result

def return_ddof(array: np.ndarray, axis: int | None = None) -> int:
    return 1 if np.size(array, axis) > 1 else 0

def return_mean_std(array: np.ndarray, axis: int | None = None) -> tuple[np.ndarray, ...] | tuple[float, ...]:
    mean_value = np.nanmean(array, axis=axis)
    ddof = return_ddof(array, axis=axis)

    std_value = np.nanstd(array, axis=axis, ddof=ddof)

    return mean_value, std_value

def find_outliers(y: np.ndarray, x: np.ndarray | None = None,
                  z_thresh: float = 1.0, num_eps: float = _num_eps, check_edges: bool = False) -> np.ndarray:
    if x is None: x = np.arange(len(y))

    if len(np.unique(x)) != len(x):
        raise ValueError('"x" input must be unique.')
    

    inds = np.argsort(x)
    x_iterate, y_iterate = x[inds], y[inds]

    z_thresh = np.clip(z_thresh, a_min=num_eps, a_max=None)

    while True:
        deriv = np.diff(y_iterate) / np.diff(x_iterate)
        mu, sigma = return_mean_std(deriv) 
        z_score = (deriv - mu) / sigma 
        positive = np.where(np.logical_or(z_score > z_thresh, ~np.isfinite(z_score)))[0]
        negative = np.where(np.logical_or(-z_score > z_thresh, ~np.isfinite(z_score)))[0]

        # noise -> the points are next to each other (overlap if compensated for "diff" shift)
        outliers = stack((np.intersect1d(positive, negative + 1),
                          np.intersect1d(negative, positive + 1)))

        if check_edges:
            if 0 in positive or 0 in negative:  # first index is outlier
                outliers = stack(([0], outliers))
            # last index is outlier
            if (len(z_score) - 1) in positive or (len(z_score) - 1) in negative:  # -1 to count "len" from 0
                outliers = stack((outliers, [len(x_iterate) - 1]))

        # print(f'outliers after 3rd check: {outliers}')
        if np.size(outliers) == 0:
            break

        x_iterate, y_iterate = np.delete(x_iterate, outliers), np.delete(y_iterate, outliers)

    return np.where(~np.in1d(x, x_iterate))[0]

def interpolate_mask_1d(spectrum: np.ndarray, mask: np.ndarray | None = None,
                        interp_nans: bool = True, fill_value: float = np.nan, keep_edges: bool = True) -> np.ndarray: #fill_value: float = np.nan
    if spectrum.ndim == 2 and spectrum.shape[0] == 1:
        spectrum = spectrum.flatten()
        mask = mask.flatten()
    if mask is None:
        mask = np.zeros_like(spectrum, dtype=bool)
    if interp_nans:
        mask = np.logical_or(mask, ~np.isfinite(spectrum))
    if keep_edges:
        mask[0] = False
        mask[-1] = False
    if np.all(~mask):  # No missing values, return as is
        return spectrum
    
    # Get known (valid) points
    known_x = np.where(~mask)[0]  # Indices of valid points
    known_y = spectrum[~mask]     # Corresponding values

    # Get missing points
    missing_x = np.where(mask)[0]

    # Perform 1D linear interpolation
    interpolator = interp1d(known_x, known_y, kind='linear', bounds_error=False, fill_value=fill_value)
    spectrum[missing_x] = interpolator(missing_x)

    # Replace any remaining NaNs if needed
    if interp_nans:
        spectrum[~np.isfinite(spectrum)] = fill_value
    
    """
    
    This part extrapolates the endpoints if keep_edges is False

    """

    # Handle extrapolation only if keep_edges is False
    if not keep_edges:
        # Manually extrapolate the first point if masked
        if mask[0]:
            # Estimate using first two valid points
            idx1, idx2 = known_x[:2]
            val1, val2 = known_y[:2]
            slope = (val2 - val1) / (idx2 - idx1)
            spectrum[0] = val1 - slope * (idx1 - 0)

        # Manually extrapolate the last point if masked
        if mask[-1]:
            # Estimate using last two valid points
            idx1, idx2 = known_x[-2:]
            val1, val2 = known_y[-2:]
            slope = (val2 - val1) / (idx2 - idx1)
            spectrum[-1] = val2 + slope * (len(spectrum) - 1 - idx2)

    return spectrum.reshape(1, -1)

def remove_outliers(y: np.ndarray, x: np.ndarray | None = None,
                    z_thresh: float = 1, num_eps: float = _num_eps, keep_edges: bool = True) -> np.ndarray | tuple[np.ndarray, ...]:
    inds_to_remove = find_outliers(y=y, x=x, z_thresh=z_thresh, num_eps=num_eps)

    if x is None:
        return np.delete(y, inds_to_remove)
    
    # Create a mask of where outliers are
    mask = np.zeros_like(y, dtype=bool)
    mask[inds_to_remove] = True

    # Interpolate the outliers
    interpolated_y = interpolate_mask_1d(y.copy(), mask=mask, keep_edges=keep_edges)

    # return np.delete(y, inds_to_remove), np.delete(x, inds_to_remove)
    return interpolated_y.flatten(), x


"""
Denoise array
"""

def normalise_array(array: np.ndarray,
                    axis: int | None = None,
                    norm_vector: np.ndarray | None = None,
                    norm_constant: float = 1.,
                    num_eps: float = _num_eps) -> np.ndarray:
    if norm_vector is None:
        norm_vector = np.nansum(array, axis=axis, keepdims=True)

    # to force correct dimensions (e.g. when passing the output of interp1d)
    if np.ndim(norm_vector) != np.ndim(array) and np.ndim(norm_vector) > 0:
        norm_vector = np.expand_dims(norm_vector, axis=axis)

    if np.any(np.abs(norm_vector) < num_eps):
        warnings.warn("You normalise with (almost) zero values. Check the normalisation vector.")

    return array / norm_vector * norm_constant

def normalise_in_columns(array: np.ndarray,
                         norm_vector: np.ndarray | None = None,
                         norm_constant: float = 1.) -> np.ndarray:
    return normalise_array(array, axis=0, norm_vector=norm_vector, norm_constant=norm_constant)

def normalise_in_rows(array: np.ndarray,
                      norm_vector: np.ndarray | None = None,
                      norm_constant: float = 1.) -> np.ndarray:
    return normalise_array(array, axis=1, norm_vector=norm_vector, norm_constant=norm_constant)


def denoise_array(array: np.ndarray, sigma: float, x: np.ndarray | None = None,
                  remove_mean: bool = False, sum_or_int: Literal["sum", "int"] = "sum") -> np.ndarray:
    if x is None:
        x = np.arange(0., np.shape(array)[-1])  # 0. to convert it to float

    equidistant_measure = np.var(np.diff(x))
    # print(f'avg step in nm: {np.mean(np.diff(x))}')
    if equidistant_measure == 0.:  # equidistant step -> gaussian_filter1d is faster
        step = x[1] - x[0]
        fwhm_to_sigma = 1. / np.sqrt(8. * np.log(2.))
        fwhm = 2 * step
        sigma = fwhm * fwhm_to_sigma 
        correction = gaussian_filter1d(np.ones(len(x)), sigma=sigma / step, mode="constant")
        array_denoised = gaussian_filter1d(array, sigma=sigma / step, mode="constant")

        array_denoised = normalise_in_columns(array_denoised, norm_vector=correction)

    else:  # transmission application
        # Gaussian filters in columns
        fwhm_to_sigma = 1. / np.sqrt(8. * np.log(2.))
        fwhm = 2 * np.mean(np.diff(x))
        sigma = fwhm * fwhm_to_sigma 
        gaussian = norm.pdf(np.reshape(x, (len(x), 1)), loc=x, scale=sigma)

        # need num_filters x num_wavelengths
        if np.ndim(gaussian) == 1:
            gaussian = np.reshape(gaussian, (1, -1))
        if np.ndim(gaussian) > 2:
            raise ValueError("Filter must be 1-D or 2-D array.")

        if sum_or_int == "sum":
            gaussian = normalise_in_columns(gaussian)
            array_denoised = array @ gaussian
        else:
            gaussian = normalise_in_columns(gaussian, trapezoid(y=gaussian, x=x))
            array_denoised = trapezoid(y=np.einsum("...j, kj -> ...kj", array, gaussian), x=x)

    if remove_mean:  # here I assume that the noise has a zero mean
        mn = np.mean(array_denoised - array, axis=-1, keepdims=True)
    else:
        mn = 0.

    return array_denoised - mn

def denoise_spectra(data: np.ndarray, wavelength: np.ndarray, sigma_nm: float | None = 7.) -> np.ndarray:
    if sigma_nm is None:
        return data

    if sigma_nm <= 0.:
        raise ValueError(f'"sigma_nm" must be positive float but equals {sigma_nm}')

    if np.ndim(data) == 1:
        data = np.reshape(data, (1, len(data)))

    return denoise_array(data, sigma=sigma_nm, x=wavelength)

"""
Part for NN 
"""

def collect_all_models(subfolder_model: str, prefix: str | None = None, suffix: str | None = None,
                       regex: str | None = None, file_suffix: str = _model_suffix, full_path: bool = True) -> list[str]:

    final_suffix = "" if file_suffix == "SavedModel" else f".{file_suffix}"

    if prefix is not None:
        model_str = os.path.join(_path_model, subfolder_model, f"{prefix}*{final_suffix}")
    elif suffix is not None:
        model_str = os.path.join(_path_model, subfolder_model, f"*{suffix}{final_suffix}")
    elif regex is not None:
        model_str = os.path.join(_path_model, subfolder_model, f"{regex}{final_suffix}")
    else:
        model_str = os.path.join(_path_model, subfolder_model, f"*{final_suffix}")

    if full_path:
        return glob(model_str)
    else:
        return [os.path.basename(x) for x in glob(model_str)]
    
def gimme_kind(x: np.ndarray) -> str:
    if len(x) > 3:
        return "cubic"
    if len(x) > 1:
        return "linear"
    return "nearest"

def gimme_model_specification(model_name: str) -> str:
    bare_name = split_path(model_name)[1]

    name_parts = np.array(bare_name.split(_sep_out))

    # dt_string is made of 14 decimals
    dt_string_index = np.where([part.isdecimal() and len(part) == 14 for part in name_parts])[0]

    if np.size(dt_string_index) > 0:
        return _sep_out.join(name_parts[1:dt_string_index[0]])  # cut model_type, dt_string and following parts
    return _sep_out.join(name_parts[1:])  # cut model_type


def gimme_bin_code_from_name(model_name: str) -> str:
    specification = gimme_model_specification(model_name=model_name)
    return specification.split(_sep_out)[-1]

def gimme_indices(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                  reduced: bool = True, return_mineral_indices: bool = False) -> np.ndarray:
    # This function returns the first and last indices of modal/mineral groups
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used


    count_endmembers = gimme_endmember_counts(used_endmembers)
    all_minerals = gimme_minerals_all(used_minerals, used_endmembers)
    num_minerals = gimme_num_minerals(all_minerals)

    indices = np.zeros((len(count_endmembers) + 1, 3), dtype=int)

    indices[0, 0], indices[1:, 0] = -1, np.cumsum(all_minerals) - 1  # cumsum - 1 to get indices

    indices[0, 1:] = 0, num_minerals

    for k, counts in enumerate(count_endmembers):
        indices[k + 1, 1:] = indices[k, 2], indices[k, 2] + counts

    indices = indices[stack(([True], all_minerals))]

    if reduced:
        indices = np.array([[ind_of_mineral, start, stop] for ind_of_mineral, start, stop in indices if start != stop])

    if return_mineral_indices:
        return indices

    return indices[:, 1:]

def gimme_used_from_name(model_name: str) -> tuple[np.ndarray, list[list[bool]]]:
    return bin_to_used(bin_code=gimme_bin_code_from_name(model_name=model_name))

def gimme_custom_objects(model_name: str, **kwargs) -> dict:
    if is_taxonomical(model_name):
        used_minerals, used_endmembers = None, None
    else:
        used_minerals, used_endmembers = gimme_used_from_name(model_name)

    return create_custom_objects(used_minerals=used_minerals, used_endmembers=used_endmembers, **kwargs)

def load_keras_model(filename: str, subfolder: str = "", custom_objects: dict | None = None,
                     compile: bool = True, custom_objects_dict: dict | None = None, **kwargs) -> Model:
    if custom_objects is None:
        if custom_objects_dict is None: custom_objects_dict = {}
        custom_objects = gimme_custom_objects(model_name=filename, **custom_objects_dict)

    filename = check_file(filename, _path_model, subfolder)

    # compile=True is needed to get metrics names for composition vs. taxonomy check
    model = load_model(filename, custom_objects=custom_objects, compile=compile, **kwargs)

    return model



def argnearest(array: np.ndarray, value: float) -> tuple[int, ...]:
    return np.unravel_index(np.nanargmin(np.abs(array - value)), np.shape(array))

def normalise_spectra(data: np.ndarray, wavelength: np.ndarray, wvl_norm_nm: float | None = 550.,
                      on_pixel: bool = True, fun: Callable[[float], float] | None = None) -> np.ndarray:
    if wvl_norm_nm is None:
        return deepcopy(data)

    if np.ndim(data) == 1:
        data = np.reshape(data, (1, len(data)))

    if wvl_norm_nm in wavelength:
        v_norm = data[:, wavelength == wvl_norm_nm]

    elif on_pixel:
        v_norm = data[:, argnearest(wavelength, wvl_norm_nm)]

    elif fun is not None:
        v_norm = fun(wvl_norm_nm)

    else:
        v_norm = interp1d(wavelength, data, kind=gimme_kind(wavelength))(wvl_norm_nm)

    return normalise_in_rows(data, norm_vector=v_norm, norm_constant=1.)

def flatten_list(nested_list: Iterable, general: bool = False) -> np.ndarray:
    # This function flattens a list of lists
    if not general:  # works for a list of lists
        return np.array([item for sub_list in nested_list for item in sub_list])
    else:  # deeply nested irregular lists, dictionaries, numpy arrays, tuples, strings, ...
        return np.array(list(flatten(nested_list)))

def split_path(filename: str, is_dir_check: bool = False) -> tuple[str, ...]:
    if is_dir_check and os.path.isdir(filename):
        return filename, "", ""

    dirname, basename = os.path.split(filename)

    if "." in basename:
        basename, extension = basename.split(".", 1)
    else:
        extension = ""

    return dirname, basename, extension

def is_taxonomical(model: str | Model | None = None, bin_code: str | None = None) -> bool:
    if isinstance(model, str):
        bare_name = split_path(model)[1]
        bin_code = gimme_bin_code_from_name(bare_name)

    if bin_code is not None:
        return len(bin_code.split(_sep_in)) == 2

    if isinstance(model, Model):  # model was compiled when loaded
        if model.metrics_names:
            possible_taxonomy_metrics = ["categorical_accuracy", "f1_score"]
            return np.any([metric in model.metrics_names for metric in possible_taxonomy_metrics])

        # The last layer is output activation or dense (if activation is set as a parameter of Dense)
        if model.get_config()["layers"][-1]["class_name"] in ["Activation", "Dense"]:
            possible_composition_activatins = ["sigmoid_norm", "softmax_norm", "relu_norm", "plu_norm",
                                               "my_sigmoid", "my_softmax", "my_relu", "my_plu"]
            return model.get_config()["layers"][-1]["config"]["activation"] not in possible_composition_activatins

    raise ValueError("Unable to distinguish between composition and taxonomy models.")

def to_list(param) -> list:
    return param if isinstance(param, list) else [param]

def check_file(filename: str, base_folder: str, subfolder: str) -> str:
    if os.path.exists(filename):
        pass
    elif os.path.exists(os.path.join(base_folder, subfolder, filename)):
        filename = os.path.join(base_folder, subfolder, filename)
    else:
        raise FileNotFoundError(f"The file {filename} was not found.")

    return os.path.abspath(filename)

def load_npz(filename: str, subfolder: str = "", list_keys: list[str] | None = None,
             allow_pickle: bool = True, **kwargs):
    filename = check_file(filename, _path_data, subfolder)

    data = np.load(filename, allow_pickle=allow_pickle, **kwargs)

    if list_keys is None:
        return data

    return {key: data[key][()] for key in list_keys if key in data.files}

def load_txt(filename: str, subfolder: str = "", **kwargs) -> pd.DataFrame:
    filename = check_file(filename, _path_data, subfolder)
    data = pd.read_csv(filename, **kwargs)

    return data