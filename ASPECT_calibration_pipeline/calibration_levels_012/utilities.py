import cv2
import numpy as np
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
from typing import Literal, Iterable, Callable
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import LinearNDInterpolator, interp1d
import inspect
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
from scipy.stats import norm
import warnings
import json
from astropy.io.fits import Header, PrimaryHDU, ImageHDU, BinTableHDU

# numerical eps
_num_eps = 1e-5

# Preprocess the data files


# Based on SP1 and channel determine the order.
# The sp value should be taken from the index 3
# to prevent miss identification.
def check_order(sp: float, channel: str) -> str:
    match channel:
        case 'VIS' | 'NIR1':
            if sp > 19000:
                return 'h'
            else:
                return 'l'
        case 'NIR2':
            if sp > 20000:
                return 'h'
            else:
                return 'l'
        case 'SWIR':
            return ''

#Extract the cahnnel from calib.json file
def read_channel(calibPath: str) -> str:
    with open(calibPath, 'r') as file:
        data = json.load(file)
        if data == None: # As the example SWIR files do not have config data
            return 'SWIR'
        firstKey = list(data.keys())[0]  # Access the first top-level key
        secondKey = list(data[firstKey].keys())[0]  # Access the first sub-key

        # Access the key indicating channel
        channel = list(data[firstKey][secondKey].keys())[0]

    return(channel)

#read the meta data from config file
def read_config(configPath: str, channel:str) -> Tuple[str, List[str], List[str], List[str], List[str]]:
    with open(configPath, 'r') as file:
        data = json.load(file)

        #read SP values for each image
        match channel:
            case 'VIS':
                taskFile = data['visTaskFile']
            case 'NIR1':
                taskFile = data['nir1TaskFile']
            case 'NIR2':
                taskFile = data['nir2TaskFile']
            case 'SWIR':
                taskFile = data['swirTaskFile']
        
        #Extract sp values from taskValues
        taskValues = [taskFile[i:i + 8] for i in range(0, len(taskFile), 8)]
        sp1Values = [taskValues[i][1] for i in range(0, len(taskValues))]
        sp2Values = [taskValues[i][2] for i in range(0, len(taskValues))]
        sp3Values = [taskValues[i][3] for i in range(0, len(taskValues))]
        #Extract exposure times
        exposureTimes = [taskValues[i][4] for i in range(0, len(taskValues))]

        #Check the order based on SP1 index 3
        order = check_order(sp1Values[3], channel)

    return(order, exposureTimes, sp1Values, sp2Values, sp3Values)

def combine_headers(vis: Header, nir1: Header, nir2: Header, swir: Header) -> Dict[str, Any]:
    header_dict = {}
    #VIS
    header_dict['V_ORDER'] = vis.get('ORDER')
    header_dict['V_WL'] = vis.get('WAVELEN')
    header_dict['V_EXPOS'] = vis.get('EXPOS')
    header_dict['V_SP1'] = vis.get('PIEZO1')
    header_dict['V_SP2'] = vis.get('PIEZO2')
    header_dict['V_SP3'] = vis.get('PIEZO3')
    header_dict['V_NUM'] = vis.get('NAXIS3')
    #NIR1
    header_dict['N1_ORDER'] = nir1.get('ORDER')
    header_dict['N1_WL'] = nir1.get('WAVELEN')
    header_dict['N1_EXPOS'] = nir1.get('EXPOS')
    header_dict['N1_SP1'] = nir1.get('PIEZO1')
    header_dict['N1_SP2'] = nir1.get('PIEZO2')
    header_dict['N1_SP3'] = nir1.get('PIEZO3')
    header_dict['N1_NUM'] = nir1.get('NAXIS3')
    #NIR2
    header_dict['N2_ORDER'] = nir2.get('ORDER')
    header_dict['N2_WL'] = nir2.get('WAVELEN')
    header_dict['N2_EXPOS'] = nir2.get('EXPOS')
    header_dict['N2_SP1'] = nir2.get('PIEZO1')
    header_dict['N2_SP2'] = nir2.get('PIEZO2')
    header_dict['N2_SP3'] = nir2.get('PIEZO3')
    header_dict['N2_NUM'] = nir2.get('NAXIS3')
    #SWIR
    header_dict['S_ORDER'] = swir.get('ORDER')
    header_dict['S_WL'] = swir.get('WAVELEN')
    header_dict['S_EXPOS'] = swir.get('EXPOS')
    header_dict['S_SP1'] = swir.get('PIEZO1')
    header_dict['S_SP2'] = swir.get('PIEZO2')
    header_dict['S_SP3'] = swir.get('PIEZO3')
    header_dict['S_NUM'] = swir.get('NAXIS2')

    return header_dict

def append_header(hdu: ImageHDU, dict: Dict[str, Any]) -> ImageHDU:
    header = hdu.header
    for key, value in dict.items():
        header[key] = value
    return hdu

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

def extract_diagnostics(image: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    # Define diagnostic pixel regions
    top = 5  # Five lines at the top
    bottom = 1  # One line at the bottom
    left = 4  # Four columns on the left
    right = 4  # Four columns on the right
    # To store the extracted pixels
    diagnosticPixels = []

    # Step 1: Extract the first 5 rows
    for row in image[:top]:
        diagnosticPixels.append(row.tolist())
    
    # Step 2: For the remaining rows (except the last one), extract the first 4 and last 4 values
    for row in image[top:-bottom]:
        left_values = row[:left]
        right_values = row[-right:]
        combined_row = np.concatenate((left_values, right_values)).tolist()
        diagnosticPixels.append(combined_row)
    
    # Step 3: Extract the last row as a separate list
    diagnosticPixels.append(image[-1].tolist())

    # Remove diagnostic pixels to create the cleaned image
    cleanedImage = image[
        top:-bottom,  # Remove top and bottom rows
        left:-right  # Remove left and right columns
    ]


    return (cleanedImage, diagnosticPixels)

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

def filter_by_orientation(matches, keypoints1, keypoints2, threshold=10) -> List[cv2.DMatch]:
    filtered_matches = []
    for m in matches:
        angle1 = keypoints1[m.queryIdx].angle
        angle2 = keypoints2[m.trainIdx].angle
        angle_diff = abs(angle1 - angle2)
        if angle_diff < threshold or angle_diff > (360 - threshold):
            filtered_matches.append(m)
    return filtered_matches

def filter_by_distance(matches: List[List[cv2.DMatch]]) -> List[cv2.DMatch]:
    ratio_thresh = 0.90  # Adjustable
    good_matches = []
    for m in matches:
        if len(m) == 2:
            match1, match2 = m
            if match1.distance < ratio_thresh * match2.distance:
                good_matches.append(match1)
        elif len(m) == 1:
            match1 = m[0]
            good_matches.append(match1)
    # for m, n in matches:
    #     if m.distance < ratio_thresh * n.distance:
    #         good_matches.append(m)
    distance_thresh = 65  # Adjustable
    good_matches = [m for m in good_matches if m.distance < distance_thresh]
    return good_matches

def estimate_matrix(vis: np.ndarray, nir: np.ndarray) -> np.ndarray:

    # Step 1: Edge detection
    edges1 = laplacian(vis)
    edges2 = laplacian(nir)


    # Step 2: Feature detection using ORB
    orb = cv2.ORB_create(nfeatures=2000) # create ORB feature detector
    # keypoints and binary descriptions
    keypoints1, descriptors1 = orb.detectAndCompute(edges1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(edges2, None)

    # Step 3: Match features
    # FLANN
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                    table_number=30,  # Number of hash tables
                    key_size=20,     # Size of the key
                    multi_probe_level=2)  # Number of probes
        
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params) # Initialize the FLANN

    flann_matches = flann.knnMatch(descriptors1, descriptors2, k=2) # Match features

    #Filter the matches based on the distance. Other option is filter_by_orientation
    matches = filter_by_distance(flann_matches)


    # Step 4: Extract location of good matches and estimate transformation matrix
    # arrays to store x and y coordinates
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    #Extract keypoint coordinates 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Estimate transformation matrix
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 10.0)

    
    #Return the transformation matrix
    return(H)

def asteroid_mask(image: np.ndarray) -> np.ndarray:
    edges = laplacian(image) # Detect asteroid edges

    _, binary_mask = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY) # convert to binary mask

    kernel = np.ones((5,5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel) # Apply morphological closing

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the outermost contours

    asteroid_mask = np.zeros_like(image, dtype=np.uint16)

    cv2.drawContours(asteroid_mask, contours, -1, 65535, thickness=cv2.FILLED) # Draw the asteroid mask

    return asteroid_mask

def extract_asteroid(image_cube: np.ndarray, mask_index: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    image = image_cube[mask_index]

    asteroid_mask = asteroid_mask(image)

    #Store the coordinates of the image where mask has value of non 0
    coords = np.argwhere(asteroid_mask != 0)

    #Extract the corresponding spectra for coords
    spectra = np.array([image_cube[:, y, x] for y, x in coords])

    #Combine coords and the spectra
    combined = list(zip(coords, spectra))

    return combined

def cropND(img: np.ndarray, bounding: tuple[int, int]) -> np.ndarray:
    start = tuple(map(lambda a, da: (a - da) // 2, np.shape(img), bounding))
    end = tuple(map(np.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def accepts_n_params(func, nparams: int):
    if not callable(func):
        return False
    sig = inspect.signature(func)
    return len(sig.parameters) == nparams

def return_ddof(array: np.ndarray, axis: int | None = None) -> int:
    return 1 if np.size(array, axis) > 1 else 0

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
    # Perform operation
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

def return_mean_std(array: np.ndarray, axis: int | None = None) -> tuple[np.ndarray, ...] | tuple[float, ...]:
    mean_value = np.nanmean(array, axis=axis)
    ddof = return_ddof(array, axis=axis)

    std_value = np.nanstd(array, axis=axis, ddof=ddof)

    return mean_value, std_value

def find_outliers(y: np.ndarray, x: np.ndarray | None = None,
                  z_thresh: float = 1.5, num_eps: float = _num_eps) -> np.ndarray:
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

        if 0 in positive or 0 in negative:  # first index is outlier
            outliers = stack(([0], outliers))

        # last index is outlier
        if (len(z_score) - 1) in positive or (len(z_score) - 1) in negative:  # -1 to count "len" from 0
            outliers = stack((outliers, [len(x_iterate) - 1]))

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
    
    return spectrum.reshape(1, -1)
    
def remove_outliers_2d(image: np.ndarray,
                       kernel_size: int = 7,
                       n_std: float = 3.,
                       interp_nans: bool = True,
                       maximum_value: float | None = None) -> np.ndarray:
    
    kernel = np.ones((1, kernel_size))
    kernel /= np.sum(kernel)
    sliding_mean = sliding_window(image=image, kernel=kernel, func="mean")
    sliding_std = sliding_window(image=image, kernel=kernel, func="std")
    mask = np.abs(image - sliding_mean) > n_std * sliding_std
    if maximum_value is not None:
        mask[np.abs(image) > maximum_value] = True
    interp_image = interpolate_mask_1d(image, mask, interp_nans=interp_nans, fill_value=np.nan)
    image = np.asarray(image).flatten()
    interp_image = np.asarray(interp_image).flatten()
    invalid_mask = ~np.isfinite(interp_image)
    interp_image[invalid_mask] = image[invalid_mask]
    return interp_image

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

def denoise_array(array: np.ndarray, sigma: float, x: np.ndarray | None = None,
                  remove_mean: bool = False, sum_or_int: Literal["sum", "int"] = "sum") -> np.ndarray:
    if x is None:
        x = np.arange(0., np.shape(array)[-1])  # 0. to convert it to float

    equidistant_measure = np.var(np.diff(x))
    fwhm_to_sigma = 1. / np.sqrt(8. * np.log(2.))
    fwhm = 2 * np.mean(np.diff(x))
    sigma = fwhm * fwhm_to_sigma

    if equidistant_measure == 0.:  # equidistant step -> gaussian_filter1d is faster
        step = x[1] - x[0]
        correction = gaussian_filter1d(np.ones(len(x)), sigma=sigma / step, mode="constant")
        array_denoised = gaussian_filter1d(array, sigma=sigma / step, mode="constant")

        array_denoised = normalise_in_columns(array_denoised, norm_vector=correction)

    else:  # transmission application
        # Gaussian filters in columns
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

    # if test:
    #     print(f'Initial:\n{nir1_wavelengths}\n{nir2_wavelengths}\n{nir1_spectra}\n{nir2_spectra}')

    coeffs_nir1 = np.polyfit(nir1_wavelengths[-3:], nir1_spectra[-3:], 1)
    coeffs_nir2 = np.polyfit(nir2_wavelengths[:3], nir2_spectra[:3], 1)
    # print(f'nir1 coeffs: {coeffs_nir1}')

    f_nir1 = np.polyval(coeffs_nir1, overlap_wavelength)
    f_nir2 = np.polyval(coeffs_nir2, overlap_wavelength)
    # print(f'f_nir1: {f_nir1}')

    # if test:
    #     print(f'f_nir1: {f_nir1}, f_nir2: {f_nir2}')

    offset = f_nir1 - f_nir2
    corrected_nir2 = nir2_spectra + offset

    # if test:
    #     print("Offset:", offset)
    #     print("Corrected NIR2 spectra:\n", corrected_nir2)
        
    return corrected_nir2, offset