import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, interp1d
from typing import Literal
from numpy.lib.stride_tricks import sliding_window_view
import inspect
import utilities

# numerical eps
_num_eps = 1e-5  # num_eps of float32 is 1e-7

def remove_outliers(y: np.ndarray, x: np.ndarray | None = None,
                    z_thresh: float = 1.5, num_eps: float = _num_eps) -> np.ndarray | tuple[np.ndarray, ...]:
    inds_to_remove = utilities.find_outliers(y=y, x=x, z_thresh=z_thresh, num_eps=num_eps)

    if x is None:
        return np.delete(y, inds_to_remove)
    
    # Create a mask of where outliers are
    mask = np.zeros_like(y, dtype=bool)
    mask[inds_to_remove] = True

    # Interpolate the outliers
    interpolated_y = utilities.interpolate_mask_1d(y.copy(), mask=mask)

    # return np.delete(y, inds_to_remove), np.delete(x, inds_to_remove)
    return interpolated_y.flatten(), x