import numpy as np
from warnings import warn
from pandas.core.common import flatten
from typing import Iterable

def gimme_separator(bin_code: str) -> str:
    non_digit = np.array([not e.isdigit() for e in list(bin_code)])
    separator = np.unique(np.array(list(bin_code))[non_digit])

    if len(separator) > 1:
        raise ValueError(f"Non-unique separator in {bin_code}.")

    return separator[0]

def gimme_used_quantities(minerals: np.ndarray,
                          endmembers: list[list[bool]]) -> tuple[np.ndarray, list[list[bool]]]:
    # If there is only one end-member for a given mineral, the information is redundant and worsens the optimisation
    endmembers_used = [endmember if (mineral and np.sum(endmember) > 1) else len(endmember) * [False]
                       for mineral, endmember in zip(minerals, endmembers)]

    # If there is only one mineral, the modal information is redundant and worsens the optimisation
    minerals_used = minerals if np.sum(minerals) > 1 else np.array([False] * len(minerals))

    return minerals_used, endmembers_used

def gimme_endmember_counts(used_endmembers: list[list[bool]]) -> np.ndarray:
    return np.array([np.sum(endmember) for endmember in used_endmembers])

def gimme_minerals_all(used_minerals: np.ndarray, used_endmembers: list[list[bool]]) -> np.ndarray:
    return np.where(gimme_endmember_counts(used_endmembers) > 0, True, used_minerals)

def gimme_num_minerals(used_minerals: np.ndarray) -> int:
    num_minerals = int(np.sum(used_minerals))
    return num_minerals if num_minerals > 1 else 0

def check_used_quantities(used_minerals: np.ndarray, used_endmembers: list[list[bool]],
                          raise_error: bool = True) -> bool:
    # check the input
    endmember_counts = gimme_endmember_counts(used_endmembers=used_endmembers)

    # no "single" labels
    if np.sum(used_minerals) == 1 or np.any(endmember_counts == 1):
        error_msg = "No singleton labels."
        if raise_error:
            raise ValueError(error_msg)
        else:
            warn(error_msg)
            return False

    # no mineral for end-member (one allowed if no minerals)
    if np.sum(used_minerals) == 0 and np.sum(endmember_counts[~used_minerals] > 0) > 1:
        error_msg = ("Missing mineral label for one of endmember group "
                     "(only one group is allowed if no minerals are present).")
        if raise_error:
            raise ValueError(error_msg)
        else:
            warn(error_msg)
            return False

    # no mineral for endmember
    if np.sum(used_minerals) > 1 and np.sum(endmember_counts[~used_minerals] > 0) > 0:
        error_msg = "Missing mineral label for one of endmember group."
        if raise_error:
            raise ValueError(error_msg)
        else:
            warn(error_msg)
            return False

    return True

def flatten_list(nested_list: Iterable, general: bool = False) -> np.ndarray:
    # This function flattens a list of lists
    if not general:  # works for a list of lists
        return np.array([item for sub_list in nested_list for item in sub_list])
    else:  # deeply nested irregular lists, dictionaries, numpy arrays, tuples, strings, ...
        return np.array(list(flatten(nested_list)))

def bin_to_used(bin_code: str, separator: str | None = None,
                return_all: bool = False) -> tuple[np.ndarray, list] | tuple[np.ndarray, list, np.ndarray]:
    # This function converts info from binary code in the name of a model into mineral abundances and composition
    # and gives you minerals_used and end_member used
    # e.g. bin_to_composition("1110-11-110-000-000") = np.array(["OL", "OPX", "CPX", "Fa" "Fo", "Fs (OPX)", "En (OPX)"])

    error_msg = f'Invalid bin code input "{bin_code}".'

    if separator is None: separator = gimme_separator(bin_code=bin_code)

    used_quantities = [list(quantity) for quantity in bin_code.split(separator)]

    used_quantities_flat = flatten_list(used_quantities)
    if not np.all(np.logical_or(used_quantities_flat == "1", used_quantities_flat == "0")):
        raise ValueError(f'{error_msg} Bin code must be made of "1" and "0" only.')

    used_quantities = [[quantity == "1" for quantity in quantities] for quantities in used_quantities]

    used_minerals = np.array(used_quantities[0], dtype=bool)
    used_endmembers = used_quantities[1:]

    if len(used_minerals) != len(used_endmembers):
        raise ValueError(f'The length of the used minerals does not equal the length of the used end-members.\n'
                         f'Probably an incorrect separator. Bin code "{bin_code}", separator "{separator}".')

    # check the input
    check_used_quantities(used_minerals=used_minerals, used_endmembers=used_endmembers, raise_error=True)

    if return_all:
        used_minerals = gimme_minerals_all(used_minerals, used_endmembers)

    return used_minerals, used_endmembers
