from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from level_3.modules.utilities import my_argmin, my_argmax, argnearest, gimme_kind
import matplotlib.pyplot as plt

def calc_band_parameters(wavelength: np.ndarray, reflectance: np.ndarray, visualise: bool = False) -> tuple[np.ndarray, ...]:
    """
    calculate spectral parameters (slope, band center, band depth, band width, band area) from the spectra
    """
    pos_max_1 = 680.
    pos_max_2 = 1500.
    pos_max_3 = 2300.

    pos_min_1 = 950.
    pos_min_2 = 2000.

    n_points = 2

    print(len(wavelength))
    print(len(reflectance))
    reflectance = np.reshape(reflectance, (-1, len(wavelength)))

    # sort wavelengths
    idx = np.argsort(wavelength)
    wavelength, reflectance = wavelength[idx], reflectance[:, idx]

    print(reflectance.shape)
    # Slope, band center, band depth, band width, band area
    SLOPES = np.zeros(len(reflectance))
    BIC = np.zeros(len(reflectance))
    BID = np.zeros(len(reflectance))
    BIW = np.zeros(len(reflectance))
    BIAR = np.zeros(len(reflectance))
    BIIC = np.zeros(len(reflectance))
    BIIAR = np.zeros(len(reflectance))
    BIID = np.zeros(len(reflectance))
    BIIW = np.zeros(len(reflectance))

    for i, spectrum in enumerate(reflectance):
        try:
            wvl_max_1 = my_argmax(wavelength, spectrum, x0=pos_max_1, dx=200., n_points=n_points)
            if np.abs(wvl_max_1 - pos_max_1) >= 200.:  # outside the interval
                wvl_max_1 = np.nan
                continue # Skip this iteration
        except Exception as e:
            wvl_max_1 = np.nan
            continue

        try:
            wvl_max_2 = my_argmax(wavelength, spectrum, x0=pos_max_2, dx=300., n_points=n_points)
            if np.abs(wvl_max_2 - pos_max_2) >= 300.:  # outside the interval
                wvl_max_2 = np.nan
                continue
        except Exception as e:
            wvl_max_2 = np.nan
            continue

        try:
            wvl_max_3 = my_argmax(wavelength, spectrum, x0=pos_max_3, dx=200., n_points=n_points)
            if np.abs(wvl_max_3 - pos_max_3) >= 200.:  # outside the interval
                wvl_max_3 = np.nan
        except Exception:
            wvl_max_3 = np.nan

        fun = interp1d(wavelength, spectrum, kind=gimme_kind(wavelength))

        # Slope and continuum line

        x1, x2 = wvl_max_1, wvl_max_2
        y1, y2 = fun(wvl_max_1), fun(wvl_max_2)
        slope = (y1 - y2) / (x1 - x2)
        const = (x1 * y2 - x2 * y1) / (x1 - x2)
        line = slope * wavelength + const
        if visualise:
            print(f'wvl_max_1: {wvl_max_1}')
            print(f'wvl_max_2: {wvl_max_2}')
            print(f'wvl_max_3: {wvl_max_3}')
            plt.figure(figsize=(8,5))
            plt.plot(wavelength, spectrum, label="Spectrum")
            plt.plot([wvl_max_1, wvl_max_2], [y1, y2], 'ro', label="Continuum anchors")
            plt.plot(wavelength, line, 'k--', label="Continuum line (slope)")
            plt.xlabel("Wavelength [nm]")
            plt.ylabel("Reflectance")
            plt.legend()
            plt.title("Spectrum with continuum line")
            plt.show()
            print(f'slope: {slope}')

        SLOPES[i] = slope

        continuum_subtracted = spectrum - line + y1

        if visualise:
            plt.figure(figsize=(8,5))
            plt.plot(wavelength, spectrum, label="Spectrum")
            plt.plot(wavelength, line, "k--", label="Continuum line")
            plt.plot(wavelength, continuum_subtracted, label="Continuum subtracted subtracted")
            plt.xlabel("Wavelength [nm]")
            plt.ylabel("Reflectance / normalized")
            plt.legend()
            plt.title("Slope Corrected Spectrum")
            plt.show()

        # ^WORKS WELL 

        try:
            BIC[i] = my_argmin(wavelength, continuum_subtracted, x0=pos_min_1, dx=250., n_points=n_points)
            if np.abs(BIC[i] - pos_min_1) >= 250.:  # outside the interval
                BIC[i] = np.nan
        except Exception:
            BIC[i] = np.nan

        try:
            BIIC[i] = my_argmin(wavelength, continuum_subtracted, x0=pos_min_2, dx=300., n_points=n_points)
            if np.abs(BIIC[i] - pos_min_2) >= 300.:  # outside the interval
                BIIC[i] = np.nan
        except Exception:
            BIIC[i] = np.nan


        for b_idx, bic in [(1, BIC[i]), (2, BIIC[i])]:
            print(f'band center: {b_idx}')
            if np.isnan(bic):
                print(f'skipping this band center')
                continue

            band_depth = y1 - fun(bic)
            print(f'band depth: {band_depth}')

            if b_idx == 1:
                BID[i] = band_depth
            else:
                BIID[i] = band_depth

            half_depth = band_depth / 2
            y_half = y1 - half_depth

            # FWHM points for band width
            left_below = np.min(np.where((wavelength < bic) & (continuum_subtracted <= y_half))[0])
            left_above = left_below - 1

            left_x1 = wavelength[left_above]
            left_x2 = wavelength[left_below]
            left_y1 = continuum_subtracted[left_above]
            left_y2 = continuum_subtracted[left_below]

            left_k = (y_half - left_y1) / (left_y2 - left_y1)

            left_wl = left_x1 + left_k * (left_x2 - left_x1)
        
            right_below = np.max(np.where((wavelength > bic) & (continuum_subtracted <= y_half))[0])
            right_above = right_below + 1

            right_x1 = wavelength[right_above]
            right_x2 = wavelength[right_below]
            right_y1 = continuum_subtracted[right_above]
            right_y2 = continuum_subtracted[right_below]

            right_k = (y_half - right_y1) / (right_y2 - right_y1)

            right_wl = right_x1 + right_k * (right_x2 - right_x1)

            band_width = right_wl - left_wl

            if b_idx == 1:
                BIW[i] = band_width
            else:
                BIIW[i] = band_width

            if visualise:
                print(f'band width: {band_width}')
                plt.figure(figsize=(7,5))
                plt.plot(wavelength, continuum_subtracted, 'b-', label="Continuum subtracted subtracted")
                plt.plot([wavelength[left_below], wavelength[left_above]], [continuum_subtracted[left_below], continuum_subtracted[left_above]], 'ro', label="left points")
                plt.plot(left_wl, y_half, 'bo', label="left point")
                plt.plot([wavelength[right_below], wavelength[right_above]], [continuum_subtracted[right_below], continuum_subtracted[right_above]], 'ro', label="right points")
                plt.plot(right_wl, y_half, 'go', label="right point")
                plt.axvline(bic, color='k', linestyle='--', alpha=0.5, label="Band center")
                plt.axhline(y_half, color='g', linestyle='--', alpha=0.5, label="Half-depth")
                plt.xlabel("Wavelength [nm]")
                plt.ylabel("Reflectance")
                plt.legend()
                plt.title("Quadratic fit around absorption band")
                plt.show()

                print(f'band width points determined')

            # Band area
            arg_wvl_start = argnearest(wavelength, wvl_max_1)[0]
            arg_wvl_stop = argnearest(wavelength, wvl_max_2)[0]
            fc = y1 - continuum_subtracted
            band_area = trapezoid(y=fc[arg_wvl_start:arg_wvl_stop + 1], x=wavelength[arg_wvl_start:arg_wvl_stop + 1])

            if b_idx == 1:
                BIAR[i] = band_area
            else:
                BIIAR[i] = band_area

            if visualise:
                print(f'band area: {band_area}') 
                xseg = wavelength[arg_wvl_start : arg_wvl_stop+1]
                y_spec = continuum_subtracted[arg_wvl_start : arg_wvl_stop+1]
                y_cont = y1

                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(wavelength, continuum_subtracted, label='Spectrum')
                ax.fill_between(xseg, y_spec, y_cont, alpha=0.3, label=f"Band area = {band_area:.3g}")

                ax.axvline(wvl_max_1, ls=':', color="k", alpha=0.5)
                ax.axvline(wvl_max_2, ls=':', color="k", alpha=0.5)

                ax.set_xlabel("wavelength")
                ax.set_ylabel("Intensity")
                ax.set_title("Band area visualization")
                ax.legend()
                plt.show()


    # return(SLOPES, BIC, BID, BIW, BIAR, BIIC, BIIAR, BIID, BIIW)
    return(SLOPES, BIC, BID, BIW, BIAR)
         


def calc_BAR_BC(wavelength: np.ndarray, reflectance: np.ndarray) -> tuple[np.ndarray, ...]:
    # find extremes around these wavelengths
    pos_max_1 = 680.
    pos_max_2 = 1500.
    pos_max_3 = 2300.

    pos_min_1 = 1000.
    pos_min_2 = 2000.

    n_points = 2

    reflectance = np.reshape(reflectance, (-1, len(wavelength)))

    # sort wavelengths
    idx = np.argsort(wavelength)
    wavelength, reflectance = wavelength[idx], reflectance[:, idx]

    if np.max(wavelength) < 100.:  # most likely in um
        print("Converting wavelengths to nm.")
        wavelength = wavelength * 1000.

    BIC = np.zeros(len(reflectance))
    BIIC = np.zeros(len(reflectance))
    BAR = np.zeros(len(reflectance))

    for i, spectrum in enumerate(reflectance):
        try:
            BIC[i] = my_argmin(wavelength, spectrum, x0=pos_min_1, dx=250., n_points=n_points)
            if np.abs(BIC[i] - pos_min_1) >= 250.:  # outside the interval
                BIC[i] = np.nan
        except Exception:
            BIC[i] = np.nan

        try:
            BIIC[i] = my_argmin(wavelength, spectrum, x0=pos_min_2, dx=300., n_points=n_points)
            if np.abs(BIIC[i] - pos_min_2) >= 300.:  # outside the interval
                BIIC[i] = np.nan
        except Exception:
            BIIC[i] = np.nan

        try:
            wvl_max_1 = my_argmax(wavelength, spectrum, x0=pos_max_1, dx=200., n_points=n_points)
            if np.abs(wvl_max_1 - pos_max_1) >= 200.:  # outside the interval
                wvl_max_1 = np.nan
        except Exception:
            wvl_max_1 = np.nan

        try:
            wvl_max_2 = my_argmax(wavelength, spectrum, x0=pos_max_2, dx=300., n_points=n_points)
            if np.abs(wvl_max_2 - pos_max_2) >= 300.:  # outside the interval
                wvl_max_2 = np.nan
        except Exception:
            wvl_max_2 = np.nan

        try:
            wvl_max_3 = my_argmax(wavelength, spectrum, x0=pos_max_3, dx=200., n_points=n_points)
            if np.abs(wvl_max_3 - pos_max_3) >= 200.:  # outside the interval
                wvl_max_3 = np.nan
        except Exception:
            wvl_max_3 = np.nan


        if np.isnan(wvl_max_1 + wvl_max_2 + wvl_max_3):
            BAR[i] = np.nan
            continue

        # If the founded maxima are out of wavelength range, shift them at the edge
        wvl_max_1 = np.max((wvl_max_1, np.min(wavelength)))
        wvl_max_3 = np.min((wvl_max_3, np.max(wavelength)))

        fun = interp1d(wavelength, spectrum, kind=gimme_kind(wavelength))

        # area of the first band
        # y = slope * x + const
        x1, x2 = wvl_max_1, wvl_max_2
        y1, y2 = fun(wvl_max_1), fun(wvl_max_2)
        slope = (y1 - y2) / (x1 - x2)
        const = (x1 * y2 - x2 * y1) / (x1 - x2)

        line = slope * wavelength + const

        arg_wvl_start = argnearest(wavelength, wvl_max_1)[0]
        arg_wvl_stop = argnearest(wavelength, wvl_max_2)[0]

        fc = line - spectrum
        band_area_1 = trapezoid(y=fc[arg_wvl_start:arg_wvl_stop + 1], x=wavelength[arg_wvl_start:arg_wvl_stop + 1])

        # area of the second band
        if wvl_max_3 > np.max(wavelength) or wvl_max_3 < wvl_max_2 or np.isnan(wvl_max_3):
            wvl_max_3 = np.max(wavelength)

        x1, x2 = wvl_max_2, wvl_max_3
        y1, y2 = fun(wvl_max_2), fun(wvl_max_3)
        slope = (y1 - y2) / (x1 - x2)
        const = (x1 * y2 - x2 * y1) / (x1 - x2)

        line = slope * wavelength + const

        arg_wvl_start = argnearest(wavelength, wvl_max_2)[0]
        arg_wvl_stop = argnearest(wavelength, wvl_max_3)[0]

        fc = line - spectrum
        band_area_2 = trapezoid(y=fc[arg_wvl_start:arg_wvl_stop + 1], x=wavelength[arg_wvl_start:arg_wvl_stop + 1])

        BAR[i] = band_area_2 / band_area_1

    return BAR, BIC, BIIC


def calc_composition(BAR: np.ndarray, BIC: np.ndarray, BIIC: np.ndarray, asteroid_types: np.ndarray | None = None,
                     method: str = "bic") -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # method "biic" for Gaffey, Cloutis
    # method "bic" for Reddy, Dunn

    def calc_Fs_Wo(bic: np.ndarray, biic: np.ndarray,
                   ast_types: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        # https://www.researchgate.net/publication/260055905_Mineralogy_of_Asteroids
        def update_wo(fs: float, bic: float, ast_type: str | None = None) -> float:
            if ast_type is None or "S" in ast_type or "Q" in ast_type or "V" in ast_type:
                if fs < 10.:
                    return 347.9 * bic / 1000. - 313.6
                elif 10. <= fs < 25.:
                    return 456.2 * bic / 1000. - 416.9
                elif 25. <= fs < 50.:
                    return 418.9 * bic / 1000. - 380.9
            return 0.

        def update_fs_bic(bic: float, ast_type: str | None = None) -> float:
            # https://arxiv.org/pdf/1502.05008.pdf Table 2
            if ast_type is None or "S" in ast_type or "Q" in ast_type:
                return -879.1 * (bic / 1000.) ** 2 + 1824.9 * (bic / 1000.) - 921.7
            elif "V" in ast_type:
                return 1023.4 * (bic / 1000.) - 913.82
            return 0.

        def update_fs_biic(wo: float, biic: float, ast_type: str | None = None) -> float:
            if ast_type is None or "S" in ast_type or "Q" in ast_type or "V" in ast_type:
                if wo < 11.:
                    return 268.2 * biic / 1000. - 483.7
                elif 11. <= wo < 30.:
                    return 57.5 * biic / 1000. - 72.7
                elif 30. <= wo < 45.:
                    return -12.9 * biic / 1000. + 45.9
                return -118.0 * biic / 1000. + 278.5
            return 0.

        Fs = np.zeros(len(bic))
        Wo = np.zeros(len(bic))

        for i, (b1c, b2c) in enumerate(zip(bic, biic)):
            if ast_types is None:
                asteroid_type = None
            else:
                asteroid_type = ast_types[i]

                if not("S" in asteroid_type or "Q" in asteroid_type or "V" in asteroid_type):
                    continue

            Fs_old = update_fs_bic(b1c, asteroid_type)
            Wo_old = update_wo(Fs_old, b1c)

            if method == "bic":
                Fs[i], Wo[i] = Fs_old, Wo_old
            else:
                counter = 0

                while 1:  # do-while-like cycle
                    counter += 1
                    Wo_new = update_wo(Fs_old, b1c, asteroid_type)
                    Fs_new = update_fs_biic(Wo_new, b2c, asteroid_type)

                    if np.abs(Fs_new - Fs_old) <= 1e-1 or counter > 10:
                        break
                    Fs_old = Fs_new

                if Wo_old < Wo_new and counter > 10:
                    Fs[i] = Fs_old
                    Wo[i] = Wo_old
                else:
                    Fs[i] = Fs_new
                    Wo[i] = Wo_new

        return Fs, Wo

    def calc_Ol(bar: np.ndarray, ast_types: np.ndarray | None = None) -> np.ndarray:

        Ol = -0.242 * bar + 0.782  # Cloutis is default

        if ast_types is not None:
            for j, (area, ast_type) in enumerate(zip(bar, ast_types)):
                if "S" in ast_type or "Q" in ast_type:
                    # Cloutis' equation (https://arxiv.org/pdf/1502.05008.pdf Table 2)
                    # based on px / (ol + px) -> ol / (ol + px) = 1 - px / (ol + px) | *100 to vol%
                    # OL_fraction = 0.417 * BAR + 0.052
                    # OL_fraction = 100 - 100 * OL_fraction
                    # based on ol / (ol + px) | *100 to vol%
                    Ol[j] = -0.242 * area + 0.782
                elif "A" in ast_type:
                    Ol[j] = -11.27 * area ** 2 + 0.3012 * area + 0.956

        return Ol * 100.

    Ol = calc_Ol(BAR, asteroid_types)
    Fs, Wo = calc_Fs_Wo(BIC, BIIC, asteroid_types)

    return Ol, Fs, Wo


def filter_data_mask(Ol: np.ndarray, Fs: np.ndarray, Wo: np.ndarray, modal_only: bool = False) -> np.ndarray:
    # filter out data outside allowed ranges
    conditions = [Ol >= 0., Ol <= 100.]

    if not modal_only:
        conditions += [Fs >= 0., Fs <= 100.,
                      Wo >= 0., Wo <= 50.,
                      Fs + Wo <= 100.]

    mask = np.logical_and.reduce(conditions)

    return mask
