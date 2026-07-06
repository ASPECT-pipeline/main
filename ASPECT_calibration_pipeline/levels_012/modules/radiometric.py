import numpy as np
from astropy.io.fits import HDUList
from config import _path_sim_coef, _path_radiance, reverse_channel_map
from pathlib import Path
import pandas as pd

"""
Function for converting the pixel values into scientific units.

    Description:
        - Iterated over all 2D images inside the data cube multiplying it with a coefficient.
        - Creates a new FITS file with the calibrated data
"""

def parse_radiance_file(txt_path: str | Path) -> pd.DataFrame: 
    """
    Parse a tab-separated radiance clibration file into a Pandas DataFrame
    Reads 4 columns (Wl[nm, Resp.[DN/s], Radiance[W/m2/nm/sr], Response[DN/(s*W/m2/sr/nm)])
    Converts to numeric, drops invalid and keeors the last duplicate, sorts by wavelength.

    Returns a DataFrane with columns:
    ['wl_nm', 'resp_dn_s', 'radiance_w_m2_nm_sr', 'response_dn_per_w']
    """
    df = pd.read_csv(
        txt_path,
        sep=r"\t+",
        engine="python",
        comment="#",
    )

    df.columns = [c.strip() for c in df.columns]

    expected_cols = 4
    if df.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {df.shape[1]} in {txt_path}")
    
    df.columns = ["wl_nm", "resp_dn_s", "radiance_w_m2_nm_sr", "response_dn_per_w"]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors='coerce')
    
    df = df.dropna(subset=["wl_nm"]).reset_index(drop=True)

    df = df.drop_duplicates(subset=['wl_nm'], keep="last")

    df = df.sort_values("wl_nm").reset_index(drop=True)

    return df

def interp_values(df: pd.DataFrame, wl: float) -> dict:
    x = df["wl_nm"].to_numpy()
    out = {}
    for col in ["resp_dn_s", "radiance_w_m2_nm_sr", "response_dn_per_w"]:
        y = df[col].to_numpy()
        out[col] = float(np.interp(wl, x, y))
    
    return out

def radiometric_calibration(hdul: HDUList) -> HDUList:
    """
    Function for converting the pixel values into scientific units.

     Parmeters:
        hdul (HDUList): The HDU list of the FITS file that is modified
    
    Returns:
        The modified HDU list of the FITS file
    """

    # Data from fits file
    hdu = hdul[0]
    header = hdu.header
    data = hdu.data
    missphas = header.get('MISSPHAS')
    channel = header.get('ASP_CHANNELS')
    channel_id = reverse_channel_map.get(channel)

    distanceToSun = 1.0 # au

    if missphas == 'SIMULATED':
        match channel:
            case 'Vis':
                coef_file = Path(_path_sim_coef) / 'calibration-coefficients-VIS.dat'
                integration_time = 10
            case 'NIR1':
                coef_file = Path(_path_sim_coef) / 'calibration-coefficients-NIR1.dat'
                integration_time = 20
            case 'NIR2':
                coef_file = Path(_path_sim_coef) / 'calibration-coefficients-NIR2.dat'
                integration_time = 20
            case _:
                coef_file = ''
                return hdul
        coefs = np.loadtxt(coef_file)
        try:
            new_data_cube = data.astype(np.float64, copy=True)
            #loop over the 2D images inside the extension
            for i, image in enumerate(data):
                cal_image = (image * coefs[i,1] / (integration_time * distanceToSun**2)).astype(np.float64) # multiply the image with the coefficient 
                new_data_cube[i] = cal_image
            data = new_data_cube
            hdul[0].data = data
            print(f'Radiometric calibrated')
            return hdul
        except Exception as e: 
            print(f'[WARNING] Radiometric calibration failed: {e}')
            return hdul
    else:
        try: 
            channel_id = reverse_channel_map.get(channel)
            if channel == 'SWIR':
                    return hdul
            order = header.get(f'AS{channel_id}_ORDER')
            if order not in ('LOW', 'HIGH'):
                print(f'[WARNING] channel {channel} order is {order}. Radiometric calibration failed.')
                return hdul

            radinace_file = _path_radiance / f'AS{channel_id}_RADIANCE_{order}.txt'
            df = parse_radiance_file(radinace_file)

            # Get task information from header
            frames = header.get(f'AS{channel_id}_FRAMES').split(',')


            new_data_cube = data.astype(np.float64, copy=True)
            for i, frame in enumerate(data):
                wl = header.get(f'AS{channel_id}_WL_{frames[i]}')
                exposure = header.get(f'AS{channel_id}_EXP_{frames[i]}')

                interp_vals = interp_values(df, float(wl))
                response = float(interp_vals.get('response_dn_per_w'))
                coefficient = float(exposure) * response

                new_data_cube[i] = frame / coefficient
            
            hdul[0].data = new_data_cube
            print(f'Radiometrically calibrated')
        except Exception as e:
            print(f'[WARNING] Radiometric calibration failed: {e}')
        return hdul