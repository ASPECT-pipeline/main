import numpy as np
from pathlib import Path
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astropy.io import fits 
import cv2

# level 2 imports
from levels_012.modules.utilities import normalize_to_8bit, laplacian, filter_by_distance, overlay_images

# Level 3 imports
from level_3.modules.utilities_spectra import (normalise_spectra, collect_all_models, load_xlsx)
from level_3.modules.NN_evaluate import evaluate
from level_3.modules.NN_data import load_transmission
from level_3.modules.collect_data import resave_ASPECT_transmission
from level_3.modules.NN_config_taxonomy import classes
from level_3.level_3_utilities import spectra_filtering, validate_wl, get_wavelengths, extract_asteroid, remove_index_from_header, asteroid_mask
from level_3.modules.BAR_BC_method import calc_band_parameters
from config import instrument

unfiltered_vis_nir1_nir2_wl = [675., 690., 705., 720., 735., 750., 765., 780., 795., 810., 825., 875., 904.20738725, 933.40538359, 962.41926832, 991.59052354, 1020.78790557, 1050., 1079.21475545, 1108.41876944, 1137.40366510, 1166.57594038, 1195.77918273, 1225., 1225., 1254.22427930, 1283.43620514, 1312.38308545, 1341.55680112, 1370.76774434, 1400., 1429.23686478, 1458.45946423, 1487.35519223, 1516.53104350, 1545.75237632, 1575.]
filtered_vis_nir1_nir2_wl = [675., 690., 705., 720., 735., 750., 765., 780., 795., 810., 825., 875., 904.20738725, 933.40538359, 962.41926832, 991.59052354, 1020.78790557, 1050., 1079.21475545, 1108.41876944, 1137.40366510, 1166.57594038, 1195.77918273, 1225., 1254.22427930, 1283.43620514, 1312.38308545, 1341.55680112, 1370.76774434, 1400., 1429.23686478, 1458.45946423, 1487.35519223, 1516.53104350, 1545.75237632, 1575.]

def get_aspect_wl():
    wl_dict = {
        'AS0' : [675., 690., 705., 720., 735., 750., 765., 780., 795., 810., 825.,],
        'AS1' : [875., 904.20738725, 933.40538359, 962.41926832, 991.59052354, 1020.78790557, 1050., 1079.21475545, 1108.41876944, 1137.40366510, 1166.57594038, 1195.77918273, 1225.],
        'AS2' : [1225., 1254.22427930, 1283.43620514, 1312.38308545, 1341.55680112, 1370.76774434, 1400., 1429.23686478, 1458.45946423, 1487.35519223, 1516.53104350, 1545.75237632, 1575.],
        'AS3' : [1675., 1711.363636, 1747.727273, 1784.090909, 1820.454545, 1856.818182 ,1893.181818, 1929.545455, 1965.909091, 2002.272727, 2038.636364, 2075., 2111.363636, 2147.727273, 2184.090909, 2220.454545, 2256.818182, 2293.181818, 2329.545455, 2365.909091, 2402.272727, 2438.636364, 2475.]
    }
    return wl_dict

"""
Plot functions
"""

def visualise_fits(file, cmap='gray'):

    with fits.open(file) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    
    print(repr(header))
    
    n_frames = data.shape[0]
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)

    idx = 0

    im = ax.imshow(data[idx], cmap=cmap)
    title = ax.set_title(f"Frame {idx+1} /{n_frames}")

    def on_key(event):
        nonlocal idx
        if event.key in ["right", "d"]:
            idx = (idx + 1) % n_frames
        elif event.key in ["left", "a"]:
            idx = (idx - 1) % n_frames
        else:
            return

        frame = data[idx]

        # Count how many are negative
        min = round(np.min(frame), 2)
        mean = round(np.mean(frame), 2)
        max = round(np.max(frame),2)
        print(f"Frame {idx + 1}: min, mean, max: {min}, {mean}, {max}")
        im.set_data(data[idx])
        title.set_text(f"Frame {idx+1}/{n_frames}")
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Use left/right arrow keys or 'a'/'d' to move between frames.")
    plt.show()

def plot_spectra(spectra: np.ndarray, wl: np.ndarray, y_label: str = 'Reflectance (I/F)', x_label: str = 'Wavelengths (nm)', title: str = 'Spectrum'):
    if not len(spectra) == len(wl):
        raise ValueError(f'Shape missmatch: mean spectra {len(spectra)} , wl {len(wl)}')
    
    plt.figure(figsize=(16, 8))
    plt.plot(wl, spectra)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.tight_layout
    plt.show()

def plot_mean_spectra(spectra: np.ndarray | Path, wl: np.ndarray, y_label: str = 'Reflectance (I/F)', x_label: str = 'Wavelengths (nm)', title: str = 'Mean spectrum'):

    if isinstance(spectra, Path):
        coords_spectra = np.load(spectra)
        spectra = coords_spectra['spectra']
        coords = coords_spectra['coords']
        spectra = np.array(spectra) 
    
    mean = np.nanmean(spectra, axis=0)
    try:
        plot_spectra(mean, wl, y_label, x_label, title)
    except Exception as e:
        print(f'Plotting mean spectra failed: {e}')
    
    return mean

def plot_composition(img: np.ndarray | Path, type: str = 'OL'):
    if not type in ('OL', 'OPX', 'CPX'):
        raise ValueError('Composition type must be one of (OL, OPX, CPX)')
    
    if isinstance(img, Path):
        with fits.open(img) as hdul:
            img = hdul[0].data
    
        
    # Rest of the background stays black
    data = img.copy()
    data[data == 0] = np.nan
    mean = round(float(np.nanmean(data)), 2)
    titles = {
        'OL' : 'Olivine abundance',
        'OPX' : 'Orthopyroxene abundance',
        'CPX' : 'Clinopyroxene abundance'
    }

    # Color bars
    if type == 'OL': 
        cmap = plt.cm.turbo.reversed()
    else:
        cmap = plt.cm.turbo

    cmap.set_bad('black')
    title = titles.get(type)
    title = f"{title} (mean: {mean})"
    plt.figure(figsize=(8,4))
    plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()

def plot_taxonomiy(img: np.ndarray | Path, type: str = 'S'):
    if not type in ('S', 'Q', 'L'):
        raise ValueError('Composition type must be one of (S, Q, L)')
    
    if isinstance(img, Path):
        with fits.open(img) as hdul:
            img = hdul[0].data
    
        
    # Rest of the background stays black
    data = img[4].copy()
    data[data == 0] = np.nan
    mean = round(float(np.nanmean(data)), 2)

    # Color bars
    if type == 'S': 
        cmap = plt.cm.turbo
    else:
        cmap = plt.cm.turbo.reversed()

    cmap.set_bad('black')
    title = f"{type} (mean: {mean})"
    plt.figure(figsize=(8,4))
    plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()

def plot_spectra_by_type(img: np.ndarray | Path, model: str = 'C', norm_wl: int = 1546):

    if not model in ('C', 'T'):
        raise ValueError(f"model should be either 'C' (Composition) or 'T' (Taxonomy) ")
    
    if isinstance(img, Path):
        with fits.open(img) as hdul:
            img = hdul[0].data
            header = hdul[0].header
            wavelengths = get_wavelengths(header)
            all_wl = np.sort(np.concatenate([wavelengths[ch] for ch in wavelengths.keys()]))

    combined = extract_asteroid(img, mask_index=0)
    coords, spectras = zip(*combined)
    coords = np.array(coords) 
    spectras = np.array(spectras)

    model_name = 'ASPECT-vis-nir1-nir2-1546'
    print(f'Normalising spectras at {norm_wl}nm')
    spectra_normalized = normalise_spectra(
        data=spectras,
        wavelength=all_wl,
        wvl_norm_nm=norm_wl
    )

    if model == 'C':
        model_subdir = os.path.join('composition', model_name)
        prefix = ""
        model_names = collect_all_models(prefix=prefix, subfolder_model=model_subdir, full_path=True)
        composition = evaluate(model_names, spectra_normalized)
        composition = np.array(composition)

        df = pd.DataFrame(composition)

        print(df.mean())

def filter_spectra(spectra: np.ndarray | Path, wl: np.ndarray):
    if isinstance(spectra, Path):
        coords_spectra = np.load(spectra)
        spectra = coords_spectra['spectra']
        coords = coords_spectra['coords']
        spectra = np.array(spectra) 
    
    wl_dict = get_aspect_wl()
    filtered = spectra_filtering(spectras=spectra, wavelengths=wl_dict, instrument='Vis-NIR1-NIR2')

    plot_mean_spectra(spectra=filtered, wl=wl, title='Filtered Simulated Mean Spectra')

def test_filtering_and_nn(npz_path: str | Path, filtering: bool = True, analysis: bool = False):
    data = np.load(npz_path, allow_pickle=True)
    spectra, coords = data["spectra"], data["coords"]
    data.close()
    # _, _, wvl_central = load_transmission("ASPECT-vis-nir1-nir2")
    all_wl = get_aspect_wl()
    channel_ids = validate_wl(wl=all_wl, instrument=instrument)
    selected_wl = []
    for c in channel_ids:
        selected_wl.append(all_wl.get(f'AS{c}'))
    
    selected_wl = [wl for channel in selected_wl for wl in channel]
    matplotlib.use('MacOSX')
    mean = np.mean(spectra, axis=0)
    plt.figure()
    plt.plot(selected_wl, mean, linewidth=0.8)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Reflectance', fontsize=12)
    plt.title('Mean before filtering')
    plt.tight_layout()
    plt.show()
    if filtering:
        denoised_spectra = spectra_filtering(spectras=spectra, wavelengths=all_wl, instrument=instrument)

    matplotlib.use('MacOSX')
    mean = np.mean(denoised_spectra, axis=0)
    cleaned_wl = np.array(list(dict.fromkeys(selected_wl))) # remove overlap wl
    plt.figure()
    plt.plot(cleaned_wl, mean, linewidth=0.8)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Reflectance', fontsize=12)
    plt.title('Mean before filtering')
    plt.tight_layout()
    plt.show()
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

def creatre_transmissions():
    resave_ASPECT_transmission()

def make_fits(folder, image_shape, name):

    folder = Path(folder)
    rows, cols = image_shape
    expected_size = rows * cols

    def parse_time(name):
        parts = name.replace(".bin", "").split("-")
        hh, mm, ss = map(int, parts[-3:])
        return hh * 3600 + mm * 60 + ss
    
    files = sorted(
        [f for f in folder.iterdir() if f.suffix == '.bin'],
        key=lambda f: parse_time(f.name)
    )
    print("File order:")
    for f in files:
        print(" ", f.name)
    
    images = []
    for file in files:
        data = np.fromfile(file, dtype='<u2')

        if data.size != expected_size:
            raise ValueError(
                f"{file.name}: expected {expected_size} values for shape "
                f"{image_shape}, but found {data.size}"
            )
        
        img = data.reshape(image_shape)
        images.append(img)
            
    cube = np.stack(images, axis=0)

    hdr = fits.Header()

    wl = get_aspect_wl()
    vis_wl = wl.get('AS0')

    for i, frame_wl in enumerate(vis_wl):
        num = str(i).zfill(3)
        hdr[f"HIERARCH AS0_WL_{num}"] = frame_wl
    
    hdu = fits.PrimaryHDU(data=cube, header=hdr)
    hdul = fits.HDUList([hdu])
    out_path = Path(folder) / name
    hdul.writeto(out_path, overwrite=True)

"""
Spectra
"""
def spectral_filtering(fits_file: Path):
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    
    combined = extract_asteroid(data)
    coords, spectra = zip(*combined)
    coords = np.array(coords)
    spectra = np.array(spectra)
    wl = get_wavelengths(header)
    all_wl = np.sort(np.concatenate([wl[ch] for ch in wl.keys()]))

    denoised, cleaned_wl = spectra_filtering(spectra, wavelengths=wl)
    denoised = np.array(denoised)
    print(len(denoised))
    print(denoised.shape)
    denoised_bands = denoised.shape[1]
    height, width = data.shape[1:]
    new_cube = np.zeros((denoised_bands, height, width), dtype=data.dtype)
    for i, (y, x) in enumerate(coords):
        new_cube[:, y, x] = denoised[i]
    
    n_frames = new_cube.shape[0]
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)

    idx = 0

    im = ax.imshow(new_cube[idx], cmap='gray')
    title = ax.set_title(f"Frame {idx+1} /{n_frames}")

    def on_key(event):
        nonlocal idx
        if event.key in ["right", "d"]:
            idx = (idx + 1) % n_frames
        elif event.key in ["left", "a"]:
            idx = (idx - 1) % n_frames
        else:
            return

        frame = new_cube[idx]

        # Count how many are negative
        min = round(np.min(frame), 2)
        mean = round(np.mean(frame), 2)
        max = round(np.max(frame),2)
        print(f"Frame {idx + 1}: min, mean, max: {min}, {mean}, {max}")
        im.set_data(data[idx])
        title.set_text(f"Frame {idx+1}/{n_frames}")
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Use left/right arrow keys or 'a'/'d' to move between frames.")
    plt.show()
   
def spectral_parameters(img: np.ndarray | Path, wl):

    if isinstance(img, Path):
        with fits.open(img) as hdul:
            img = hdul[0].data
            header = hdul[0].header
            wavelengths = get_wavelengths(header)
            all_wl = np.sort(np.concatenate([wavelengths[ch] for ch in wavelengths.keys()]))
    else:
        all_wl = filtered_vis_nir1_nir2_wl
    coords = np.argwhere(img[0] != 0)
    #Extract the corresponding spectra for coords
    spectra = np.array([img[:, y, x] for y, x in coords])
    print(len(spectra))

    spect = spectra[45999]
    all_results = calc_band_parameters(all_wl, spect, visualise=True)

"""
Channel alignment
"""
def visualise_alignment(as0: str, as1:str):

    try:
        with fits.open(as0) as as0_hdul, fits.open(as1) as as1_hdul:
            as0_data = as0_hdul[0].data
            as1_data = as1_hdul[0].data
    except Exception as e:
        print(f'Error occured reading the files: {e}')

    print(f'Visualising alignment')
    vis = as0_data[0]
    nir = as1_data[0]
    print(f"Min, mean, and max values: {np.min(vis)}, {np.mean(vis)}, {np.max(vis)} W sr^-1 m^-2")
    print(f"Min, mean, and max values: {np.min(nir)}, {np.mean(nir)}, {np.max(nir)} W sr^-1 m^-2")
    vis_f = normalize_to_8bit(vis)
    nir_f = normalize_to_8bit(nir)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(vis, cmap='gray')
    plt.title(f'VIS Frame_0')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(nir, cmap='gray')
    plt.title(f'NIR1 Frame_0')
    plt.axis('off')
    plt.suptitle('Comparison of VIS and NIR Channels') 
    plt.show()

    # Step 1: Edge detection
    edges1 = laplacian(vis)
    edges2 = laplacian(nir)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(edges1, cmap='gray')
    plt.title(f'VIS')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges2, cmap='gray')
    plt.title(f'NIR1')
    plt.axis('off')
    plt.suptitle('Laplacian edges') 
    plt.show()

    # Step 2: Feature detection using ORB
    orb = cv2.ORB_create(nfeatures=5000) # create ORB feature detector
    # keypoints and binary descriptions
    keypoints1, descriptors1 = orb.detectAndCompute(edges1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(edges2, None)
    # Draw keypoints on each image
    image1_with_kp = cv2.drawKeypoints(vis_f, keypoints1, None, color=(0, 255, 0), flags=0)
    image2_with_kp = cv2.drawKeypoints(nir_f, keypoints2, None, color=(0, 255, 0), flags=0)

    # Display using matplotlib
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image1_with_kp, cmap='gray')
    plt.title(f'ORB Keypoints ({len(keypoints1)}) - VIS')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2_with_kp, cmap='gray')
    plt.title(f'ORB Keypoints ({len(keypoints2)}) - NIR')
    plt.axis('off')
    plt.suptitle('ORB Feature Keypoints')
    plt.show()

    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                    table_number=30,  # Number of hash tables
                    key_size=20,     # Size of the key
                    multi_probe_level=2)  # Number of probes
        
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params) # Initialize the FLANN
    flann_matches = flann.knnMatch(descriptors1, descriptors2, k=2) # Match features
    print(f'FLANN matches before filtering: {len(flann_matches)}')
    matches = filter_by_distance(flann_matches)
    print(f'FLANN matches after filtering: {len(matches)}')
    N = 1000
    matches_to_draw = matches[:N]
    # Draw matches on combined image
    matched_img = cv2.drawMatches(
        vis_f, keypoints1,
        nir_f, keypoints2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    # Show the match visualization
    plt.figure(figsize=(15, 8))
    plt.imshow(matched_img)
    plt.title(f'{N} FLANN feature matches')
    plt.axis('off')
    plt.show()

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

    little_endian = np.ascontiguousarray(vis.astype('<f4'))
    warpped = cv2.warpPerspective(little_endian, H, (640, 512), flags=cv2.INTER_LINEAR )

    # Convert back to big_endian float32
    vis_aligned = np.ascontiguousarray(warpped.astype('>f4'))
    matplotlib.use('MacOSX')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(vis_aligned, cmap='gray')
    plt.title(f'Aligned VIS')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(nir, cmap='gray')
    plt.title(f' NIR')
    plt.axis('off')
    plt.suptitle('Images after Alignment')
    plt.show()

    print('Visualising the results of alignment')
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='black', label='Aligned regions'),
        Patch(facecolor='red', edgecolor='black', label='Only in vis image'),
        Patch(facecolor='green', edgecolor='black', label='Only in nir image')
    ]
    overlay = overlay_images(vis_aligned, nir)
    plt.figure()
    plt.suptitle('Vis and Nir frame overlay', fontsize=16)
    plt.imshow(overlay)
    plt.axis('off')      
    plt.figlegend(handles=legend_elements, loc='lower center', ncol=3, frameon=True, fontsize='medium')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust to make room for legend and title
    plt.show()

    # Check the difference
    aligned_vis = []
    for f in as0_data:
        f_little_endian = np.ascontiguousarray(f.astype('<f4'))
        f_wrapped = cv2.warpPerspective(f_little_endian, H, (640, 512), flags=cv2.INTER_LINEAR )
        # Convert back to big_endian float32
        f_big_endian = np.ascontiguousarray(f_wrapped.astype('>f4'))
        aligned_vis.append(f_big_endian)
    
    aligned_vis = np.array(aligned_vis)

    N, h_orig, w_orig = as0_data.shape
    _, h_warp, w_warp = aligned_vis.shape

    H_inv = np.linalg.inv(H)

    xs2, ys2 = np.meshgrid(np.arange(w_warp), np.arange(h_warp))
    ones = np.ones_like(xs2)

    pts2 = np.stack([xs2, ys2, ones], axis=-1).reshape(-1, 3).T
    pts1 = H_inv @ pts2
    pts1 /= pts1[2, :]  # normalize by last coordinate

    x1 = pts1[0, :].reshape(h_warp, w_warp).astype(np.float32)
    y1 = pts1[1, :].reshape(h_warp, w_warp).astype(np.float32)

    cube_orig_on_warp = np.empty_like(aligned_vis)
    for k in range(N):
        cube_orig_on_warp[k] = cv2.remap(
            as0_data[k].astype(np.float32),
            x1, y1,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan
        )

    # RMSE per pixel over all bands
    diff = cube_orig_on_warp - aligned_vis
    rmse_per_pixel = np.sqrt(np.nanmean(diff**2, axis=0))  # (h_warp, w_warp)

    print("Mean spectral RMSE:", np.nanmean(rmse_per_pixel))
    print("Max spectral RMSE:", np.nanmax(rmse_per_pixel))
    plt.imshow(rmse_per_pixel, cmap="inferno")
    plt.colorbar(label="Spectral RMSE")
    plt.title("Spectral change per pixel after homography")
    plt.show()

    y, x = 200, 300
    plt.plot(cube_orig_on_warp[:, y, x], label="original→warped grid")
    plt.plot(aligned_vis[:, y, x], label="warped")
    plt.legend()
    plt.title(f"spectrum difference at ({x}, {y})")
    plt.show()


"""
Neural Network
"""

def nn(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    spectra, coords = data["spectra"], data["coords"]
    data.close()

    didymos=[0.169857,0.171412,0.172656,0.173603,0.174266,0.17466,0.174796,0.174689,0.174352,0.173798,0.173041,0.169495,0.167261,0.165391,0.164279,0.164316,0.165744,0.168301,0.171615,0.175316,0.179033,0.1824,0.18523,0.187614,0.187614,0.189665,0.191495,0.193219,0.194941,0.196678,0.198381,0.199998,0.201481,0.202778,0.203842,0.204641,0.205149]
    stype=[0.169741,0.171845,0.173606,0.174945,0.175775,0.176011,0.175228,0.173887,0.172107,0.169953,0.167601,0.160935,0.15909,0.159376,0.161203,0.164059,0.166845,0.169667,0.172985,0.176706,0.180713,0.184249,0.187147,0.189446,0.189446,0.191515,0.193721,0.196152,0.198753,0.201035,0.20301,0.204841,0.206338,0.207399,0.208081,0.208328,0.207962]
    qtype=[0.162714,0.163861,0.164438,0.164274,0.163471,0.161995,0.159473,0.156419,0.153004,0.149456,0.145886,0.135411,0.131097,0.128393,0.127078,0.127003,0.127898,0.129685,0.132363,0.13577,0.139681,0.143305,0.146396,0.148892,0.148892,0.151097,0.15331,0.155687,0.158284,0.160966,0.163618,0.166129,0.168323,0.170073,0.171428,0.172371,0.172848]

    print(len(didymos))
    print(len(stype))
    print(len(qtype))

    spectra = qtype.pop(24)


    _, _, wvl_central = load_transmission("ASPECT-vis-nir1-nir2")

    model_subdir, model_name = "taxonomy/ASPECT-vis-nir1-nir2-1546", "CNN_ASPECT-vis-nir1-nir2-1546_9-1"
    spectra_norm = normalise_spectra(qtype, wavelength=wvl_central, wvl_norm_nm=float(model_name.split("_")[1].split("-")[-1]))
    model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=False)
    predictions = evaluate(model_names, spectra_norm, proportiontocut=0.2, subfolder_model=model_subdir)
    taxonomy = {k: predictions[:, index] for k, index in classes.items()}

    model_subdir, model_name = "composition/ASPECT-vis-nir1-nir2-1546", "CNN_ASPECT-vis-nir1-nir2-1546_1110-11-110-111-000"
    model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=False)
    predictions = evaluate(model_names, spectra_norm, proportiontocut=0.2, subfolder_model=model_subdir)
    quantities = {"OL": 0, "OPX": 1, "CPX": 2, "Fa": 3, "Fo": 4, "Fs (OPX)": 5, "En (OPX)": 6, "Fs (CPX)": 7, "En (CPX)": 8, "Wo (CPX)": 9}
    composition = {k: predictions[:, index] for k, index in quantities.items()}

    df = pd.DataFrame(taxonomy | composition)

    df_percent = df * 100
    print(df_percent.mean())
"""
Asteroid rotation correction
"""
def display_debug(debug_dict, figsize_per_image=4):
    keys = list(debug_dict.keys())
    n = len(keys)

    cols = 3
    rows = int(np.ceil(n / cols))

    plt.figure (figsize=(cols * figsize_per_image, rows * figsize_per_image))

    for idx, key in enumerate(keys, 1):
        img = debug_dict[key]

        plt.subplot(rows, cols, idx)
        plt.imshow(img)
        plt.title(key)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def segment_asteroid(image, blur_ksize=5, min_area=50):

    debug = {}

    img = image.copy()
    debug['original'] = img

    if img.dtype != np.uint8:
        img = normalize_to_8bit(img)

    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    debug['blurred'] = blurred

    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    debug["thresh"] = thresh
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contours found for asteroid segmentation.")

    contours_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contours_vis, contours, -1, (0, 255, 0), 1)
    debug['contours_overlay'] = contours_vis

    per_asteroid_masks = []
    asteroid_props = []

    combined_mask = np.zeros_like(img, dtype=np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < min_area:
            continue  # ignore tiny junk

        # individual mask
        mask_i = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(mask_i, [c], -1, color=255, thickness=cv2.FILLED)
        mask_i = cv2.morphologyEx(mask_i, cv2.MORPH_CLOSE, kernel, iterations=1)

        per_asteroid_masks.append(mask_i)
        combined_mask = cv2.bitwise_or(combined_mask, mask_i)

        # centroid
        m = cv2.moments(c)
        if m["m00"] != 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        else:
            cx = cy = np.nan

        asteroid_props.append(
            {"area": area, "centroid": (cx, cy), "index": idx}
        )

    # mark kept contours in red
    kept_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for c in [contours[p["index"]] for p in asteroid_props]:
        cv2.drawContours(kept_vis, [c], -1, (0, 0, 255), 2)
    debug["asteroids_kept"] = kept_vis

    debug["combined_mask"] = combined_mask

    return combined_mask, debug

    largest = max(contours, key=cv2.contourArea)
    largest_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(largest_vis, [largest], -1, (0, 0, 255), 2)
    debug["largest_contour"] = largest_vis


    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, color=255, thickness=cv2.FILLED)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    debug["mask"] = mask

    return mask, debug

def mask_centroid(mask):
    m = cv2.moments(mask.astype(np.uint8))
    if m["m00"] == 0:
        return mask.shape[1] / 2.0, mask.shape[0] / 2.0
    
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    debug = {}

    debug["mask"] = mask.astype(np.uint8)

    mask_rgb = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.circle(mask_rgb, (int(cx), int(cy)), 5, (0, 255, 0,), -1)
    debug["mask_with_centroid"] = mask_rgb
    return cx, cy, debug

def translate_image_and_mask(image, mask, dx, dy):
    h, w = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img_shifted = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    mask_shifted = cv2.warpAffine(
        mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    debug = {}
    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img_s_rgb = cv2.cvtColor(img_shifted, cv2.COLOR_GRAY2BGR)

    # Original and shifted masks
    debug["original_image"] = img_rgb
    debug["original_mask"] = mask.astype(np.uint8)
    debug["shifted_image"] = img_s_rgb
    debug["shifted_mask"] = mask_shifted.astype(np.uint8)

    # Optionally mark centroids before/after on the images
    cx_before, cy_before, debug = mask_centroid(mask)
    cx_after, cy_after, debug = mask_centroid(mask_shifted)

    img_with_c_before = img_rgb.copy()
    img_with_c_after = img_s_rgb.copy()
    cv2.circle(img_with_c_before, (int(cx_before), int(cy_before)), 5, (0, 255, 0), -1)
    cv2.circle(img_with_c_after, (int(cx_after), int(cy_after)), 5, (0, 255, 0), -1)

    debug["original_image_centroid"] = img_with_c_before
    debug["shifted_image_centroid"] = img_with_c_after
    return img_shifted, mask_shifted, debug

def compute_common_mask(mask, erode_iters=1):
    common = mask[0].copy()
    for m in mask[1:]:
        common = cv2.bitwise_and(common, m)
    
    if erode_iters > 0:
        kernel = np.ones((3,3), np.uint8)
        common = cv2.erode(common, kernel, iterations=erode_iters)
    return common

def register_to_reference(ref_img, img, mask=None):
    h, w = ref_img.shape[:2]

    orb = cv2.ORB_create(nfeatures=1000)

    kpts_ref, des_ref = orb.detectAndCompute(ref_img, mask)
    kpts_img, des_img = orb.detectAndCompute(img, mask)

    if des_ref is None or des_img is None or len(kpts_ref) < 4 or len(kpts_img) < 4:
        raise ValueError("Not enough features for registration.")
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.knnMatch(des_img, des_ref, k=2)

    good = []
    ration = 0.75
    for m, n in matches:
        if m.distance < ration * n.distance:
            good.append(m)
    
    if len(good) < 4:
        raise ValueError("Not enough good matches after ratio test.")
    
    src_pts = np.float32([kpts_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99
    )

    if M is None:
        raise ValueError("Affine estimation failed.")
    
    aligned = cv2.warpAffine(
        img, M, (w, h),
        flags= cv2.INTER_LINEAR,
        borderMode = cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return aligned, M

import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_alignment_grayscale(ref_img, cur_img, aligned_img,
                                  kpts_ref, kpts_cur, matches, frame_idx):
    """
    Visualize alignment steps using grayscale images.
    Only feature matches are shown in color.
    """

    # Convert to 3-channel grayscale so drawMatches can color matches
    ref_color = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    cur_color = cv2.cvtColor(cur_img, cv2.COLOR_GRAY2BGR)

    # Visualize matches (OpenCV draws keypoints & lines in color)
    matches_vis = cv2.drawMatches(
        cur_color, kpts_cur,
        ref_color, kpts_ref,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # RED=reference, GREEN=aligned
    ref_norm = cv2.normalize(ref_img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    ali_norm = cv2.normalize(aligned_img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

    overlay = np.dstack([
        ref_norm,     # R
        ali_norm,     # G
        np.zeros_like(ref_norm)  # B
    ])

    # Make figure
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Frame {frame_idx} Alignment", fontsize=14)

    axs[0,0].imshow(ref_img, cmap="gray")
    axs[0,0].set_title("Reference image")
    axs[0,0].axis("off")

    axs[0,1].imshow(cur_img, cmap="gray")
    axs[0,1].set_title("Current image")
    axs[0,1].axis("off")

    axs[0,2].imshow(matches_vis[..., ::-1])   # BGR→RGB
    axs[0,2].set_title(f"Matches ({len(matches)})")
    axs[0,2].axis("off")

    axs[1,0].imshow(aligned_img, cmap="gray")
    axs[1,0].set_title("Aligned image")
    axs[1,0].axis("off")

    axs[1,1].imshow(overlay)
    axs[1,1].set_title("Overlay: ref (R) + aligned (G)")
    axs[1,1].axis("off")

    axs[1,2].axis("off")

    plt.tight_layout()
    plt.show()


def rotation_correction(fits_file):
    with fits.open(fits_file) as hdul:
        data = hdul[0].data

    # frame_0 = data[0]
    # mask = asteroid_mask(frame_0, erode=0, visualise=True)

    # Step 1: segment asteroid in each original frame
    orig_masks = []
    all_seg_debug = []

    for img in data:
        mask, debug = segment_asteroid(img)
        orig_masks.append(mask)
        all_seg_debug.append(debug)  

    ref_img = normalize_to_8bit(data[0])
    orig_mask = normalize_to_8bit(orig_masks[0])
    aligned_images = [ref_img]
    aligned_masks  = [orig_masks[0]]
    affine_matrices = [np.eye(2, 3, dtype=np.float32)]

    orb = cv2.ORB_create(nfeatures=1000)

    N = len(data)

    for i in range(1, N):
        cur_img = normalize_to_8bit(data[i])
        cur_mask = normalize_to_8bit(orig_masks[i])

        kpts_ref, des_ref = orb.detectAndCompute(ref_img, mask=orig_mask)
        kpts_cur, des_cur = orb.detectAndCompute(cur_img, mask=cur_mask)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des_cur, des_ref, k=2)

        good = []
        ration = 0.75
        for m, n in matches:
            if m.distance < ration * n.distance:
                good.append(m)
        
        if len(good) < 4:
            continue

        src_pts = np.float32([kpts_cur[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M_affine, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, 
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99
        )
        h, w = ref_img.shape[:2]
        aligned_img = cv2.warpAffine(
            cur_img, M_affine, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        aligned_mask = cv2.warpAffine(
            orig_masks[i], M_affine, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        aligned_images.append(aligned_img)
        aligned_masks.append(aligned_mask)
        affine_matrices.append(M_affine)

        # Draw matched keypoints
        matched_vis = cv2.drawMatches(
            cur_img, kpts_cur,
            ref_img, kpts_ref,
            good, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        visualize_alignment_grayscale(
            ref_img=ref_img,
            cur_img=cur_img,
            aligned_img=aligned_img,
            kpts_ref=kpts_ref,
            kpts_cur=kpts_cur,
            matches=good,
            frame_idx=i
        )


    # for i in range(1, N):
    #     print(f"Registering frame {i} to reference...")
    #     aligned_i, M = register_to_reference(ref_img, shifted_images[i], mask=common_mask)
    #     aligned_images.append(aligned_i)
    #     transforms.append(M)

    return aligned_images






""" 
Python3 ASPECT_calibration_pipeline/test_functions.py
"""

### Path library

_results = (Path(__file__).parent.parent / 'pipeline_results').resolve()
D1D2_10km = _results / 'ASPECT_simulated' / 'D1D2_10km' / 'acq_000'
_test_data = (Path(__file__).parent.parent / 'test_data').resolve()

# D1D2
as0 = D1D2_10km / 'AS0_000000_270323T060000_1C.fits'
as1 = D1D2_10km / 'AS1_000000_270323T060000_1C.fits'
as2 = D1D2_10km / 'AS2_000000_270323T060000_1C.fits'
asp = D1D2_10km / 'ASP_000000_270323T060000_2B.fits'
asp_denoised = D1D2_10km / 'ASP_000000_270323T060000_3C_Denoised.fits'

# Spectra
spectra_npz = _results / 'ASPECT_simulated' / 'D1D2_10km' / 'acq_000' / 'coords_spectra_calibrated_unfiltered.npz'
ol = D1D2_10km / 'ASP_000000_270323T060000_3C_OL_comp.fits'
opx = D1D2_10km / 'ASP_000000_270323T060000_3C_OPX_comp.fits'
cpx = D1D2_10km / 'ASP_000000_270323T060000_3C_CPX_comp.fits'
taxonomies = D1D2_10km / 'ASP_000000_270323T060000_3C_Taxonomy.fits'

# Asteroid correction
vis_23 = _test_data / 'ASPECT_simulated_images' / 'vis-2027-03-23-06-00-00' / 'bin_files'
vis_24 = _test_data / 'ASPECT_simulated_images' / 'vis-2027-03-24-00-00-00' / 'bin_files'
vis_rotation_10 = _test_data / 'ASPECT_simulated_images' / 'vis-2027-03-23-06-00-00' / 'vis-2027-03-23-06-00-00.fits'
vis_rotation_5 = _test_data / 'ASPECT_simulated_images' / 'vis-2027-03-24-00-00-00' / 'vis-2027-03-24-00-00-00.fits'

### Function calls
# test_filtering_and_nn(npz_path=spectra_npz)

# plot_composition(cpx, 'CPX')
# plot_taxonomiy(taxonomies, type='L')

# plot_spectra_by_type(asp_denoised, model='C')
# visualise_fits(as2)

# vis_spectral_change(as0, asp)

# spectral_parameters(asp_denoised, None)

# visualise_alignment(as0, as1)

nn(spectra_npz)