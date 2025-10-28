import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon
import time
import pandas as pd
import itertools
from ASPECT_setpoint_wavelength_recalibration import (
	vis_lo_setpoint_to_wavelength,
	vis_ho_setpoint_to_wavelength,
	nir1_lo_setpoint_to_wavelength,
	nir1_ho_setpoint_to_wavelength,
	nir2_lo_setpoint_to_wavelength,
	nir2_ho_setpoint_to_wavelength
)

# Acquisitions of ASPECT_TESTS_25_7:

# ASPECT_TESTS_25_7_non_dark = [
# 	'2024-07-25_11-59-06_vis_h_vis_lo_600w_2500',
# 	'2024-07-25_12-04-30_vis_h_vis_ho_600w_2500',
# 	# '2024-07-25_12-37-11_vis_h_vis_lo_600w_1875',# Error setpoint 19939 (claims to be lo but has ho setpoints)
# 	'2024-07-25_12-46-50_vis_h_vis_ho_600w_1875',
# 	'2024-07-25_12-55-01_vis_h_vis_lo_600w_1250',
# 	'2024-07-25_13-01-06_vis_h_vis_ho_600w_1250',
# 	# '2024-07-25_13-07-40_vis_h_vis_lo_600w_625',#
# 	# '2024-07-25_13-13-49_vis_h_vis_ho_600w_625',#
# 	'2024-07-25_13-22-13_vis_h_vis_lo_400w_2500',
# 	'2024-07-25_13-28-18_vis_h_vis_ho_400w_2500',
# 	'2024-07-25_13-35-53_vis_h_vis_lo_200w_2500',
# 	'2024-07-25_13-42-25_vis_h_vis_ho_200w_2500',
# 	'2024-07-25_14-20-42_nir1_h_nir1_lo_600w_10000',
# 	'2024-07-25_14-23-40_nir1_h_nir1_ho_600w_10000',
# 	'2024-07-25_14-27-03_nir1_h_nir1_lo_600w_7500',#
# 	'2024-07-25_14-30-18_nir1_h_nir1_ho_600w_7500',
# 	'2024-07-25_14-34-04_nir1_h_nir1_lo_600w_5000',
# 	'2024-07-25_14-37-12_nir1_h_nir1_ho_600w_5000',
# 	'2024-07-25_14-40-51_nir1_h_nir1_lo_600w_2500',
# 	'2024-07-25_14-43-53_nir1_h_nir1_ho_600w_2500',
# 	'2024-07-25_14-48-27_nir1_h_nir1_lo_400w_10000',
# 	'2024-07-25_14-52-39_nir1_h_nir1_ho_400w_10000',
# 	'2024-07-25_14-57-21_nir1_h_nir1_lo_200w_10000',
# 	'2024-07-25_15-00-41_nir1_h_nir1_ho_200w_10000',
# 	'2024-07-25_15-19-38_nir2_h_nir2_lo_600w_10000',
# 	'2024-07-25_15-25-02_nir2_h_nir2_ho_600w_10000',
# 	'2024-07-25_15-36-06_nir2_h_nir2_lo_600w_7500',#
# 	'2024-07-25_15-41-18_nir2_h_nir2_ho_600w_7500',
# 	'2024-07-25_15-45-37_nir2_h_nir2_lo_600w_5000',
# 	'2024-07-25_15-50-56_nir2_h_nir2_ho_600w_5000',
# 	'2024-07-25_15-54-56_nir2_h_nir2_lo_600w_2500',
# 	'2024-07-25_15-58-35_nir2_h_nir2_ho_600w_2500',
# 	'2024-07-25_16-05-19_nir2_h_nir2_lo_400w_10000',
# 	'2024-07-25_16-08-47_nir2_h_nir2_ho_400w_10000',
# 	'2024-07-25_16-14-10_nir2_h_nir2_lo_200w_10000',
# 	'2024-07-25_16-18-11_nir2_h_nir2_ho_200w_10000',
# ]

def load_image(file_path, shape):
	with open(file_path, 'rb') as f:
		data = np.frombuffer(f.read(), dtype=np.int16)
	if data.size != np.prod(shape):
		raise ValueError(f"Data size in {file_path} does not match the specified shape {shape}.")
	return data.reshape(shape)

def get_title(file_path, format=0):
	file_name = os.path.basename(file_path)
	directory = os.path.basename(os.path.dirname(file_path))
	parent_of_directory = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
	if format == 0:
		return os.path.join(directory, file_name)
	elif format == 1:
		return os.path.join(parent_of_directory, directory, file_name)
	elif format == 2:
		return os.path.join(parent_of_directory, file_name)
	elif format == 3:
		return file_name
	elif format == 4:
		return directory
	
# print(get_title('/home/sysa/HERA/test_data/test_outputs/Dark_after-subtracted(ASPECT_TESTS_25.7)/2024-07-25_11-59-06_vis_h_vis_lo_600w_2500', format=3))

def read_channel(calibPath):
	with open(calibPath, 'r') as file:
		data = json.load(file)
		if data == None:
			return 'SWIR'
		firstKey = list(data.keys())[0]
		secondKey = list(data[firstKey].keys())[0]
		channel = list(data[firstKey][secondKey].keys())[0]
	return(channel)

def visualize_binary_files_scroll(folder_path, shape):
	"""
	Visualizes int16 binary files in a folder using a Matplotlib slider to scroll through images.
	
	Parameters:
		folder_path (str): The path to the folder containing the binary files.
		shape (tuple): The (height, width) tuple to reshape each binary file into an image.
	"""
	# List and sort the .bin files in the folder
	files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.bin')])
	if not files:
		raise ValueError("No binary files with a .bin extension found in the folder.")
	
	# Read the first file to set up the initial plot.
	
	# Initial image setup
	init_index = 0
	image = load_image(files[init_index], shape)
	
	# Create the figure and a subplot for the image
	fig, ax = plt.subplots(figsize=(8, 6))
	plt.subplots_adjust(bottom=0.25)  # Make room for the slider below the image
	
	im_plot = ax.imshow(image, cmap='gray', aspect='equal')
	name = get_title(files[init_index])
	title = ax.set_title(name)
	cbar = plt.colorbar(im_plot, ax=ax)
	cbar.set_label('Intensity')
	
	# Create an axes area for the slider
	slider_ax = plt.axes([0.20, 0.1, 0.65, 0.03])
	slider = Slider(
		ax=slider_ax, 
		label='Image Index', 
		valmin=0, 
		valmax=len(files)-1, 
		valinit=init_index,
		valstep=1,
		valfmt='%d'
	)
	
	# Update function: loads a new image and updates the plot when slider value changes
	def update(val):
		idx = int(slider.val)
		new_image = load_image(files[idx], shape)
		im_plot.set_data(new_image)
		name = get_title(files[idx])
		title.set_text(name)
		# Redraw the image with the new data.
		fig.canvas.draw_idle()
	
	slider.on_changed(update)
	plt.show()
	

visualize_bin_folder = False
if visualize_bin_folder:
	visualize_binary_files_scroll(
		'/home/sysa/HERA/test_data/test_outputs/Dark_after-subtracted(ASPECT_TESTS_25.7)/2024-07-25_14-23-40_nir1_h_nir1_ho_600w_10000',
		(506, 636)#648(1024, 1024)
	)

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

def nir1_offset_correction(
		nir1_wavelengths: np.ndarray,
		nir1_spectra: np.ndarray,
		nir2_wavelengths: np.ndarray,
		nir2_spectra: np.ndarray,
		overlap_wavelength: int = 1225,
	):
    """
    Corrects NIR1 spectra by aligning the overlap region with NIR2 spectra using linear regression.

    Parameters:
        nir1_wavelengths (np.ndarray): Wavelengths of NIR1 wavelengths.
        nir1_spectra (np.ndarray): NIR1 spectra.
        nir2_wavelengths (np.ndarray): Wavelengths of NIR2 wavelengths.
        nir2_spectra (np.ndarray): NIR2 spectra.
        overlap_wavelength (int): Wavelength at which to align the spectra.
        test (bool): If True, prints the calculated values.

    Returns:
        corrected_nir1 (np.ndarray): Corrected NIR1 spectra.
        offset (float): Offset value.
    """

    coeffs_nir1 = np.polyfit(nir1_wavelengths[-3:], nir1_spectra[-3:], 1)
    coeffs_nir2 = np.polyfit(nir2_wavelengths[:3], nir2_spectra[:3], 1)

    f_nir1 = np.polyval(coeffs_nir1, overlap_wavelength)
    f_nir2 = np.polyval(coeffs_nir2, overlap_wavelength)


    offset = f_nir2 - f_nir1
    corrected_nir1 = nir1_spectra + offset
        
    return corrected_nir1, offset

def visualize_int16_binary(file_path, shape):
	"""
	Visualizes an int16 binary file as an image using matplotlib.

	Parameters:
		file_path (str): The path to the binary file.
		shape (tuple): The (height, width) to reshape the binary data into an image.

	Example:
		visualize_int16_binary('path/to/image.bin', (1024, 1024))
	"""
	# Read the binary data as int16
	with open(file_path, 'rb') as file:
		data = np.frombuffer(file.read(), dtype=np.int16)
	
	# Ensure the binary data can be reshaped to the given dimensions
	if data.size != np.prod(shape):
		raise ValueError(f"Data size {data.size} does not match the specified shape {shape}.")

	# Reshape the data into an image
	image = data.reshape(shape)

	title = get_title(file_path)

	# Create the plot
	plt.figure(figsize=(8, 6))
	plt.imshow(image, cmap='gray', aspect='equal')
	plt.colorbar(label='Intensity')
	plt.title(title)
	plt.xlabel('Columns')
	plt.ylabel('Rows')
	plt.show()

visualize_single_binary = False
if visualize_single_binary:
	visualize_int16_binary(
		'/home/sysa/HERA/test_data/test_outputs/Dark_after-subtracted(ASPECT_TESTS_25.7)/2024-07-25_11-59-06_vis_h_vis_lo_600w_2500/dc_0_exp_000.bin',
		(1024, 1024)
	)

def visualize_vis_nir1_nir2(vis_file_path, nir1_file_path, nir2_file_path, shapes=[(1024, 1024), (518, 648), (518, 648)]):
	"""
	Visualizes VIS, NIR1, and NIR2 int16 binary files as images in a single matplotlib figure.

	Parameters:
		vis_file_path (str): Path to the VIS binary file.
		nir1_file_path (str): Path to the NIR1 binary file.
		nir2_file_path (str): Path to the NIR2 binary file.
		shapes (list of tuple): A list containing three (height, width) tuples for VIS, NIR1, and NIR2 images respectively.
		
	Example:
		visualize_vis_nir1_nir2('path/to/VIS.bin', 'path/to/NIR1.bin', 'path/to/NIR2.bin')
	"""
	
	# Load images
	vis_image   = load_image(vis_file_path, shapes[0])
	nir1_image  = load_image(nir1_file_path, shapes[1])
	nir2_image  = load_image(nir2_file_path, shapes[2])
	
	vis_title  = get_title(vis_file_path, format=2)
	nir1_title = get_title(nir1_file_path, format=2)
	nir2_title = get_title(nir2_file_path, format=2)
	
	# Create subplots for the three images
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	
	# VIS image
	im0 = axes[0].imshow(vis_image, cmap='gray', aspect='equal')
	axes[0].set_title(vis_title)
	axes[0].set_xlabel('Columns')
	axes[0].set_ylabel('Rows')
	plt.colorbar(im0, ax=axes[0], label='Intensity')
	
	# NIR1 image
	im1 = axes[1].imshow(nir1_image, cmap='gray', aspect='equal')
	axes[1].set_title(nir1_title)
	axes[1].set_xlabel('Columns')
	axes[1].set_ylabel('Rows')
	plt.colorbar(im1, ax=axes[1], label='Intensity')
	
	# NIR2 image
	im2 = axes[2].imshow(nir2_image, cmap='gray', aspect='equal')
	axes[2].set_title(nir2_title)
	axes[2].set_xlabel('Columns')
	axes[2].set_ylabel('Rows')
	plt.colorbar(im2, ax=axes[2], label='Intensity')
	
	plt.tight_layout()
	plt.show()

# This might be deprecated

# imshow_vis_nir1_nir2 = False
# if imshow_vis_nir1_nir2:
# 	visualize_vis_nir1_nir2(
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_12-04-30_vis_h_vis_ho_600w_2500/acq_000/dc_0_exp_005.bin',
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_14-23-40_nir1_h_nir1_ho_600w_10000/acq_000/dc_1_exp_005.bin',
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_15-25-02_nir2_h_nir2_ho_600w_10000/acq_000/dc_2_exp_005.bin'
# 	)

def visualize_vis_nir1_nir2_with_markers(vis_file_path, nir1_file_path, nir2_file_path, coordinates, shapes=[(1024, 1024), (518, 648), (518, 648)]):
	"""
	Visualizes VIS, NIR1, and NIR2 int16 binary files as images in a single matplotlib figure,
	and overlays markers for given coordinates.

	Parameters:
		vis_file_path (str): Path to the VIS binary file.
		nir1_file_path (str): Path to the NIR1 binary file.
		nir2_file_path (str): Path to the NIR2 binary file.
		coordinates (list of list of tuple): A list where each element is a list/tuple containing
			three coordinate pairs:
				[ (vis_x, vis_y), (nir1_x, nir1_y), (nir2_x, nir2_y) ]
			representing marker positions for each image.
		shapes (list of tuple): A list containing three (height, width) tuples for VIS, NIR1, and NIR2 images respectively.
		
	Example:
		coordinates = [
			[(100, 150), (50, 75), (60, 80)],  # Marker positions for coordinate set 1
			[(400, 450), (200, 220), (210, 230)] # Marker positions for coordinate set 2
		]
		visualize_with_markers('path/to/VIS.bin', 'path/to/NIR1.bin', 'path/to/NIR2.bin', coordinates)
	"""
	
	# Load the images using the provided load_image function
	vis_image   = load_image(vis_file_path, shapes[0])
	nir1_image  = load_image(nir1_file_path, shapes[1])
	nir2_image  = load_image(nir2_file_path, shapes[2])
	
	# Get titles using the provided get_title function
	vis_title  = get_title(vis_file_path, format=2)
	nir1_title = get_title(nir1_file_path, format=2)
	nir2_title = get_title(nir2_file_path, format=2)
	
	# Create subplots for the three images
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	
	# VIS image subplot
	im0 = axes[0].imshow(vis_image, cmap='gray', aspect='equal')
	axes[0].set_title(vis_title)
	axes[0].set_xlabel('Columns')
	axes[0].set_ylabel('Rows')
	plt.colorbar(im0, ax=axes[0], label='Intensity')
	
	# NIR1 image subplot
	im1 = axes[1].imshow(nir1_image, cmap='gray', aspect='equal')
	axes[1].set_title(nir1_title)
	axes[1].set_xlabel('Columns')
	axes[1].set_ylabel('Rows')
	plt.colorbar(im1, ax=axes[1], label='Intensity')
	
	# NIR2 image subplot
	im2 = axes[2].imshow(nir2_image, cmap='gray', aspect='equal')
	axes[2].set_title(nir2_title)
	axes[2].set_xlabel('Columns')
	axes[2].set_ylabel('Rows')
	plt.colorbar(im2, ax=axes[2], label='Intensity')
	
	# Plot the provided markers on each subplot
	# Each coordinate list contains three tuples corresponding to (vis, nir1, nir2).
	for i, coord in enumerate(coordinates):
		if len(coord) != 3:
			raise ValueError("Each set of coordinates must contain three tuples: one for VIS, one for NIR1, and one for NIR2.")
		
		# Unpack coordinates for the three images.
		vis_coord, nir1_coord, nir2_coord = coord
		
		# Plot markers on each subplot. Here we use scatter with a red marker with yellow edge.
		axes[0].scatter(vis_coord[0], vis_coord[1], marker='o', s=1, c='red', label='Marker')
		axes[1].scatter(nir1_coord[0], nir1_coord[1], marker='o', s=1, c='red', label='Marker')
		axes[2].scatter(nir2_coord[0], nir2_coord[1], marker='o', s=1, c='red', label='Marker')

		axes[0].text(vis_coord[0] + 5, vis_coord[1] + 5, str(i), color='red', fontsize=8)
		axes[1].text(nir1_coord[0] + 5, nir1_coord[1] + 5, str(i), color='red', fontsize=8)
		axes[2].text(nir2_coord[0] + 5, nir2_coord[1] + 5, str(i), color='red', fontsize=8)

	# To avoid multiple labels in legend if many markers are plotted, display legend for only one occurrence per image.
	# for ax in axes:
	# 	handles, labels = ax.get_legend_handles_labels()
	# 	if handles:
	# 		ax.legend([handles[0]], ['Marker'])
	
	plt.tight_layout()
	plt.show()

# This might be deprecated

# imshow_vis_nir1_nir2_with_coordinates = False
# if imshow_vis_nir1_nir2_with_coordinates:
# 	visualize_vis_nir1_nir2_with_markers(
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_12-04-30_vis_h_vis_ho_600w_2500/acq_000/dc_0_exp_005.bin',
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_14-23-40_nir1_h_nir1_ho_600w_10000/acq_000/dc_1_exp_005.bin',
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_15-25-02_nir2_h_nir2_ho_600w_10000/acq_000/dc_2_exp_005.bin',
# 		[[(480,506),(347,132),(438,397)],[(530,677),(295,298),(481,242)]]
# 	)

def visualize_vis_nir1_nir2_with_markers_and_rectangles(vis_file_path, nir1_file_path, nir2_file_path, 
														coordinates, rectangles,
														shapes=[(1024, 1024), (518, 648), (518, 648)]):
	"""
	Visualizes VIS, NIR1, and NIR2 int16 binary files as images in a single matplotlib figure,
	overlays markers for given coordinates, and draws a rectangle on each image using provided corner coordinates.

	Parameters:
		vis_file_path (str): Path to the VIS binary file.
		nir1_file_path (str): Path to the NIR1 binary file.
		nir2_file_path (str): Path to the NIR2 binary file.
		coordinates (list of list of tuple): A list where each element is a list/tuple containing
			three coordinate pairs:
				[ (vis_x, vis_y), (nir1_x, nir1_y), (nir2_x, nir2_y) ]
			representing marker positions for each image.
		rectangles (tuple or list): A tuple (or list) containing three elements (for VIS, NIR1, NIR2),
			each of which is itself a tuple of four corner coordinates:
				((x1, y1), (x2, y2), (x3, y3), (x4, y4))
			These corners specify the vertices of the rectangle to be drawn on the respective image.
		shapes (list of tuple): A list containing three (height, width) tuples for VIS, NIR1, and NIR2 images respectively.

	Example:
		coordinates = [
			[(100, 150), (50, 75), (60, 80)],        # Marker positions for set 1 (VIS, NIR1, NIR2)
			[(400, 450), (200, 220), (210, 230)]        # Marker positions for set 2
		]

		rectangles = (
			((10, 10), (300, 10), (300, 300), (10, 300)),   # Rectangle for VIS image
			((20, 20), (250, 20), (250, 250), (20, 250)),     # Rectangle for NIR1 image
			((30, 30), (200, 30), (200, 200), (30, 200))      # Rectangle for NIR2 image
		)

		visualize_vis_nir1_nir2_with_markers_and_rectangles(
			'path/to/VIS.bin', 'path/to/NIR1.bin', 'path/to/NIR2.bin',
			coordinates, rectangles
		)
	"""
	# Load the images using the provided load_image function
	vis_image  = load_image(vis_file_path, shapes[0])
	nir1_image = load_image(nir1_file_path, shapes[1])
	nir2_image = load_image(nir2_file_path, shapes[2])
	
	# Get titles using the provided get_title function
	vis_title  = get_title(vis_file_path, format=2)
	nir1_title = get_title(nir1_file_path, format=2)
	nir2_title = get_title(nir2_file_path, format=2)
	
	# Create subplots for the three images
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	
	# VIS image subplot
	im0 = axes[0].imshow(vis_image, cmap='gray', aspect='equal')
	axes[0].set_title(vis_title)
	axes[0].set_xlabel('Columns')
	axes[0].set_ylabel('Rows')
	plt.colorbar(im0, ax=axes[0], label='Intensity')
	
	# NIR1 image subplot
	im1 = axes[1].imshow(nir1_image, cmap='gray', aspect='equal')
	axes[1].set_title(nir1_title)
	axes[1].set_xlabel('Columns')
	axes[1].set_ylabel('Rows')
	plt.colorbar(im1, ax=axes[1], label='Intensity')
	
	# NIR2 image subplot
	im2 = axes[2].imshow(nir2_image, cmap='gray', aspect='equal')
	axes[2].set_title(nir2_title)
	axes[2].set_xlabel('Columns')
	axes[2].set_ylabel('Rows')
	plt.colorbar(im2, ax=axes[2], label='Intensity')
	
	# Plot the provided markers on each subplot
	for i, coord in enumerate(coordinates):
		if len(coord) != 3:
			raise ValueError("Each set of coordinates must contain three tuples: one for VIS, one for NIR1, and one for NIR2.")
		
		# Unpack coordinates for the three images.
		vis_coord, nir1_coord, nir2_coord = coord
		
		# Plot markers on each subplot using scatter.
		axes[0].scatter(vis_coord[0], vis_coord[1], marker='o', s=1, c='red', label='Marker' if i==0 else "")
		axes[1].scatter(nir1_coord[0], nir1_coord[1], marker='o', s=1, c='red', label='Marker' if i==0 else "")
		axes[2].scatter(nir2_coord[0], nir2_coord[1], marker='o', s=1, c='red', label='Marker' if i==0 else "")
		
		# Optionally annotate markers with their index.
		axes[0].text(vis_coord[0] + 5, vis_coord[1] + 5, str(i), color='red', fontsize=8)
		axes[1].text(nir1_coord[0] + 5, nir1_coord[1] + 5, str(i), color='red', fontsize=8)
		axes[2].text(nir2_coord[0] + 5, nir2_coord[1] + 5, str(i), color='red', fontsize=8)
	
	# Draw rectangles on each image using the provided rectangle corner coordinates.
	# Expecting rectangles to be a tuple/list with three items: one per image.
	if len(rectangles) != 3:
		raise ValueError("Rectangles parameter must contain three elements: one for VIS, one for NIR1, and one for NIR2.")
	
	# Unpack rectangle coordinates for each image.
	vis_rect, nir1_rect, nir2_rect = rectangles
	
	# Create a polygon patch for each rectangle.
	vis_patch = Polygon(vis_rect, closed=True, fill=False, edgecolor='blue', linewidth=1, label='Rectangle')
	nir1_patch = Polygon(nir1_rect, closed=True, fill=False, edgecolor='blue', linewidth=1, label='Rectangle')
	nir2_patch = Polygon(nir2_rect, closed=True, fill=False, edgecolor='blue', linewidth=1, label='Rectangle')
	
	# Add the rectangle patches to the corresponding axes.
	axes[0].add_patch(vis_patch)
	axes[1].add_patch(nir1_patch)
	axes[2].add_patch(nir2_patch)
	
	# Optionally add legends (ensuring a single entry per marker and rectangle).
	# for ax in axes:
	#     handles, labels = ax.get_legend_handles_labels()
	#     if handles:
	#         ax.legend(handles[:2], labels[:2])
	
	plt.tight_layout()
	plt.show()

# This might be deprecated

# # Example usage:
# imshow_vis_nir1_nir2_with_coordinates_and_rectangle = False
# if imshow_vis_nir1_nir2_with_coordinates_and_rectangle:
# 	# Define marker coordinates: each list element contains three tuples (VIS, NIR1, NIR2).
# 	markers = [[(480,506),(347,132),(438,397)],[(530,677),(295,298),(481,242)]]
	
# 	# Define rectangle coordinates for the three images.
# 	rectangle_coordinates = (
# 		((445,400), (450,400), (450,405), (445,405)),  # Rectangle for VIS image
# 		((365,45), (370,45), (370,50), (365,50)),       # Rectangle for NIR1 image
# 		((405,480), (410,480), (410,485), (405,485))        # Rectangle for NIR2 image
# 	)
	
# 	visualize_vis_nir1_nir2_with_markers_and_rectangles(
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_12-04-30_vis_h_vis_ho_600w_2500/acq_000/dc_0_exp_005.bin',
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_14-23-40_nir1_h_nir1_ho_600w_10000/acq_000/dc_1_exp_005.bin',
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_15-25-02_nir2_h_nir2_ho_600w_10000/acq_000/dc_2_exp_005.bin',
# 		markers,
# 		rectangle_coordinates
# 	)

def binning_coordinates(coordinates: tuple, bin_size: int):
	"""
	Returns coordinates (X, Y) of a bin_size * bin_size area where the given coordinate is at the center whenever possible.

	Args:
		coordinates (tuple): A tuple (x, y) representing the input coordinate.
		bin_size (int): The size of the bin (assumed to be square).

	Returns:
		tuple: The (X, Y) coordinate of the top-left corner of the bin that centers the input coordinate.
	"""
	x, y = coordinates

	half_bin = bin_size // 2

	top_left_x = x - half_bin
	top_left_y = y - half_bin
	top_right_x = top_left_x + bin_size-1
	top_right_y = y - half_bin
	bottom_left_x = x - half_bin
	bottom_left_y = top_left_y + bin_size-1
	bottom_right_x = bottom_left_x + bin_size-1
	bottom_right_y = top_right_y + bin_size-1

	return (top_left_x, top_left_y), (top_right_x, top_right_y), (bottom_right_x, bottom_right_y), (bottom_left_x, bottom_left_y)

def average_pixel_value_in_rectangle(file_path, shape, rectangle_coordinates):
	"""
	Calculates the average pixel value inside a given rectangle on a binary image.

	Parameters:
		image (numpy.ndarray): A 2D NumPy array representing the binary image.
		rect_coords (tuple or list): A tuple/list containing four corner coordinates of the rectangle.
			Each coordinate is a tuple (x, y). For example:
				((x1, y1), (x2, y2), (x3, y3), (x4, y4))
			It is assumed that the rectangle is axis-aligned.

	Returns:
		float: The average pixel value inside the specified rectangle.
	
	Raises:
		ValueError: If the rectangle coordinates list does not contain exactly four points.
	
	Example:
		# Suppose 'img' is a numpy array loaded from a binary file.
		img = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
		# Define a rectangle with four corner coordinates.
		rect = ((445, 400), (450, 400), (450, 405), (445, 405))
		avg_value = average_pixel_value_in_rectangle(img, rect)
		print("Average pixel value:", avg_value)
	"""
	image  = load_image(file_path, shape)

	if len(rectangle_coordinates) != 4:
		raise ValueError("Rectangle coordinates must contain exactly four points.")
	
	# Extract x and y coordinates separately.
	xs = [point[0] for point in rectangle_coordinates]
	ys = [point[1] for point in rectangle_coordinates]
	
	# For an axis-aligned rectangle, the bounding box is defined by the minimum and maximum coordinates.
	x_min = min(xs)
	x_max = max(xs)
	y_min = min(ys)
	y_max = max(ys)
	
	# Crop the image to the rectangle region. Adding 1 to x_max and y_max to include the boundary.
	# Note: This assumes that the coordinates are within the valid image bounds.
	cropped_region = image[y_min:y_max+1, x_min:x_max+1]
	
	# Calculate and return the average pixel value within the cropped region.
	average_value = np.mean(cropped_region)
	return average_value

def visualize_image_with_rectangle(file_path, shape, rectangle_coordinates):
	image = load_image(file_path, shape)

	title = get_title(file_path, format=2)

	# Create the plot
	plt.figure(figsize=(8, 6))
	plt.imshow(image, cmap='gray', aspect='equal')
	plt.colorbar(label='Intensity')
	plt.title(title)
	plt.xlabel('Columns')
	plt.ylabel('Rows')

	if len(rectangle_coordinates) != 4:
		raise ValueError("Rectangle coordinates parameter must contain four elements")
	patch = Polygon(rectangle_coordinates, closed=True, fill=False, edgecolor='blue', linewidth=1, label='Rectangle')
	
	# Add the rectangle to the figure.
	plt.gca().add_patch(patch)

	plt.tight_layout()
	plt.show()

# This might be deprecated

# print_rectangle_average = False
# if print_rectangle_average:
# 	rectangle_coordinates = ((445,400), (450,400), (450,405), (445,405))
# 	image_shape = (1024, 1024) # (1024, 1024), (518, 648)
# 	avg_pixel_value = average_pixel_value_in_rectangle(
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_12-04-30_vis_h_vis_ho_600w_2500/acq_000/dc_0_exp_005.bin',
# 		image_shape,
# 		rectangle_coordinates
# 	)
# 	print("Average pixel value inside the rectangle:", avg_pixel_value)
# 	visualize_image_with_rectangle(
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_12-04-30_vis_h_vis_ho_600w_2500/acq_000/dc_0_exp_005.bin',
# 		image_shape,
# 		rectangle_coordinates
# 	)

def visualize_pixel_values_from_directory(directory, coordinates, shape):
	"""
	Loads all binary image files in the given directory, extracts the pixel values at the specified 
	coordinates from each image, and creates subplots visualizing the values. Each subplot corresponds 
	to a coordinate (given as a tuple (x, y)), with the x-axis showing file names and the y-axis showing 
	the pixel intensity.

	Parameters:
		directory (str): Path to the directory containing binary image files.
		coordinates (list of tuple): A list of (x, y) pixel coordinates.
			Example: [(10, 15), (200, 300)]
		shape (tuple): The shape (height, width) of the images.
		
	Example:
		directory = '/path/to/binary_images'
		coordinates = [(50, 100), (150, 200)]
		shape = (1024, 1024)
		visualize_pixel_values_from_directory(directory, coordinates, shape)
	"""
	
	# Get list of files in the directory that are assumed to be binary images
	files = sorted([f for f in os.listdir(directory) if f.endswith('.bin')])
	
	if not files:
		raise ValueError("No binary files (.bin) found in the provided directory.")
	
	# Initialize a dictionary to hold the loaded images.
	images = {}
	
	for f in files:
		file_path = os.path.join(directory, f)
		# load_image should be defined elsewhere to read a binary file given its shape.
		image = load_image(file_path, shape)
		images[f] = image

	# Create a figure with as many subplots as there are coordinates.
	num_coords = len(coordinates)
	fig, axes = plt.subplots(num_coords, 1, figsize=(10, 3*num_coords), sharex=True)#squeeze=False)

	if num_coords == 1:
		axes = [axes]

	# For each specified coordinate, extract the pixel value from every image and plot it.
	for idx, coord in enumerate(coordinates):
		# The coordinate is (x, y) but image access is image[y, x]
		pixel_values = []
		for fname in files:
			image = images[fname]
			pixel = image[coord[1], coord[0]]
			pixel_values.append(pixel)
		
		ax = axes[idx]
		# Plot values using markers to emphasize the individual points
		ax.scatter(range(len(files)), pixel_values, c='blue')
		ax.set_xticks(range(len(files)))
		ax.set_xticklabels(files, rotation=30, ha='right')
		ax.set_ylabel("Pixel Value")
		ax.set_title(f"Coordinate {idx}: {coord}")
		ax.grid(True)
		
	plt.tight_layout()
	plt.show()

# This might be deprecated

# plot_pixel_values = False
# if plot_pixel_values:
# 	image_shape = (1024, 1024)
# 	coords = [(480,506), (445,400), (530,677), (295,298), (481,242), (438,397), (347,132), (307,192)]
# 	image_directory = "/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_12-04-30_vis_h_vis_ho_600w_2500/acq_000"
	
# 	visualize_pixel_values_from_directory(image_directory, coords, image_shape)

def parse_PFM_and_FS_default_wl_and_setpoints(file_path):
	df_raw = pd.read_excel(
		file_path,
		sheet_name='FS',
		header=None
	)
	# 2. Parse into a clean DataFrame
	records = []
	current_channel = None
	for _, row in df_raw.iterrows():
		channel = row[0]
		# Detect new spectrometer blocks
		if isinstance(channel, str) and channel.strip() in ['Vis', 'NIR1', 'NIR2']:
			current_channel = channel.strip()
			continue
		# Skip header rows and blank lines
		if row[1] == 'wl' or pd.isna(row[1]):
			continue
		# Append data record
		records.append({
			'channel': current_channel.upper(),
			'wl': round(row[1]),
			'lo': round(row[2]),
			'ho': round(row[3])
		})

	df_fs = pd.DataFrame(records)

	# 4. Build a lookup dictionary for reverse queries: given spectrometer, order, and setpoint, get wl
	wl_lookup = {}
	for channel in df_fs['channel'].unique():
		sub = df_fs[df_fs['channel'] == channel]
		wl_lookup[channel] = {
			'lo': dict(zip(sub['lo'], sub['wl'])),
			'ho': dict(zip(sub['ho'], sub['wl']))
		}

	return df_fs, wl_lookup

# df_fs, wl_lookup = parse_PFM_and_FS_default_wl_and_setpoints('/home/sysa/HERA/test_data/PFM_and_FS_default_wl_and_setpoints.xlsx')
# print(wl_lookup)

def reflectance(image_value, calibration_square_average):
	return image_value / calibration_square_average * 0.2

def get_rectangle_averages(bin_files, acquisition_path, rectangle_coordinates, naming_key, height, width):
	rectangle_averages = {}
	if len(rectangle_coordinates) != 4:
		raise ValueError("Rectangle coordinates must contain exactly four points.")
	xs = [point[0] for point in rectangle_coordinates]
	ys = [point[1] for point in rectangle_coordinates]
	x_min = min(xs)
	x_max = max(xs)
	y_min = min(ys)
	y_max = max(ys)
	for j, bin_file in enumerate(bin_files):
		file_path = os.path.join(acquisition_path, bin_file)
		with open(file_path, 'rb') as file:
			bin_data = file.read()
			img_array = np.frombuffer(bin_data, dtype=np.uint16)
			img_array = img_array.reshape((height, width))
			cropped_region = img_array[y_min:y_max+1, x_min:x_max+1]
			average_value = np.mean(cropped_region)
			rectangle_averages[os.path.join(naming_key, bin_file)] = average_value
	return rectangle_averages

def get_rectangle_averages_from_different_file(asteroid_bin_files, calibration_bin_files, calibration_acquisition_path, rectangle_coordinates, naming_key, height, width):
	rectangle_averages = {}
	if len(rectangle_coordinates) != 4:
		raise ValueError("Rectangle coordinates must contain exactly four points.")
	xs = [point[0] for point in rectangle_coordinates]
	ys = [point[1] for point in rectangle_coordinates]
	x_min = min(xs)
	x_max = max(xs)
	y_min = min(ys)
	y_max = max(ys)
	for asteroid_bin_file, calibration_bin_file in zip(asteroid_bin_files, calibration_bin_files):
		file_path = os.path.join(calibration_acquisition_path, calibration_bin_file)
		with open(file_path, 'rb') as file:
			bin_data = file.read()
			img_array = np.frombuffer(bin_data, dtype=np.uint16)
			img_array = img_array.reshape((height, width))
			cropped_region = img_array[y_min:y_max+1, x_min:x_max+1]
			average_value = np.mean(cropped_region)
			rectangle_averages[os.path.join(naming_key, asteroid_bin_file)] = average_value
	return rectangle_averages

def get_calibration_pixel_values_from_different_file(asteroid_bin_files, calibration_bin_files, calibration_acquisition_path, calibration_pixel_coordinates, naming_key, height, width):
	binning = 3 # Set calibration binning size, example: 3 for 3x3 or None for single pixel
	calibration_pixel_values = {}
	for asteroid_bin_file, calibration_bin_file in zip(asteroid_bin_files, calibration_bin_files):
		calibration_file_pixel_values = []
		file_path = os.path.join(calibration_acquisition_path, calibration_bin_file)
		with open(file_path, 'rb') as file:
			bin_data = file.read()
			img_array = np.frombuffer(bin_data, dtype=np.uint16)
			img_array = img_array.reshape((height, width))
			for pixel in calibration_pixel_coordinates:
				if binning:
					(top_left_x, top_left_y), (top_right_x, top_right_y), (bottom_right_x, bottom_right_y), (bottom_left_x, bottom_left_y) = binning_coordinates(pixel, binning)
					cropped_region = img_array[top_left_y:bottom_left_y+1, top_left_x:top_right_x+1]
					average_value = np.mean(cropped_region)
					calibration_file_pixel_values.append(average_value)
				else:
					calibration_file_pixel_values.append(img_array[pixel[1], pixel[0]])
			calibration_pixel_values[os.path.join(naming_key, asteroid_bin_file)] = calibration_file_pixel_values
	return calibration_pixel_values

# this is deprecated!

# def process_directory(
# 		parent_directory: str,
# 		main_directory: str, # vis, nir1, nir2
# 		output_directory: str,
# 		pixel_coordinates: dict,
# 		rectangle_coordinates: dict,
# 		test_mode: bool = False,
# 	):
# 	"""
# 	- Visualize all coordinates on each channel
# 	- Save rectangle averages of every binary file
# 	- Make a scatterplot of each coordinate comparing the value per binary file
# 	"""
# 	if not test_mode:
# 		os.makedirs(output_directory, exist_ok=True)

# 	acquisition_path = os.path.join(parent_directory, main_directory, "acq_000/")
# 	calib_path = os.path.join(parent_directory, main_directory, "meta/calib.json")
# 	config_path = os.path.join(parent_directory, main_directory, "meta/config.json")
# 	channel = read_channel(calib_path)

# 	if channel == 'VIS':
# 		height = 1024
# 		width = 1024
# 	elif channel == 'NIR1' or channel == 'NIR2':
# 		height = 518
# 		width = 648
# 	rectangle_coordinates = rectangle_coordinates[channel]
# 	channel_pixel_coordinates = pixel_coordinates[channel]

# 	if '_lo_' in main_directory:
# 		order = 'lo'
# 	elif '_ho_' in main_directory:
# 		order = 'ho'
# 	else:
# 		raise ValueError("ASPECT_TESTS_25_7_non_dark must contain either '_lo_' or '_ho_'")

# 	df_fs, wl_lookup = parse_PFM_and_FS_default_wl_and_setpoints('/home/sysa/HERA/test_data/PFM_and_FS_default_wl_and_setpoints.xlsx')

# 	acquisition_wavelengths = []
# 	task_file_name = channel.lower() + 'TaskFile'
# 	with open(config_path, 'r') as file:
# 		data = json.load(file)
# 	task_file = data[task_file_name]
# 	task_file_sequences = [task_file[i:i+8] for i in range(0, len(task_file), 8)]
# 	for sequence in task_file_sequences:
# 		acquisition_wavelengths.append(wl_lookup[channel][order][sequence[1]])

# 	bin_files = sorted([f for f in os.listdir(acquisition_path) if f.endswith('.bin')])

# 	# Rectangles
# 	rectangle_averages = get_rectangle_averages(
# 		bin_files,
# 		acquisition_path,
# 		rectangle_coordinates,
# 		main_directory,
# 		height,
# 		width
# 	)

# 	num_coordinates = len(channel_pixel_coordinates)
# 	fig, axes = plt.subplots(num_coordinates, 1, figsize=(10, 3*num_coordinates), sharex=True)
# 	if num_coordinates == 1:
# 		axes = [axes]

# 	for i, coordinates in enumerate(channel_pixel_coordinates):
# 		pixel_values = []
# 		for j, bin_file in enumerate(bin_files):
# 			file_path = os.path.join(acquisition_path, bin_file)
# 			with open(file_path, 'rb') as file:
# 				bin_data = file.read()
# 				img_array = np.frombuffer(bin_data, dtype=np.uint16)
# 				img_array = img_array.reshape((height, width))
# 				pixel = img_array[coordinates[1], coordinates[0]]
# 				pixel_values.append(pixel)
# 				rectangle_average = rectangle_averages[os.path.join(main_directory, bin_file)]
# 				pixel_reflectance = reflectance(pixel, rectangle_average)

# 		ax = axes[i]
# 		# Plot values using markers to emphasize the individual points
# 		ax.scatter(range(len(bin_files)), pixel_values, c='blue')
# 		ax.set_xticks(range(len(bin_files)))
# 		ax.set_xticklabels(bin_files, rotation=30, ha='right')
# 		ax.set_ylabel("Pixel Value")
# 		ax.set_title(f"Coordinate {i}: {coordinates}")
# 		ax.grid(True)
		
# 	plt.tight_layout()
# 	# plt.show()
# 	plot_path = os.path.join(output_directory, f"{main_directory}_pixel_values.png")
# 	if not test_mode:
# 		plt.savefig(plot_path)
# 	plt.close(fig)

# 	# Reflectances
# 	unique_idxs = [
# 		i for i in range(len(acquisition_wavelengths)) if i == len(acquisition_wavelengths)-1 or acquisition_wavelengths[i] != acquisition_wavelengths[i+1]
# 	]
# 	filtered_wls      = [acquisition_wavelengths[i] for i in unique_idxs]
# 	filtered_binfiles = [bin_files[i] for i in unique_idxs]
# 	x_labels = [f"{wl} wl ({bf})" for wl, bf in zip(filtered_wls, filtered_binfiles)]
	
# 	fig_ref, axes_ref = plt.subplots(num_coordinates, 1, figsize=(10, 3*num_coordinates), sharex=True)
# 	if num_coordinates == 1:
# 		axes_ref = [axes_ref]
# 	for i, (x, y) in enumerate(channel_pixel_coordinates):
# 		pixel_reflectances = []
# 		for wl, bin_file in zip(filtered_wls, filtered_binfiles):
# 			path = os.path.join(acquisition_path, bin_file)
# 			with open(path, 'rb') as f:
# 				buf = f.read()
# 			img = np.frombuffer(buf, dtype=np.uint16).reshape((height, width))
# 			pix = img[y, x]
# 			pixel_reflectance = reflectance(pix, rectangle_averages[os.path.join(main_directory, bin_file)])
# 			pixel_reflectances.append(pixel_reflectance)
# 		axr = axes_ref[i]
# 		axr.scatter(range(len(x_labels)), pixel_reflectances)
# 		axr.set_xticks(range(len(x_labels)))
# 		axr.set_xticklabels(x_labels, rotation=30, ha='right')
# 		axr.set_ylabel("Pixel Reflectance")
# 		axr.set_title(f"Coordinate {i}: {(x, y)}")
# 		axr.grid(True)
# 	plt.tight_layout()
# 	# plt.show()
# 	out_ref = os.path.join(output_directory, f"{main_directory}_pixel_reflectances.png")
# 	if not test_mode:
# 		fig_ref.savefig(out_ref)
# 	plt.close(fig_ref)

# 	return rectangle_averages

def query_std_from_excel(target_directory: str, sheet_name: str = 'Sheet1'):
	channel = target_directory.split('_')[2]
	exposure = target_directory.split('_')[7]
	df = pd.read_excel('/home/sysa/HERA/test_data/dark(after)_filtered_histogram_stats.xlsx', sheet_name=sheet_name)
	match = df[
		df['File'].astype(str).str.contains(channel) &
		# df['File'].astype(str).str.contains(order) &
		# df['File'].astype(str).str.contains(target_bin) &
		df['File'].astype(str).str.contains(exposure)
	]
	if match.empty:
		raise ValueError(f"No matching entry found for channel: {channel} and exposure: {exposure}")
	standard_deviation = match['Standard Deviation'].mean()
	return standard_deviation

def ESA_meteorite_average_std_from_excel(target_directory: str, sheet_name: str = 'averages'):
	df = pd.read_excel('/home/sysa/HERA/test_data/ESA_meteorites_dark_filtered_histogram_stats.xlsx', sheet_name=sheet_name)
	match = df[df['File'] == target_directory]

	if match.empty:
		raise ValueError(f"File '{target_directory}' not found in sheet '{sheet_name}'.")

	return float(match['Standard Deviation'].iloc[0])

def calculate_signal_to_noise_ratio(pixel_value, standard_deviation):
	return pixel_value / standard_deviation

def process_directory(
		metadata_parent_directory: str,
		parent_directory: str,
		key,
		lo_vis_directory: str,
		lo_nir1_directory: str,
		lo_nir2_directory: str,
		ho_vis_directory: str,
		ho_nir1_directory: str,
		ho_nir2_directory: str,
		output_directory: str,
		pixel_coordinates: dict,
		rectangle_coordinates: dict,
		test_mode: bool = False,
	):
	if not test_mode:
		os.makedirs(output_directory, exist_ok=True)

	if lo_vis_directory:
		lo_vis_acquisition_path = os.path.join(parent_directory, lo_vis_directory)
		lo_vis_calib_path = os.path.join(metadata_parent_directory, lo_vis_directory, "meta/calib.json")
		lo_vis_config_path = os.path.join(metadata_parent_directory, lo_vis_directory, "meta/config.json")
		lo_vis_channel = read_channel(lo_vis_calib_path)
	lo_nir1_acquisition_path = os.path.join(parent_directory, lo_nir1_directory)
	lo_nir1_calib_path = os.path.join(metadata_parent_directory, lo_nir1_directory, "meta/calib.json")
	lo_nir1_config_path = os.path.join(metadata_parent_directory, lo_nir1_directory, "meta/config.json")
	lo_nir1_channel = read_channel(lo_nir1_calib_path)
	lo_nir2_acquisition_path = os.path.join(parent_directory, lo_nir2_directory)
	lo_nir2_calib_path = os.path.join(metadata_parent_directory, lo_nir2_directory, "meta/calib.json")
	lo_nir2_config_path = os.path.join(metadata_parent_directory, lo_nir2_directory, "meta/config.json")
	lo_nir2_channel = read_channel(lo_nir2_calib_path)
	ho_vis_acquisition_path = os.path.join(parent_directory, ho_vis_directory)
	ho_vis_calib_path = os.path.join(metadata_parent_directory, ho_vis_directory, "meta/calib.json")
	ho_vis_config_path = os.path.join(metadata_parent_directory, ho_vis_directory, "meta/config.json")
	ho_vis_channel = read_channel(ho_vis_calib_path)
	ho_nir1_acquisition_path = os.path.join(parent_directory, ho_nir1_directory)
	ho_nir1_calib_path = os.path.join(metadata_parent_directory, ho_nir1_directory, "meta/calib.json")
	ho_nir1_config_path = os.path.join(metadata_parent_directory, ho_nir1_directory, "meta/config.json")
	ho_nir1_channel = read_channel(ho_nir1_calib_path)
	ho_nir2_acquisition_path = os.path.join(parent_directory, ho_nir2_directory)
	ho_nir2_calib_path = os.path.join(metadata_parent_directory, ho_nir2_directory, "meta/calib.json")
	ho_nir2_config_path = os.path.join(metadata_parent_directory, ho_nir2_directory, "meta/config.json")
	ho_nir2_channel = read_channel(ho_nir2_calib_path)

	vis_height = 1024
	vis_width = 1024
	nir_height = 506#518
	nir_width = 636#648

	vis_rectangle_coordinates = rectangle_coordinates[ho_vis_channel]#
	vis_channel_pixel_coordinates = pixel_coordinates[ho_vis_channel]#
	nir1_rectangle_coordinates = rectangle_coordinates[lo_nir1_channel]
	nir1_channel_pixel_coordinates = pixel_coordinates[lo_nir1_channel]
	nir2_rectangle_coordinates = rectangle_coordinates[lo_nir2_channel]
	nir2_channel_pixel_coordinates = pixel_coordinates[lo_nir2_channel]

	df_fs, wl_lookup = parse_PFM_and_FS_default_wl_and_setpoints('/home/sysa/HERA/test_data/PFM_and_FS_default_wl_and_setpoints.xlsx')

	if lo_vis_directory:
		lo_vis_acquisition_wavelengths = []
		task_file_name = lo_vis_channel.lower() + 'TaskFile'
		with open(lo_vis_config_path, 'r') as file:
			data = json.load(file)
		task_file = data[task_file_name]
		task_file_sequences = [task_file[i:i+8] for i in range(0, len(task_file), 8)]
		for sequence in task_file_sequences:
			lo_vis_acquisition_wavelengths.append(wl_lookup[lo_vis_channel]['lo'][sequence[1]])
	
	ho_vis_acquisition_wavelengths = []
	task_file_name = ho_vis_channel.lower() + 'TaskFile'
	with open(ho_vis_config_path, 'r') as file:
		data = json.load(file)
	task_file = data[task_file_name]
	task_file_sequences = [task_file[i:i+8] for i in range(0, len(task_file), 8)]
	for sequence in task_file_sequences:
		ho_vis_acquisition_wavelengths.append(wl_lookup[ho_vis_channel]['ho'][sequence[1]])

	lo_nir1_acquisition_wavelengths = []
	task_file_name = lo_nir1_channel.lower() + 'TaskFile'
	with open(lo_nir1_config_path, 'r') as file:
		data = json.load(file)
	task_file = data[task_file_name]
	task_file_sequences = [task_file[i:i+8] for i in range(0, len(task_file), 8)]
	for sequence in task_file_sequences:
		lo_nir1_acquisition_wavelengths.append(wl_lookup[lo_nir1_channel]['lo'][sequence[1]])

	ho_nir1_acquisition_wavelengths = []
	task_file_name = ho_nir1_channel.lower() + 'TaskFile'
	with open(ho_nir1_config_path, 'r') as file:
		data = json.load(file)
	task_file = data[task_file_name]
	task_file_sequences = [task_file[i:i+8] for i in range(0, len(task_file), 8)]
	for sequence in task_file_sequences:
		ho_nir1_acquisition_wavelengths.append(wl_lookup[ho_nir1_channel]['ho'][sequence[1]])

	lo_nir2_acquisition_wavelengths = []
	task_file_name = lo_nir2_channel.lower() + 'TaskFile'
	with open(lo_nir2_config_path, 'r') as file:
		data = json.load(file)
	task_file = data[task_file_name]
	task_file_sequences = [task_file[i:i+8] for i in range(0, len(task_file), 8)]
	for sequence in task_file_sequences:
		lo_nir2_acquisition_wavelengths.append(wl_lookup[lo_nir2_channel]['lo'][sequence[1]])

	ho_nir2_acquisition_wavelengths = []
	task_file_name = ho_nir2_channel.lower() + 'TaskFile'
	with open(ho_nir2_config_path, 'r') as file:
		data = json.load(file)
	task_file = data[task_file_name]
	task_file_sequences = [task_file[i:i+8] for i in range(0, len(task_file), 8)]
	for sequence in task_file_sequences:
		ho_nir2_acquisition_wavelengths.append(wl_lookup[ho_nir2_channel]['ho'][sequence[1]])

	if lo_vis_directory:
		lo_vis_bin_files = sorted([f for f in os.listdir(lo_vis_acquisition_path) if f.endswith('.bin')])
	lo_nir1_bin_files = sorted([f for f in os.listdir(lo_nir1_acquisition_path) if f.endswith('.bin')])
	lo_nir2_bin_files = sorted([f for f in os.listdir(lo_nir2_acquisition_path) if f.endswith('.bin')])
	ho_vis_bin_files = sorted([f for f in os.listdir(ho_vis_acquisition_path) if f.endswith('.bin')])
	ho_nir1_bin_files = sorted([f for f in os.listdir(ho_nir1_acquisition_path) if f.endswith('.bin')])
	ho_nir2_bin_files = sorted([f for f in os.listdir(ho_nir2_acquisition_path) if f.endswith('.bin')])


	# Rectangles
	if lo_vis_directory:
		lo_vis_rectangle_averages = get_rectangle_averages(
			lo_vis_bin_files,
			lo_vis_acquisition_path,
			vis_rectangle_coordinates,
			lo_vis_directory,
			vis_height,
			vis_width
		)
	lo_nir1_rectangle_averages = get_rectangle_averages(
		lo_nir1_bin_files,
		lo_nir1_acquisition_path,
		nir1_rectangle_coordinates,
		lo_nir1_directory,
		nir_height,
		nir_width
	)
	lo_nir2_rectangle_averages = get_rectangle_averages(
		lo_nir2_bin_files,
		lo_nir2_acquisition_path,
		nir2_rectangle_coordinates,
		lo_nir2_directory,
		nir_height,
		nir_width
	)
	ho_vis_rectangle_averages = get_rectangle_averages(
		ho_vis_bin_files,
		ho_vis_acquisition_path,
		vis_rectangle_coordinates,
		ho_vis_directory,
		vis_height,
		vis_width
	)
	ho_nir1_rectangle_averages = get_rectangle_averages(
		ho_nir1_bin_files,
		ho_nir1_acquisition_path,
		nir1_rectangle_coordinates,
		ho_nir1_directory,
		nir_height,
		nir_width
	)
	ho_nir2_rectangle_averages = get_rectangle_averages(
		ho_nir2_bin_files,
		ho_nir2_acquisition_path,
		nir2_rectangle_coordinates,
		ho_nir2_directory,
		nir_height,
		nir_width
	)

	# Pixel Values
	num_coordinates = len(vis_channel_pixel_coordinates)
	fig, axes = plt.subplots(num_coordinates, 2, figsize=(20, 4*num_coordinates), sharex=True)
	if num_coordinates == 1:
		axes = [axes]
	min_y_value, max_y_value = float('inf'), float('-inf')
	all_coordinates_lo_pixel_values = []
	all_coordinates_lo_binning_pixel_values = []
	all_coordinates_ho_pixel_values = []
	all_coordinates_ho_binning_pixel_values = []

	# Low Order
	for i, (vis_coordinates, nir1_coordinates, nir2_coordinates) in enumerate(zip(vis_channel_pixel_coordinates, nir1_channel_pixel_coordinates, nir2_channel_pixel_coordinates)):
		vis_pixel_values = []
		binning_vis_pixel_values = []
		if lo_vis_directory:
			for j, bin_file in enumerate(lo_vis_bin_files):
				file_path = os.path.join(lo_vis_acquisition_path, bin_file)
				with open(file_path, 'rb') as file:
					bin_data = file.read()
					img_array = np.frombuffer(bin_data, dtype=np.uint16)
					img_array = img_array.reshape((vis_height, vis_width))
					pixel1 = img_array[vis_coordinates[1], vis_coordinates[0]]
					vis_pixel_values.append(pixel1)
					pixel2 = img_array[vis_coordinates[1], vis_coordinates[0]+1]
					pixel3 = img_array[vis_coordinates[1]+1, vis_coordinates[0]]
					pixel4 = img_array[vis_coordinates[1]+1, vis_coordinates[0]+1]
					average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
					binning_vis_pixel_values.append(average_pixel)
		
		nir1_pixel_values = []
		binning_nir1_pixel_values = []
		for j, bin_file in enumerate(lo_nir1_bin_files):
			file_path = os.path.join(lo_nir1_acquisition_path, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((nir_height, nir_width))
				pixel1 = img_array[nir1_coordinates[1], nir1_coordinates[0]]
				nir1_pixel_values.append(pixel1)
				pixel2 = img_array[nir1_coordinates[1], nir1_coordinates[0]+1]
				pixel3 = img_array[nir1_coordinates[1]+1, nir1_coordinates[0]]
				pixel4 = img_array[nir1_coordinates[1]+1, nir1_coordinates[0]+1]
				average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
				binning_nir1_pixel_values.append(average_pixel)

		nir2_pixel_values = []
		binning_nir2_pixel_values = []
		for j, bin_file in enumerate(lo_nir2_bin_files):
			file_path = os.path.join(lo_nir2_acquisition_path, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((nir_height, nir_width))
				pixel1 = img_array[nir2_coordinates[1], nir2_coordinates[0]]
				nir2_pixel_values.append(pixel1)
				pixel2 = img_array[nir2_coordinates[1], nir2_coordinates[0]+1]
				pixel3 = img_array[nir2_coordinates[1]+1, nir2_coordinates[0]]
				pixel4 = img_array[nir2_coordinates[1]+1, nir2_coordinates[0]+1]
				average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
				binning_nir2_pixel_values.append(average_pixel)

		ax = axes[i, 0]
		# Plot values using markers to emphasize the individual points
		lo_pixel_values = vis_pixel_values+nir1_pixel_values+nir2_pixel_values
		lo_binning_pixel_values = binning_vis_pixel_values+binning_nir1_pixel_values+binning_nir2_pixel_values
		all_coordinates_lo_pixel_values.append(lo_pixel_values)
		all_coordinates_lo_binning_pixel_values.append(lo_binning_pixel_values)
		if lo_vis_directory:
			ax.plot(range(len(lo_vis_bin_files)+len(lo_nir1_bin_files)+len(lo_nir2_bin_files)), lo_binning_pixel_values, c='blue')
			ax.set_xticks(range(len(lo_vis_bin_files)+len(lo_nir1_bin_files)+len(lo_nir2_bin_files)))
			ax.set_xticklabels(lo_vis_bin_files+lo_nir1_bin_files+lo_nir2_bin_files, rotation=40, ha='right', fontsize=12)
		else:
			ax.plot(range(len([0,0,0,0,0,0,0])+len(lo_nir1_bin_files)+len(lo_nir2_bin_files)), [0,0,0,0,0,0,0]+lo_binning_pixel_values, c='blue')
			ax.set_xticks(range(len([0,0,0,0,0,0,0])+len(lo_nir1_bin_files)+len(lo_nir2_bin_files)))
			ax.set_xticklabels([0,0,0,0,0,0,0]+lo_nir1_bin_files+lo_nir2_bin_files, rotation=40, ha='right', fontsize=12)
		ax.set_ylabel("Pixel Value")
		ax.set_title(f"Low Order, Coordinate {i}")
		ax.grid(True, axis='y')
		if min(lo_binning_pixel_values) < min_y_value:
			min_y_value = min(lo_binning_pixel_values)
		if max(lo_binning_pixel_values) > max_y_value:
			max_y_value = max(lo_binning_pixel_values)

	# High Order
	for i, (vis_coordinates, nir1_coordinates, nir2_coordinates) in enumerate(zip(vis_channel_pixel_coordinates, nir1_channel_pixel_coordinates, nir2_channel_pixel_coordinates)):
		vis_pixel_values = []
		binning_vis_pixel_values = []
		for j, bin_file in enumerate(ho_vis_bin_files):
			file_path = os.path.join(ho_vis_acquisition_path, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((vis_height, vis_width))
				pixel1 = img_array[vis_coordinates[1], vis_coordinates[0]]
				vis_pixel_values.append(pixel1)
				pixel2 = img_array[vis_coordinates[1], vis_coordinates[0]+1]
				pixel3 = img_array[vis_coordinates[1]+1, vis_coordinates[0]]
				pixel4 = img_array[vis_coordinates[1]+1, vis_coordinates[0]+1]
				average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
				binning_vis_pixel_values.append(average_pixel)
		
		nir1_pixel_values = []
		binning_nir1_pixel_values = []
		for j, bin_file in enumerate(ho_nir1_bin_files):
			file_path = os.path.join(ho_nir1_acquisition_path, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((nir_height, nir_width))
				pixel1 = img_array[nir1_coordinates[1], nir1_coordinates[0]]
				nir1_pixel_values.append(pixel1)
				pixel2 = img_array[nir1_coordinates[1], nir1_coordinates[0]+1]
				pixel3 = img_array[nir1_coordinates[1]+1, nir1_coordinates[0]]
				pixel4 = img_array[nir1_coordinates[1]+1, nir1_coordinates[0]+1]
				average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
				binning_nir1_pixel_values.append(average_pixel)

		nir2_pixel_values = []
		binning_nir2_pixel_values = []
		for j, bin_file in enumerate(ho_nir2_bin_files):
			file_path = os.path.join(ho_nir2_acquisition_path, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((nir_height, nir_width))
				pixel1 = img_array[nir2_coordinates[1], nir2_coordinates[0]]
				nir2_pixel_values.append(pixel1)
				pixel2 = img_array[nir2_coordinates[1], nir2_coordinates[0]+1]
				pixel3 = img_array[nir2_coordinates[1]+1, nir2_coordinates[0]]
				pixel4 = img_array[nir2_coordinates[1]+1, nir2_coordinates[0]+1]
				average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
				binning_nir2_pixel_values.append(average_pixel)

		ax = axes[i, 1]
		# Plot values using markers to emphasize the individual points
		ho_pixel_values = vis_pixel_values+nir1_pixel_values+nir2_pixel_values
		ho_binning_pixel_values = binning_vis_pixel_values+binning_nir1_pixel_values+binning_nir2_pixel_values
		all_coordinates_ho_pixel_values.append(ho_pixel_values)
		all_coordinates_ho_binning_pixel_values.append(ho_binning_pixel_values)
		ax.plot(range(len(ho_vis_bin_files)+len(ho_nir1_bin_files)+len(ho_nir2_bin_files)), ho_binning_pixel_values, c='blue')
		ax.set_xticks(range(len(ho_vis_bin_files)+len(ho_nir1_bin_files)+len(ho_nir2_bin_files)))
		ax.set_xticklabels(ho_vis_bin_files+ho_nir1_bin_files+ho_nir2_bin_files, rotation=40, ha='right', fontsize=12)
		ax.set_ylabel("Pixel Value")
		ax.set_title(f"High Order, Coordinate {i}")
		ax.grid(True, axis='y')
		if min(ho_binning_pixel_values) < min_y_value:
			min_y_value = min(ho_binning_pixel_values)
		if max(ho_binning_pixel_values) > max_y_value:
			max_y_value = max(ho_binning_pixel_values)
		
	for axr in axes.flatten():
		axr.set_ylim(0, max_y_value) # Optionally min_y_value instead of 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"{key}_pixel_values.png")
	if test_mode:
		pass
		# plt.show()
	else:
		plt.savefig(plot_path)
	plt.close(fig)
	
	# Reflectances
	if lo_vis_directory:
		lo_vis_unique_idxs = [
			i for i in range(len(lo_vis_acquisition_wavelengths)) if i == len(lo_vis_acquisition_wavelengths)-1 or lo_vis_acquisition_wavelengths[i] != lo_vis_acquisition_wavelengths[i+1]
		]
	lo_nir1_unique_idxs = [
		i for i in range(len(lo_nir1_acquisition_wavelengths)) if i == len(lo_nir1_acquisition_wavelengths)-1 or lo_nir1_acquisition_wavelengths[i] != lo_nir1_acquisition_wavelengths[i+1]
	]
	lo_nir2_unique_idxs = [
		i for i in range(len(lo_nir2_acquisition_wavelengths)) if i == len(lo_nir2_acquisition_wavelengths)-1 or lo_nir2_acquisition_wavelengths[i] != lo_nir2_acquisition_wavelengths[i+1]
	]
	ho_vis_unique_idxs = [
		i for i in range(len(ho_vis_acquisition_wavelengths)) if i == len(ho_vis_acquisition_wavelengths)-1 or ho_vis_acquisition_wavelengths[i] != ho_vis_acquisition_wavelengths[i+1]
	]
	ho_nir1_unique_idxs = [
		i for i in range(len(ho_nir1_acquisition_wavelengths)) if i == len(ho_nir1_acquisition_wavelengths)-1 or ho_nir1_acquisition_wavelengths[i] != ho_nir1_acquisition_wavelengths[i+1]
	]
	ho_nir2_unique_idxs = [
		i for i in range(len(ho_nir2_acquisition_wavelengths)) if i == len(ho_nir2_acquisition_wavelengths)-1 or ho_nir2_acquisition_wavelengths[i] != ho_nir2_acquisition_wavelengths[i+1]
	]
	if lo_vis_directory:
		lo_vis_filtered_wls = [lo_vis_acquisition_wavelengths[i] for i in lo_vis_unique_idxs]
		lo_vis_filtered_binfiles = [lo_vis_bin_files[i] for i in lo_vis_unique_idxs]
		lo_vis_x_labels = [f"{wl}" for wl, bf in zip(lo_vis_filtered_wls, lo_vis_filtered_binfiles)]
	lo_nir1_filtered_wls = [lo_nir1_acquisition_wavelengths[i] for i in lo_nir1_unique_idxs]
	lo_nir2_filtered_wls = [lo_nir2_acquisition_wavelengths[i] for i in lo_nir2_unique_idxs]
	lo_nir1_filtered_binfiles = [lo_nir1_bin_files[i] for i in lo_nir1_unique_idxs]
	lo_nir2_filtered_binfiles = [lo_nir2_bin_files[i] for i in lo_nir2_unique_idxs]
	lo_nir1_x_labels = [f"{wl}" for wl, bf in zip(lo_nir1_filtered_wls, lo_nir1_filtered_binfiles)]
	lo_nir2_x_labels = [f"{wl}" for wl, bf in zip(lo_nir2_filtered_wls, lo_nir2_filtered_binfiles)]
	ho_vis_filtered_wls = [ho_vis_acquisition_wavelengths[i] for i in ho_vis_unique_idxs]
	ho_nir1_filtered_wls = [ho_nir1_acquisition_wavelengths[i] for i in ho_nir1_unique_idxs]
	ho_nir2_filtered_wls = [ho_nir2_acquisition_wavelengths[i] for i in ho_nir2_unique_idxs]
	ho_vis_filtered_binfiles = [ho_vis_bin_files[i] for i in ho_vis_unique_idxs]
	ho_nir1_filtered_binfiles = [ho_nir1_bin_files[i] for i in ho_nir1_unique_idxs]
	ho_nir2_filtered_binfiles = [ho_nir2_bin_files[i] for i in ho_nir2_unique_idxs]
	ho_vis_x_labels = [f"{wl}" for wl, bf in zip(ho_vis_filtered_wls, ho_vis_filtered_binfiles)]
	ho_nir1_x_labels = [f"{wl}" for wl, bf in zip(ho_nir1_filtered_wls, ho_nir1_filtered_binfiles)]
	ho_nir2_x_labels = [f"{wl}" for wl, bf in zip(ho_nir2_filtered_wls, ho_nir2_filtered_binfiles)]
	
	fig_ref, axes_ref = plt.subplots(num_coordinates, 2, figsize=(20, 4*num_coordinates), sharex=True)
	min_y_value, max_y_value = float('inf'), float('-inf')
	if num_coordinates == 1:
		axes_ref = [axes_ref]

	# Low Order
	# if lo_vis_directory:
	# 	all_x_labels = lo_vis_x_labels + lo_nir1_x_labels + lo_nir2_x_labels
	# else:
	# 	all_x_labels = [0,0,0,0,0,0,0] + lo_nir1_x_labels + lo_nir2_x_labels
	all_x_labels = ho_vis_x_labels + ho_nir1_x_labels + ho_nir2_x_labels
	n_wls = len(all_x_labels)
	n_coords = len(vis_channel_pixel_coordinates)
	all_coordinates_lo_pixel_reflectances = []
	all_coordinates_lo_binning_pixel_reflectances = []
	all_coordinates_ho_pixel_reflectances = []
	all_coordinates_ho_binning_pixel_reflectances = []
	reflectance_matrix = np.zeros((n_coords*2, n_wls), dtype=float)
	binning_reflectance_matrix = np.zeros((n_coords*2, n_wls), dtype=float)
	for i, ((vis_x, vis_y), (nir1_x, nir1_y), (nir2_x, nir2_y)) in enumerate(zip(vis_channel_pixel_coordinates, nir1_channel_pixel_coordinates, nir2_channel_pixel_coordinates)):
		pixel_reflectances = []
		binning_pixel_reflectances = []
		if lo_vis_directory:
			for wl, bin_file in zip(lo_vis_filtered_wls, lo_vis_filtered_binfiles):
				path = os.path.join(lo_vis_acquisition_path, bin_file)
				with open(path, 'rb') as f:
					buf = f.read()
				img = np.frombuffer(buf, dtype=np.uint16).reshape((vis_height, vis_width))
				pix1 = img[vis_y, vis_x]
				pix2 = img[vis_y+1, vis_x]
				pix3 = img[vis_y, vis_x+1]
				pix4 = img[vis_y+1, vis_x+1]
				average_pix = (pix1 + pix2 + pix3 + pix4) / 4
				pixel_reflectance = reflectance(pix1, lo_vis_rectangle_averages[os.path.join(lo_vis_directory, bin_file)])
				binning_pixel_reflectance = reflectance(average_pix, lo_vis_rectangle_averages[os.path.join(lo_vis_directory, bin_file)])
				binning_pixel_reflectances.append(binning_pixel_reflectance)
				pixel_reflectances.append(pixel_reflectance)
		for wl, bin_file in zip(lo_nir1_filtered_wls, lo_nir1_filtered_binfiles):
			path = os.path.join(lo_nir1_acquisition_path, bin_file)
			with open(path, 'rb') as f:
				buf = f.read()
			img = np.frombuffer(buf, dtype=np.uint16).reshape((nir_height, nir_width))
			pix1 = img[nir1_y, nir1_x]
			pix2 = img[nir1_y+1, nir1_x]
			pix3 = img[nir1_y, nir1_x+1]
			pix4 = img[nir1_y+1, nir1_x+1]
			average_pix = (pix1 + pix2 + pix3 + pix4) / 4
			pixel_reflectance = reflectance(pix1, lo_nir1_rectangle_averages[os.path.join(lo_nir1_directory, bin_file)])
			binning_pixel_reflectance = reflectance(average_pix, lo_nir1_rectangle_averages[os.path.join(lo_nir1_directory, bin_file)])
			binning_pixel_reflectances.append(binning_pixel_reflectance)
			pixel_reflectances.append(pixel_reflectance)
		for wl, bin_file in zip(lo_nir2_filtered_wls, lo_nir2_filtered_binfiles):
			path = os.path.join(lo_nir2_acquisition_path, bin_file)
			with open(path, 'rb') as f:
				buf = f.read()
			img = np.frombuffer(buf, dtype=np.uint16).reshape((nir_height, nir_width))
			pix1 = img[nir2_y, nir2_x]
			pix2 = img[nir2_y+1, nir2_x]
			pix3 = img[nir2_y, nir2_x+1]
			pix4 = img[nir2_y+1, nir2_x+1]
			average_pix = (pix1 + pix2 + pix3 + pix4) / 4
			pixel_reflectance = reflectance(pix1, lo_nir2_rectangle_averages[os.path.join(lo_nir2_directory, bin_file)])
			binning_pixel_reflectance = reflectance(average_pix, lo_nir2_rectangle_averages[os.path.join(lo_nir2_directory, bin_file)])
			binning_pixel_reflectances.append(binning_pixel_reflectance)
			pixel_reflectances.append(pixel_reflectance)
		lo_pixel_reflectances = pixel_reflectances
		lo_binning_pixel_reflectances = binning_pixel_reflectances
		all_coordinates_lo_pixel_reflectances.append(lo_pixel_reflectances)
		all_coordinates_lo_binning_pixel_reflectances.append(lo_binning_pixel_reflectances)
		if not lo_vis_directory:
			reflectance_matrix[i, :] = [0, 0, 0, 0, 0, 0, 0] + lo_pixel_reflectances
			binning_reflectance_matrix[i, :] = [0, 0, 0, 0, 0, 0, 0] + lo_binning_pixel_reflectances
		else:
			reflectance_matrix[i, :] = lo_pixel_reflectances
			binning_reflectance_matrix[i, :] = lo_binning_pixel_reflectances
		axr = axes_ref[i, 0]
		if lo_vis_directory:
			axr.plot(range(len(lo_vis_x_labels+lo_nir1_x_labels+lo_nir2_x_labels)), lo_binning_pixel_reflectances)
			axr.set_xticks(range(len(lo_vis_x_labels+lo_nir1_x_labels+lo_nir2_x_labels)))
			axr.set_xticklabels(lo_vis_x_labels+lo_nir1_x_labels+lo_nir2_x_labels, rotation=40, ha='right', fontsize=12)
		else:
			axr.plot(range(len([0,0,0,0,0,0,0]+lo_nir1_x_labels+lo_nir2_x_labels)), [0.18,0.18,0.18,0.18,0.18,0.18,0.18]+lo_binning_pixel_reflectances)
			axr.set_xticks(range(len([0,0,0,0,0,0,0]+lo_nir1_x_labels+lo_nir2_x_labels)))
			axr.set_xticklabels([0,0,0,0,0,0,0]+lo_nir1_x_labels+lo_nir2_x_labels, rotation=40, ha='right', fontsize=12)
		axr.set_ylabel("Pixel Reflectance")
		axr.set_title(f"Low Order, Coordinate {i}")
		axr.grid(True, axis='y')
		if min(lo_binning_pixel_reflectances) < min_y_value:
			min_y_value = min(lo_binning_pixel_reflectances)
		if max(lo_binning_pixel_reflectances) > max_y_value:
			max_y_value = max(lo_binning_pixel_reflectances)

	# High Order
	for i, ((vis_x, vis_y), (nir1_x, nir1_y), (nir2_x, nir2_y)) in enumerate(zip(vis_channel_pixel_coordinates, nir1_channel_pixel_coordinates, nir2_channel_pixel_coordinates)):
		pixel_reflectances = []
		binning_pixel_reflectances = []
		for wl, bin_file in zip(ho_vis_filtered_wls, ho_vis_filtered_binfiles):
			path = os.path.join(ho_vis_acquisition_path, bin_file)
			with open(path, 'rb') as f:
				buf = f.read()
			img = np.frombuffer(buf, dtype=np.uint16).reshape((vis_height, vis_width))
			pix1 = img[vis_y, vis_x]
			pix2 = img[vis_y+1, vis_x]
			pix3 = img[vis_y, vis_x+1]
			pix4 = img[vis_y+1, vis_x+1]
			average_pix = (pix1 + pix2 + pix3 + pix4) / 4
			pixel_reflectance = reflectance(pix1, ho_vis_rectangle_averages[os.path.join(ho_vis_directory, bin_file)])
			binning_pixel_reflectance = reflectance(average_pix, ho_vis_rectangle_averages[os.path.join(ho_vis_directory, bin_file)])
			binning_pixel_reflectances.append(binning_pixel_reflectance)
			pixel_reflectances.append(pixel_reflectance)
		for wl, bin_file in zip(ho_nir1_filtered_wls, ho_nir1_filtered_binfiles):
			path = os.path.join(ho_nir1_acquisition_path, bin_file)
			with open(path, 'rb') as f:
				buf = f.read()
			img = np.frombuffer(buf, dtype=np.uint16).reshape((nir_height, nir_width))
			pix1 = img[nir1_y, nir1_x]
			pix2 = img[nir1_y+1, nir1_x]
			pix3 = img[nir1_y, nir1_x+1]
			pix4 = img[nir1_y+1, nir1_x+1]
			average_pix = (pix1 + pix2 + pix3 + pix4) / 4
			pixel_reflectance = reflectance(pix1, ho_nir1_rectangle_averages[os.path.join(ho_nir1_directory, bin_file)])
			binning_pixel_reflectance = reflectance(average_pix, ho_nir1_rectangle_averages[os.path.join(ho_nir1_directory, bin_file)])
			binning_pixel_reflectances.append(binning_pixel_reflectance)
			pixel_reflectances.append(pixel_reflectance)
		for wl, bin_file in zip(ho_nir2_filtered_wls, ho_nir2_filtered_binfiles):
			path = os.path.join(ho_nir2_acquisition_path, bin_file)
			with open(path, 'rb') as f:
				buf = f.read()
			img = np.frombuffer(buf, dtype=np.uint16).reshape((nir_height, nir_width))
			pix1 = img[nir2_y, nir2_x]
			pix2 = img[nir2_y+1, nir2_x]
			pix3 = img[nir2_y, nir2_x+1]
			pix4 = img[nir2_y+1, nir2_x+1]
			average_pix = (pix1 + pix2 + pix3 + pix4) / 4
			pixel_reflectance = reflectance(pix1, ho_nir2_rectangle_averages[os.path.join(ho_nir2_directory, bin_file)])
			binning_pixel_reflectance = reflectance(average_pix, ho_nir2_rectangle_averages[os.path.join(ho_nir2_directory, bin_file)])
			binning_pixel_reflectances.append(binning_pixel_reflectance)
			pixel_reflectances.append(pixel_reflectance)
		ho_pixel_reflectances = pixel_reflectances
		ho_binning_pixel_reflectances = binning_pixel_reflectances
		all_coordinates_ho_pixel_reflectances.append(ho_pixel_reflectances)
		all_coordinates_ho_binning_pixel_reflectances.append(ho_binning_pixel_reflectances)
		reflectance_matrix[n_coords+i, :] = ho_pixel_reflectances
		binning_reflectance_matrix[n_coords+i, :] = ho_binning_pixel_reflectances
		axr = axes_ref[i, 1]
		axr.plot(range(len(ho_vis_x_labels+ho_nir1_x_labels+ho_nir2_x_labels)), ho_binning_pixel_reflectances)
		axr.set_xticks(range(len(ho_vis_x_labels+ho_nir1_x_labels+ho_nir2_x_labels)))
		axr.set_xticklabels(ho_vis_x_labels+ho_nir1_x_labels+ho_nir2_x_labels, rotation=40, ha='right', fontsize=12)
		axr.set_ylabel("Pixel Reflectance")
		axr.set_title(f"High Order, Coordinate {i}")
		axr.grid(True, axis='y')
		if min(ho_binning_pixel_reflectances) < min_y_value:
			min_y_value = min(ho_binning_pixel_reflectances)
		if max(ho_binning_pixel_reflectances) > max_y_value:
			max_y_value = max(ho_binning_pixel_reflectances)
	df = pd.DataFrame(reflectance_matrix, columns=all_x_labels)
	excel_path = os.path.join(output_directory, f"{key}_pixel_reflectances(single_pixel).xlsx")
	if not test_mode:
		df.to_excel(excel_path, index=False)
	df = pd.DataFrame(binning_reflectance_matrix, columns=all_x_labels)
	excel_path = os.path.join(output_directory, f"{key}_pixel_reflectances(4-pixel_binning).xlsx")
	if not test_mode:
		df.to_excel(excel_path, index=False)

	for axr in axes_ref.flatten():
		axr.set_ylim(min_y_value, max_y_value)

	plt.tight_layout()
	out_ref = os.path.join(output_directory, f"{key}_pixel_reflectances.png")
	if test_mode:
		pass
		# plt.show()
	else:
		fig_ref.savefig(out_ref)
	plt.close(fig_ref)

	# Signal to Noise Ratio
	fig, axes = plt.subplots(num_coordinates, 2, figsize=(20, 4*num_coordinates), sharex=True)
	if num_coordinates == 1:
		axes = [axes]
	min_y_value, max_y_value = float('inf'), float('-inf')
	all_coordinates_lo_signal_to_noise_ratios = []
	all_coordinates_lo_binning_signal_to_noise_ratios = []
	all_coordinates_ho_signal_to_noise_ratios = []
	all_coordinates_ho_binning_signal_to_noise_ratios = []
	# Low Order
	for i, (vis_coordinates, nir1_coordinates, nir2_coordinates) in enumerate(zip(vis_channel_pixel_coordinates, nir1_channel_pixel_coordinates, nir2_channel_pixel_coordinates)):
		vis_signal_to_noise_ratios = []
		binning_vis_signal_to_noise_ratios = []
		if lo_vis_directory:
			for j, bin_file in enumerate(lo_vis_filtered_binfiles):
				file_path = os.path.join(lo_vis_acquisition_path, bin_file)
				with open(file_path, 'rb') as file:
					bin_data = file.read()
					img_array = np.frombuffer(bin_data, dtype=np.uint16)
					img_array = img_array.reshape((vis_height, vis_width))
					pixel1 = img_array[vis_coordinates[1], vis_coordinates[0]]
					pixel2 = img_array[vis_coordinates[1], vis_coordinates[0]+1]
					pixel3 = img_array[vis_coordinates[1]+1, vis_coordinates[0]]
					pixel4 = img_array[vis_coordinates[1]+1, vis_coordinates[0]+1]
					average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
					standard_deviation = query_std_from_excel(lo_vis_directory)
					signal_to_noise_ratio = calculate_signal_to_noise_ratio(pixel1, standard_deviation)
					binning_signal_to_noise_ratio = calculate_signal_to_noise_ratio(average_pixel, standard_deviation)
					binning_vis_signal_to_noise_ratios.append(binning_signal_to_noise_ratio)
					vis_signal_to_noise_ratios.append(signal_to_noise_ratio)

		nir1_signal_to_noise_ratios = []
		binning_nir1_signal_to_noise_ratios = []
		for j, bin_file in enumerate(lo_nir1_filtered_binfiles):
			file_path = os.path.join(lo_nir1_acquisition_path, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((nir_height, nir_width))
				pixel1 = img_array[nir1_coordinates[1], nir1_coordinates[0]]
				pixel2 = img_array[nir1_coordinates[1], nir1_coordinates[0]+1]
				pixel3 = img_array[nir1_coordinates[1]+1, nir1_coordinates[0]]
				pixel4 = img_array[nir1_coordinates[1]+1, nir1_coordinates[0]+1]
				average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
				standard_deviation = query_std_from_excel(lo_nir1_directory)
				signal_to_noise_ratio = calculate_signal_to_noise_ratio(pixel1, standard_deviation)
				binning_signal_to_noise_ratio = calculate_signal_to_noise_ratio(average_pixel, standard_deviation)
				binning_nir1_signal_to_noise_ratios.append(binning_signal_to_noise_ratio)
				nir1_signal_to_noise_ratios.append(signal_to_noise_ratio)

		nir2_signal_to_noise_ratios = []
		binning_nir2_signal_to_noise_ratios = []
		for j, bin_file in enumerate(lo_nir2_filtered_binfiles):
			file_path = os.path.join(lo_nir2_acquisition_path, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((nir_height, nir_width))
				pixel1 = img_array[nir2_coordinates[1], nir2_coordinates[0]]
				pixel2 = img_array[nir2_coordinates[1], nir2_coordinates[0]+1]
				pixel3 = img_array[nir2_coordinates[1]+1, nir2_coordinates[0]]
				pixel4 = img_array[nir2_coordinates[1]+1, nir2_coordinates[0]+1]
				average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
				standard_deviation = query_std_from_excel(lo_nir2_directory)
				signal_to_noise_ratio = calculate_signal_to_noise_ratio(pixel1, standard_deviation)
				binning_signal_to_noise_ratio = calculate_signal_to_noise_ratio(average_pixel, standard_deviation)
				binning_nir2_signal_to_noise_ratios.append(binning_signal_to_noise_ratio)
				nir2_signal_to_noise_ratios.append(signal_to_noise_ratio)

		ax = axes[i, 0]
		lo_signal_to_noise_ratios = vis_signal_to_noise_ratios+nir1_signal_to_noise_ratios+nir2_signal_to_noise_ratios
		all_coordinates_lo_signal_to_noise_ratios.append(lo_signal_to_noise_ratios)
		lo_binning_signal_to_noise_ratios = binning_vis_signal_to_noise_ratios+binning_nir1_signal_to_noise_ratios+binning_nir2_signal_to_noise_ratios
		all_coordinates_lo_binning_signal_to_noise_ratios.append(lo_binning_signal_to_noise_ratios)
		if lo_vis_directory:
			ax.plot(range(len(lo_vis_x_labels+lo_nir1_x_labels+lo_nir2_x_labels)), lo_binning_signal_to_noise_ratios, c='blue')
			ax.set_xticks(range(len(lo_vis_x_labels+lo_nir1_x_labels+lo_nir2_x_labels)))
			ax.set_xticklabels(lo_vis_x_labels+lo_nir1_x_labels+lo_nir2_x_labels, rotation=40, ha='right', fontsize=12)
		else:
			ax.plot(range(len([0,0,0,0,0,0,0]+lo_nir1_x_labels+lo_nir2_x_labels)), [0,0,0,0,0,0,0]+lo_binning_signal_to_noise_ratios, c='blue')
			ax.set_xticks(range(len([0,0,0,0,0,0,0]+lo_nir1_x_labels+lo_nir2_x_labels)))
			ax.set_xticklabels([0,0,0,0,0,0,0]+lo_nir1_x_labels+lo_nir2_x_labels, rotation=40, ha='right', fontsize=12)
		ax.set_ylabel("Signal to Noise Ratio")
		ax.set_title(f"Low Order, Coordinate {i}")
		ax.grid(True, axis='y')
		if min(lo_binning_signal_to_noise_ratios) < min_y_value:
			min_y_value = min(lo_binning_signal_to_noise_ratios)
		if max(lo_binning_signal_to_noise_ratios) > max_y_value:
			max_y_value = max(lo_binning_signal_to_noise_ratios)

	# High Order
	for i, (vis_coordinates, nir1_coordinates, nir2_coordinates) in enumerate(zip(vis_channel_pixel_coordinates, nir1_channel_pixel_coordinates, nir2_channel_pixel_coordinates)):
		vis_signal_to_noise_ratios = []
		binning_vis_signal_to_noise_ratios = []
		for j, bin_file in enumerate(ho_vis_filtered_binfiles):
			file_path = os.path.join(ho_vis_acquisition_path, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((vis_height, vis_width))
				pixel1 = img_array[vis_coordinates[1], vis_coordinates[0]]
				pixel2 = img_array[vis_coordinates[1], vis_coordinates[0]+1]
				pixel3 = img_array[vis_coordinates[1]+1, vis_coordinates[0]]
				pixel4 = img_array[vis_coordinates[1]+1, vis_coordinates[0]+1]
				average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
				standard_deviation = query_std_from_excel(ho_vis_directory)
				signal_to_noise_ratio = calculate_signal_to_noise_ratio(pixel1, standard_deviation)
				binning_signal_to_noise_ratio = calculate_signal_to_noise_ratio(average_pixel, standard_deviation)
				binning_vis_signal_to_noise_ratios.append(binning_signal_to_noise_ratio)
				vis_signal_to_noise_ratios.append(signal_to_noise_ratio)
		
		nir1_signal_to_noise_ratios = []
		binning_nir1_signal_to_noise_ratios = []
		for j, bin_file in enumerate(ho_nir1_filtered_binfiles):
			file_path = os.path.join(ho_nir1_acquisition_path, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((nir_height, nir_width))
				pixel1 = img_array[nir1_coordinates[1], nir1_coordinates[0]]
				pixel2 = img_array[nir1_coordinates[1], nir1_coordinates[0]+1]
				pixel3 = img_array[nir1_coordinates[1]+1, nir1_coordinates[0]]
				pixel4 = img_array[nir1_coordinates[1]+1, nir1_coordinates[0]+1]
				average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
				standard_deviation = query_std_from_excel(ho_nir1_directory)
				signal_to_noise_ratio = calculate_signal_to_noise_ratio(pixel1, standard_deviation)
				binning_signal_to_noise_ratio = calculate_signal_to_noise_ratio(average_pixel, standard_deviation)
				binning_nir1_signal_to_noise_ratios.append(binning_signal_to_noise_ratio)
				nir1_signal_to_noise_ratios.append(signal_to_noise_ratio)

		nir2_signal_to_noise_ratios = []
		binning_nir2_signal_to_noise_ratios = []
		for j, bin_file in enumerate(ho_nir2_filtered_binfiles):
			file_path = os.path.join(ho_nir2_acquisition_path, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((nir_height, nir_width))
				pixel1 = img_array[nir2_coordinates[1], nir2_coordinates[0]]
				pixel2 = img_array[nir2_coordinates[1], nir2_coordinates[0]+1]
				pixel3 = img_array[nir2_coordinates[1]+1, nir2_coordinates[0]]
				pixel4 = img_array[nir2_coordinates[1]+1, nir2_coordinates[0]+1]
				average_pixel = (pixel1 + pixel2 + pixel3 + pixel4) / 4
				standard_deviation = query_std_from_excel(ho_nir2_directory)
				signal_to_noise_ratio = calculate_signal_to_noise_ratio(pixel1, standard_deviation)
				binning_signal_to_noise_ratio = calculate_signal_to_noise_ratio(average_pixel, standard_deviation)
				binning_nir2_signal_to_noise_ratios.append(binning_signal_to_noise_ratio)
				nir2_signal_to_noise_ratios.append(signal_to_noise_ratio)

		ax = axes[i, 1]
		ho_signal_to_noise_ratios = vis_signal_to_noise_ratios+nir1_signal_to_noise_ratios+nir2_signal_to_noise_ratios
		all_coordinates_ho_signal_to_noise_ratios.append(ho_signal_to_noise_ratios)
		ho_binning_signal_to_noise_ratios = binning_vis_signal_to_noise_ratios+binning_nir1_signal_to_noise_ratios+binning_nir2_signal_to_noise_ratios
		all_coordinates_ho_binning_signal_to_noise_ratios.append(ho_binning_signal_to_noise_ratios)
		ax.plot(range(len(ho_vis_x_labels+ho_nir1_x_labels+ho_nir2_x_labels)), ho_binning_signal_to_noise_ratios, c='blue')
		ax.set_xticks(range(len(ho_vis_x_labels+ho_nir1_x_labels+ho_nir2_x_labels)))
		ax.set_xticklabels(ho_vis_x_labels+ho_nir1_x_labels+ho_nir2_x_labels, rotation=40, ha='right', fontsize=12)
		ax.set_ylabel("Signal to Noise Ratio")
		ax.set_title(f"High Order, Coordinate {i}")
		ax.grid(True, axis='y')
		if min(ho_binning_signal_to_noise_ratios) < min_y_value:
			min_y_value = min(ho_binning_signal_to_noise_ratios)
		if max(ho_binning_signal_to_noise_ratios) > max_y_value:
			max_y_value = max(ho_binning_signal_to_noise_ratios)
		
	for axr in axes.flatten():
		axr.set_ylim(0, max_y_value) # Optionally min_y_value instead of 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"{key}_signal_to_noise_ratio.png")
	if test_mode:
		pass
		# plt.show()
	else:
		plt.savefig(plot_path)
	plt.close(fig)

	if lo_vis_directory:
		rectangle_averages = lo_vis_rectangle_averages | lo_nir1_rectangle_averages | lo_nir2_rectangle_averages | ho_vis_rectangle_averages | ho_nir1_rectangle_averages | ho_nir2_rectangle_averages
		lo_x_labels = lo_vis_x_labels+lo_nir1_x_labels+lo_nir2_x_labels
	else:
		rectangle_averages = lo_nir1_rectangle_averages | lo_nir2_rectangle_averages | ho_vis_rectangle_averages | ho_nir1_rectangle_averages | ho_nir2_rectangle_averages
		lo_x_labels = lo_nir1_x_labels+lo_nir2_x_labels
	ho_x_labels = ho_vis_x_labels+ho_nir1_x_labels+ho_nir2_x_labels
	pixel_values = (all_coordinates_lo_pixel_values, all_coordinates_lo_binning_pixel_values, all_coordinates_ho_pixel_values, all_coordinates_ho_binning_pixel_values)
	pixel_reflectances = (all_coordinates_lo_pixel_reflectances, all_coordinates_lo_binning_pixel_reflectances, all_coordinates_ho_pixel_reflectances, all_coordinates_ho_binning_pixel_reflectances)
	signal_to_noise_ratios = (all_coordinates_lo_signal_to_noise_ratios, all_coordinates_lo_binning_signal_to_noise_ratios, all_coordinates_ho_signal_to_noise_ratios, all_coordinates_ho_binning_signal_to_noise_ratios)

	return rectangle_averages, pixel_values, pixel_reflectances, signal_to_noise_ratios, lo_x_labels, ho_x_labels, num_coordinates

def exposure_combination_graphs(
		all_keys: list,
		all_lo_pixel_reflectances: list, # key[ coordinates[ bins[] ] ]
		all_lo_binned_pixel_reflectances: list,
		all_ho_pixel_reflectances: list,
		all_ho_binned_pixel_reflectances: list,
		all_lo_signal_to_noise_ratios: list,
		all_lo_binned_signal_to_noise_ratios: list,
		all_ho_signal_to_noise_ratios: list,
		all_ho_binned_signal_to_noise_ratios: list,
		lo_x_labels: list,
		ho_x_labels: list,
		num_coordinates: int,
		output_directory: str,
		test_mode: bool = False,
	):
	# Single Pixel Reflectance
	fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=True)
	min_y_value, max_y_value = float('inf'), float('-inf')

	# history = []
	for key_i, key in enumerate(all_keys):
		colors = plt.cm.tab10.colors
		color_cycle = itertools.cycle(colors)
		if '1250' in key:
			# if '1250' in history:
			# 	continue
			# history.append('1250')
			lo_ax = axes[0, 0]
			ho_ax = axes[0, 1]
		elif '1875' in key:
			# if '1875' in history:
			# 	continue
			# history.append('1875')
			lo_ax = axes[1, 0]
			ho_ax = axes[1, 1]
		elif '2500' in key:
			# if '2500' in history:
			# 	continue
			# history.append('2500')
			lo_ax = axes[2, 0]
			ho_ax = axes[2, 1]
		for coordinate_i in range(len(all_lo_pixel_reflectances[0])):
			c = next(color_cycle)

			# Low Order
			if len(all_lo_pixel_reflectances[key_i][coordinate_i]) < 30:
				lo_ax.plot(range(len(lo_x_labels)), [0.18,0.18,0.18,0.18,0.18,0.18,0.18]+all_lo_pixel_reflectances[key_i][coordinate_i], c=c)
			else:
				lo_ax.plot(range(len(lo_x_labels)), all_lo_pixel_reflectances[key_i][coordinate_i], c=c)
			if min(all_lo_pixel_reflectances[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_lo_pixel_reflectances[key_i][coordinate_i])
			if max(all_lo_pixel_reflectances[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_lo_pixel_reflectances[key_i][coordinate_i])

			# High Order
			ho_ax.plot(range(len(ho_x_labels)), all_ho_pixel_reflectances[key_i][coordinate_i], c=c)
			if min(all_ho_pixel_reflectances[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_ho_pixel_reflectances[key_i][coordinate_i])
			if max(all_ho_pixel_reflectances[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_ho_pixel_reflectances[key_i][coordinate_i])

		lo_ax.set_xticks(range(len(lo_x_labels)))
		lo_ax.set_xticklabels(lo_x_labels, rotation=40, ha='right', fontsize=12)
		lo_ax.set_ylabel("Pixel Reflectance")
		lo_ax.set_title(f"Low Order, Exposures {key[15:]}")
		lo_ax.grid(True, axis='y')

		ho_ax.set_xticks(range(len(ho_x_labels)))
		ho_ax.set_xticklabels(ho_x_labels, rotation=40, ha='right', fontsize=12)
		ho_ax.set_ylabel("Pixel Reflectance")
		ho_ax.set_title(f"High Order, Exposures {key[15:]}")
		ho_ax.grid(True, axis='y')

	for axr in axes.flatten():
		axr.set_ylim(min_y_value, max_y_value) # Optionally min_y_value or 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"all_reflectances_grouped_by_exposures_(single_pixel).png")
	if test_mode:
		plt.show()
	else:
		plt.savefig(plot_path)
	plt.close(fig)

	# Binned Pixel Reflectance
	fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=True)
	min_y_value, max_y_value = float('inf'), float('-inf')

	# history = []
	for key_i, key in enumerate(all_keys):
		colors = plt.cm.tab10.colors
		color_cycle = itertools.cycle(colors)
		if '1250' in key:
			# if '1250' in history:
			# 	continue
			# history.append('1250')
			lo_ax = axes[0, 0]
			ho_ax = axes[0, 1]
		elif '1875' in key:
			# if '1875' in history:
			# 	continue
			# history.append('1875')
			lo_ax = axes[1, 0]
			ho_ax = axes[1, 1]
		elif '2500' in key:
			# if '2500' in history:
			# 	continue
			# history.append('2500')
			lo_ax = axes[2, 0]
			ho_ax = axes[2, 1]
		for coordinate_i in range(len(all_lo_pixel_reflectances[0])):
			c = next(color_cycle)

			# Low Order
			if len(all_lo_binned_pixel_reflectances[key_i][coordinate_i]) < 30:
				lo_ax.plot(range(len(lo_x_labels)), [0.18,0.18,0.18,0.18,0.18,0.18,0.18]+all_lo_binned_pixel_reflectances[key_i][coordinate_i], c=c)
			else:
				lo_ax.plot(range(len(lo_x_labels)), all_lo_binned_pixel_reflectances[key_i][coordinate_i], c=c)
			if min(all_lo_binned_pixel_reflectances[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_lo_binned_pixel_reflectances[key_i][coordinate_i])
			if max(all_lo_binned_pixel_reflectances[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_lo_binned_pixel_reflectances[key_i][coordinate_i])

			# High Order
			ho_ax.plot(range(len(ho_x_labels)), all_ho_binned_pixel_reflectances[key_i][coordinate_i], c=c)
			if min(all_ho_binned_pixel_reflectances[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_ho_binned_pixel_reflectances[key_i][coordinate_i])
			if max(all_ho_binned_pixel_reflectances[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_ho_binned_pixel_reflectances[key_i][coordinate_i])

		lo_ax.set_xticks(range(len(lo_x_labels)))
		lo_ax.set_xticklabels(lo_x_labels, rotation=40, ha='right', fontsize=12)
		lo_ax.set_ylabel("Pixel Reflectance")
		lo_ax.set_title(f"Low Order, Exposures {key[15:]}")
		lo_ax.grid(True, axis='y')

		ho_ax.set_xticks(range(len(ho_x_labels)))
		ho_ax.set_xticklabels(ho_x_labels, rotation=40, ha='right', fontsize=12)
		ho_ax.set_ylabel("Pixel Reflectance")
		ho_ax.set_title(f"High Order, Exposures {key[15:]}")
		ho_ax.grid(True, axis='y')

	for axr in axes.flatten():
		axr.set_ylim(min_y_value, max_y_value) # Optionally min_y_value or 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"all_reflectances_grouped_by_exposures_(4-pixel_binning).png")
	if test_mode:
		plt.show()
	else:
		plt.savefig(plot_path)
	plt.close(fig)

	# Single Pixel Signal to Noise Ratio
	fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=True)
	min_y_value, max_y_value = float('inf'), float('-inf')

	# history = []
	for key_i, key in enumerate(all_keys):
		colors = plt.cm.tab10.colors
		color_cycle = itertools.cycle(colors)
		if '1250' in key:
			# if '1250' in history:
			# 	continue
			# history.append('1250')
			lo_ax = axes[0, 0]
			ho_ax = axes[0, 1]
		elif '1875' in key:
			# if '1875' in history:
			# 	continue
			# history.append('1875')
			lo_ax = axes[1, 0]
			ho_ax = axes[1, 1]
		elif '2500' in key:
			# if '2500' in history:
			# 	continue
			# history.append('2500')
			lo_ax = axes[2, 0]
			ho_ax = axes[2, 1]
		for coordinate_i in range(len(all_lo_pixel_reflectances[0])):
			c = next(color_cycle)

			# Low Order
			if len(all_lo_signal_to_noise_ratios[key_i][coordinate_i]) < 30:
				lo_ax.plot(range(len(lo_x_labels)), [0.18,0.18,0.18,0.18,0.18,0.18,0.18]+all_lo_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			else:
				lo_ax.plot(range(len(lo_x_labels)), all_lo_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			if min(all_lo_signal_to_noise_ratios[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_lo_signal_to_noise_ratios[key_i][coordinate_i])
			if max(all_lo_signal_to_noise_ratios[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_lo_signal_to_noise_ratios[key_i][coordinate_i])

			# High Order
			ho_ax.plot(range(len(ho_x_labels)), all_ho_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			if min(all_ho_signal_to_noise_ratios[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_ho_signal_to_noise_ratios[key_i][coordinate_i])
			if max(all_ho_signal_to_noise_ratios[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_ho_signal_to_noise_ratios[key_i][coordinate_i])

		lo_ax.set_xticks(range(len(lo_x_labels)))
		lo_ax.set_xticklabels(lo_x_labels, rotation=40, ha='right', fontsize=12)
		lo_ax.set_ylabel("Signal to Noise Ratio")
		lo_ax.set_title(f"Low Order, Exposures {key[15:]}")
		lo_ax.grid(True, axis='y')

		ho_ax.set_xticks(range(len(ho_x_labels)))
		ho_ax.set_xticklabels(ho_x_labels, rotation=40, ha='right', fontsize=12)
		ho_ax.set_ylabel("Signal to Noise Ratio")
		ho_ax.set_title(f"High Order, Exposures {key[15:]}")
		ho_ax.grid(True, axis='y')

	for axr in axes.flatten():
		axr.set_ylim(0, max_y_value) # Optionally min_y_value or 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"all_signal_to_noise_ratios_grouped_by_exposure_(single_pixel).png")
	if test_mode:
		plt.show()
	else:
		plt.savefig(plot_path)
	plt.close(fig)

	# Binned Signal to Noise Ratio
	fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=True)
	min_y_value, max_y_value = float('inf'), float('-inf')

	# history = []
	for key_i, key in enumerate(all_keys):
		colors = plt.cm.tab10.colors
		color_cycle = itertools.cycle(colors)
		if '1250' in key:
			# if '1250' in history:
			# 	continue
			# history.append('1250')
			lo_ax = axes[0, 0]
			ho_ax = axes[0, 1]
		elif '1875' in key:
			# if '1875' in history:
			# 	continue
			# history.append('1875')
			lo_ax = axes[1, 0]
			ho_ax = axes[1, 1]
		elif '2500' in key:
			# if '2500' in history:
			# 	continue
			# history.append('2500')
			lo_ax = axes[2, 0]
			ho_ax = axes[2, 1]
		for coordinate_i in range(len(all_lo_pixel_reflectances[0])):
			c = next(color_cycle)

			# Low Order
			if len(all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i]) < 30:
				lo_ax.plot(range(len(lo_x_labels)), [0.18,0.18,0.18,0.18,0.18,0.18,0.18]+all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			else:
				lo_ax.plot(range(len(lo_x_labels)), all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			if min(all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i])
			if max(all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i])

			# High Order
			ho_ax.plot(range(len(ho_x_labels)), all_ho_binned_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			if min(all_ho_binned_signal_to_noise_ratios[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_ho_binned_signal_to_noise_ratios[key_i][coordinate_i])
			if max(all_ho_binned_signal_to_noise_ratios[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_ho_binned_signal_to_noise_ratios[key_i][coordinate_i])

		lo_ax.set_xticks(range(len(lo_x_labels)))
		lo_ax.set_xticklabels(lo_x_labels, rotation=40, ha='right', fontsize=12)
		lo_ax.set_ylabel("Signal to Noise Ratio")
		lo_ax.set_title(f"Low Order, Exposures {key[15:]}")
		lo_ax.grid(True, axis='y')

		ho_ax.set_xticks(range(len(ho_x_labels)))
		ho_ax.set_xticklabels(ho_x_labels, rotation=40, ha='right', fontsize=12)
		ho_ax.set_ylabel("Signal to Noise Ratio")
		ho_ax.set_title(f"High Order, Exposures {key[15:]}")
		ho_ax.grid(True, axis='y')

	for axr in axes.flatten():
		axr.set_ylim(0, max_y_value) # Optionally min_y_value or 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"all_signal_to_noise_ratios_grouped_by_exposure_(4-pixel_binning).png")
	if test_mode:
		plt.show()
	else:
		plt.savefig(plot_path)
	plt.close(fig)

def power_combination_graphs(
		all_keys: list,
		all_lo_pixel_reflectances: list, # key[ coordinates[ bins[] ] ]
		all_lo_binned_pixel_reflectances: list,
		all_ho_pixel_reflectances: list,
		all_ho_binned_pixel_reflectances: list,
		all_lo_signal_to_noise_ratios: list,
		all_lo_binned_signal_to_noise_ratios: list,
		all_ho_signal_to_noise_ratios: list,
		all_ho_binned_signal_to_noise_ratios: list,
		lo_x_labels: list,
		ho_x_labels: list,
		num_coordinates: int,
		output_directory: str,
		test_mode: bool = False,
	):
	# Single Pixel Reflectance
	fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=True)
	min_y_value, max_y_value = float('inf'), float('-inf')

	# history = []
	for key_i, key in enumerate(all_keys):
		colors = plt.cm.tab10.colors
		color_cycle = itertools.cycle(colors)
		if '200w' in key:
			# if '200w' in history:
			# 	continue
			# history.append('200w')
			lo_ax = axes[2, 0]
			ho_ax = axes[2, 1]
		elif '400w' in key:
			# if '400w' in history:
			# 	continue
			# history.append('400w')
			lo_ax = axes[1, 0]
			ho_ax = axes[1, 1]
		elif '600w' in key:
			# if '600w' in history:
			# 	continue
			# history.append('600w')
			lo_ax = axes[0, 0]
			ho_ax = axes[0, 1]
		for coordinate_i in range(len(all_lo_pixel_reflectances[0])):
			c = next(color_cycle)

			# Low Order
			if len(all_lo_pixel_reflectances[key_i][coordinate_i]) < 30:
				lo_ax.plot(range(len(lo_x_labels)), [0.18,0.18,0.18,0.18,0.18,0.18,0.18]+all_lo_pixel_reflectances[key_i][coordinate_i], c=c)
			else:
				lo_ax.plot(range(len(lo_x_labels)), all_lo_pixel_reflectances[key_i][coordinate_i], c=c)
			if min(all_lo_pixel_reflectances[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_lo_pixel_reflectances[key_i][coordinate_i])
			if max(all_lo_pixel_reflectances[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_lo_pixel_reflectances[key_i][coordinate_i])

			# High Order
			ho_ax.plot(range(len(ho_x_labels)), all_ho_pixel_reflectances[key_i][coordinate_i], c=c)
			if min(all_ho_pixel_reflectances[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_ho_pixel_reflectances[key_i][coordinate_i])
			if max(all_ho_pixel_reflectances[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_ho_pixel_reflectances[key_i][coordinate_i])

		lo_ax.set_xticks(range(len(lo_x_labels)))
		lo_ax.set_xticklabels(lo_x_labels, rotation=40, ha='right', fontsize=12)
		lo_ax.set_ylabel("Pixel Reflectance")
		lo_ax.set_title(f"Low Order, Power {key[:4]}")
		lo_ax.grid(True, axis='y')

		ho_ax.set_xticks(range(len(ho_x_labels)))
		ho_ax.set_xticklabels(ho_x_labels, rotation=40, ha='right', fontsize=12)
		ho_ax.set_ylabel("Pixel Reflectance")
		ho_ax.set_title(f"High Order, Power {key[:4]}")
		ho_ax.grid(True, axis='y')

	for axr in axes.flatten():
		axr.set_ylim(min_y_value, max_y_value) # Optionally min_y_value or 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"all_reflectances_grouped_by_power_(single_pixel).png")
	if test_mode:
		plt.show()
	else:
		plt.savefig(plot_path)
	plt.close(fig)

	# Binned Pixel Reflectance
	fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=True)
	min_y_value, max_y_value = float('inf'), float('-inf')

	# history = []
	for key_i, key in enumerate(all_keys):
		colors = plt.cm.tab10.colors
		color_cycle = itertools.cycle(colors)
		if '200w' in key:
			# if '200w' in history:
			# 	continue
			# history.append('200w')
			lo_ax = axes[2, 0]
			ho_ax = axes[2, 1]
		elif '400w' in key:
			# if '400w' in history:
			# 	continue
			# history.append('400w')
			lo_ax = axes[1, 0]
			ho_ax = axes[1, 1]
		elif '600w' in key:
			# if '600w' in history:
			# 	continue
			# history.append('600w')
			lo_ax = axes[0, 0]
			ho_ax = axes[0, 1]
		for coordinate_i in range(len(all_lo_pixel_reflectances[0])):
			c = next(color_cycle)

			# Low Order
			if len(all_lo_pixel_reflectances[key_i][coordinate_i]) < 30:
				lo_ax.plot(range(len(lo_x_labels)), [0.18,0.18,0.18,0.18,0.18,0.18,0.18]+all_lo_binned_pixel_reflectances[key_i][coordinate_i], c=c)
			else:
				lo_ax.plot(range(len(lo_x_labels)), all_lo_binned_pixel_reflectances[key_i][coordinate_i], c=c)
			if min(all_lo_binned_pixel_reflectances[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_lo_binned_pixel_reflectances[key_i][coordinate_i])
			if max(all_lo_binned_pixel_reflectances[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_lo_binned_pixel_reflectances[key_i][coordinate_i])

			# High Order
			ho_ax.plot(range(len(ho_x_labels)), all_ho_binned_pixel_reflectances[key_i][coordinate_i], c=c)
			if min(all_ho_binned_pixel_reflectances[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_ho_binned_pixel_reflectances[key_i][coordinate_i])
			if max(all_ho_binned_pixel_reflectances[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_ho_binned_pixel_reflectances[key_i][coordinate_i])

		lo_ax.set_xticks(range(len(lo_x_labels)))
		lo_ax.set_xticklabels(lo_x_labels, rotation=40, ha='right', fontsize=12)
		lo_ax.set_ylabel("Pixel Reflectance")
		lo_ax.set_title(f"Low Order, Power {key[:4]}")
		lo_ax.grid(True, axis='y')

		ho_ax.set_xticks(range(len(ho_x_labels)))
		ho_ax.set_xticklabels(ho_x_labels, rotation=40, ha='right', fontsize=12)
		ho_ax.set_ylabel("Pixel Reflectance")
		ho_ax.set_title(f"High Order, Power {key[:4]}")
		ho_ax.grid(True, axis='y')

	for axr in axes.flatten():
		axr.set_ylim(min_y_value, max_y_value) # Optionally min_y_value or 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"all_reflectances_grouped_by_power_(4-pixel_binning).png")
	if test_mode:
		plt.show()
	else:
		plt.savefig(plot_path)
	plt.close(fig)

	# Single Pixel Signal to Noise Ratio
	fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=True)
	min_y_value, max_y_value = float('inf'), float('-inf')

	# history = []
	for key_i, key in enumerate(all_keys):
		colors = plt.cm.tab10.colors
		color_cycle = itertools.cycle(colors)
		if '200w' in key:
			# if '200w' in history:
			# 	continue
			# history.append('200w')
			lo_ax = axes[2, 0]
			ho_ax = axes[2, 1]
		elif '400w' in key:
			# if '400w' in history:
			# 	continue
			# history.append('400w')
			lo_ax = axes[1, 0]
			ho_ax = axes[1, 1]
		elif '600w' in key:
			# if '600w' in history:
			# 	continue
			# history.append('600w')
			lo_ax = axes[0, 0]
			ho_ax = axes[0, 1]
		for coordinate_i in range(len(all_lo_pixel_reflectances[0])):
			c = next(color_cycle)

			# Low Order
			if len(all_lo_pixel_reflectances[key_i][coordinate_i]) < 30:
				lo_ax.plot(range(len(lo_x_labels)), [0.18,0.18,0.18,0.18,0.18,0.18,0.18]+all_lo_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			else:
				lo_ax.plot(range(len(lo_x_labels)), all_lo_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			if min(all_lo_signal_to_noise_ratios[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_lo_signal_to_noise_ratios[key_i][coordinate_i])
			if max(all_lo_signal_to_noise_ratios[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_lo_signal_to_noise_ratios[key_i][coordinate_i])

			# High Order
			ho_ax.plot(range(len(ho_x_labels)), all_ho_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			if min(all_ho_signal_to_noise_ratios[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_ho_signal_to_noise_ratios[key_i][coordinate_i])
			if max(all_ho_signal_to_noise_ratios[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_ho_signal_to_noise_ratios[key_i][coordinate_i])

		lo_ax.set_xticks(range(len(lo_x_labels)))
		lo_ax.set_xticklabels(lo_x_labels, rotation=40, ha='right', fontsize=12)
		lo_ax.set_ylabel("Signal to Noise Ratio")
		lo_ax.set_title(f"Low Order, Power {key[:4]}")
		lo_ax.grid(True, axis='y')

		ho_ax.set_xticks(range(len(ho_x_labels)))
		ho_ax.set_xticklabels(ho_x_labels, rotation=40, ha='right', fontsize=12)
		ho_ax.set_ylabel("Signal to Noise Ratio")
		ho_ax.set_title(f"High Order, Power {key[:4]}")
		ho_ax.grid(True, axis='y')

	for axr in axes.flatten():
		axr.set_ylim(0, max_y_value) # Optionally min_y_value or 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"all_signal_to_noise_ratios_grouped_by_power_(single_pixel).png")
	if test_mode:
		plt.show()
	else:
		plt.savefig(plot_path)
	plt.close(fig)

	# Binned Signal to Noise Ratio
	fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=True)
	min_y_value, max_y_value = float('inf'), float('-inf')

	# history = []
	for key_i, key in enumerate(all_keys):
		colors = plt.cm.tab10.colors
		color_cycle = itertools.cycle(colors)
		if '200w' in key:
			# if '200w' in history:
			# 	continue
			# history.append('200w')
			lo_ax = axes[2, 0]
			ho_ax = axes[2, 1]
		elif '400w' in key:
			# if '400w' in history:
			# 	continue
			# history.append('400w')
			lo_ax = axes[1, 0]
			ho_ax = axes[1, 1]
		elif '600w' in key:
			# if '600w' in history:
			# 	continue
			# history.append('600w')
			lo_ax = axes[0, 0]
			ho_ax = axes[0, 1]
		for coordinate_i in range(len(all_lo_pixel_reflectances[0])):
			c = next(color_cycle)

			# Low Order
			if len(all_lo_pixel_reflectances[key_i][coordinate_i]) < 30:
				lo_ax.plot(range(len(lo_x_labels)), [0.18,0.18,0.18,0.18,0.18,0.18,0.18]+all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			else:
				lo_ax.plot(range(len(lo_x_labels)), all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			if min(all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i])
			if max(all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_lo_binned_signal_to_noise_ratios[key_i][coordinate_i])

			# High Order
			ho_ax.plot(range(len(ho_x_labels)), all_ho_binned_signal_to_noise_ratios[key_i][coordinate_i], c=c)
			if min(all_ho_binned_signal_to_noise_ratios[key_i][coordinate_i]) < min_y_value:
				min_y_value = min(all_ho_binned_signal_to_noise_ratios[key_i][coordinate_i])
			if max(all_ho_binned_signal_to_noise_ratios[key_i][coordinate_i]) > max_y_value:
				max_y_value = max(all_ho_binned_signal_to_noise_ratios[key_i][coordinate_i])

		lo_ax.set_xticks(range(len(lo_x_labels)))
		lo_ax.set_xticklabels(lo_x_labels, rotation=40, ha='right', fontsize=12)
		lo_ax.set_ylabel("Signal to Noise Ratio")
		lo_ax.set_title(f"Low Order, Power {key[:4]}")
		lo_ax.grid(True, axis='y')

		ho_ax.set_xticks(range(len(ho_x_labels)))
		ho_ax.set_xticklabels(ho_x_labels, rotation=40, ha='right', fontsize=12)
		ho_ax.set_ylabel("Signal to Noise Ratio")
		ho_ax.set_title(f"High Order, Power {key[:4]}")
		ho_ax.grid(True, axis='y')

	for axr in axes.flatten():
		axr.set_ylim(0, max_y_value) # Optionally min_y_value or 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"all_signal_to_noise_ratios_grouped_by_power_(4-pixel_binning).png")
	if test_mode:
		plt.show()
	else:
		plt.savefig(plot_path)
	plt.close(fig)

def process_directories(
		metadata_parent_directory: str,
		actual_data_parent_directory: str,
		grouped_acquisition_folders: dict,
		output_directory: str,
		pixel_coordinates: dict,
		rectangle_coordinates: dict,
		test_mode: bool = False,
	):
	rectangle_averages = {}
	all_keys = []
	all_lo_pixel_reflectances = []
	all_lo_binned_pixel_reflectances = []
	all_ho_pixel_reflectances = []
	all_ho_binned_pixel_reflectances = []
	all_lo_signal_to_noise_ratios = []
	all_lo_binned_signal_to_noise_ratios = []
	all_ho_signal_to_noise_ratios = []
	all_ho_binned_signal_to_noise_ratios = []
	print(f'processing directories {grouped_acquisition_folders}')
	for key, order_dict in grouped_acquisition_folders.items():
		print(f'Processing key: {key}')
		# try:
		for order, vis_nir1_nir2_directories in order_dict.items():
			# print(vis_nir1_nir2_directories)
			# print(order)
			if len(vis_nir1_nir2_directories) == 2 and order == 'lo': # i.e. 2024-07-25_12-37-11_vis_h_vis_lo_600w_1875 missing
				lo_vis_directory = None
				lo_nir1_directory = vis_nir1_nir2_directories[0]
				lo_nir2_directory = vis_nir1_nir2_directories[1]
			else:
				if order == 'lo':
					lo_vis_directory = vis_nir1_nir2_directories[0]
					lo_nir1_directory = vis_nir1_nir2_directories[1]
					lo_nir2_directory = vis_nir1_nir2_directories[2]
				if order == 'ho':
					ho_vis_directory = vis_nir1_nir2_directories[0]
					ho_nir1_directory = vis_nir1_nir2_directories[1]
					ho_nir2_directory = vis_nir1_nir2_directories[2]
		print(f'Processing directories: {lo_vis_directory}, {lo_nir1_directory}, {lo_nir2_directory}, {ho_vis_directory}, {ho_nir1_directory}, {ho_nir2_directory}')
		directory_rectangle_averages, pixel_values, pixel_reflectances, signal_to_noise_ratios, lo_x_labels, ho_x_labels, num_coordinates = process_directory(
				metadata_parent_directory,
				actual_data_parent_directory,
				key,
				lo_vis_directory,
				lo_nir1_directory,
				lo_nir2_directory,
				ho_vis_directory,
				ho_nir1_directory,
				ho_nir2_directory,
				output_directory,
				pixel_coordinates,
				rectangle_coordinates,
				test_mode=test_mode
			)
		rectangle_averages.update(directory_rectangle_averages)
		all_keys.append(key)
		all_lo_pixel_reflectances.append(pixel_reflectances[0])
		all_lo_binned_pixel_reflectances.append(pixel_reflectances[1])
		all_ho_pixel_reflectances.append(pixel_reflectances[2])
		all_ho_binned_pixel_reflectances.append(pixel_reflectances[3])
		all_lo_signal_to_noise_ratios.append(signal_to_noise_ratios[0])
		all_lo_binned_signal_to_noise_ratios.append(signal_to_noise_ratios[1])
		all_ho_signal_to_noise_ratios.append(signal_to_noise_ratios[2])
		all_ho_binned_signal_to_noise_ratios.append(signal_to_noise_ratios[3])
		# except ValueError as e:
		# 	print(f"\nError processing directories for key {key}: {e}")
		# 	print("There may be an empty list in the dictionary.")
		# 	print("-> Skipping this key\n")
		# 	continue
	
	exposure_combination_graphs(
		all_keys,
		all_lo_pixel_reflectances,
		all_lo_binned_pixel_reflectances,
		all_ho_pixel_reflectances,
		all_ho_binned_pixel_reflectances,
		all_lo_signal_to_noise_ratios,
		all_lo_binned_signal_to_noise_ratios,
		all_ho_signal_to_noise_ratios,
		all_ho_binned_signal_to_noise_ratios,
		lo_x_labels,
		ho_x_labels,
		num_coordinates,
		output_directory,
		test_mode=test_mode,
	)

	power_combination_graphs(
		all_keys,
		all_lo_pixel_reflectances,
		all_lo_binned_pixel_reflectances,
		all_ho_pixel_reflectances,
		all_ho_binned_pixel_reflectances,
		all_lo_signal_to_noise_ratios,
		all_lo_binned_signal_to_noise_ratios,
		all_ho_signal_to_noise_ratios,
		all_ho_binned_signal_to_noise_ratios,
		lo_x_labels,
		ho_x_labels,
		num_coordinates,
		output_directory,
		test_mode=test_mode,
	)

	if not test_mode:
		df = pd.DataFrame(list(rectangle_averages.items()), columns=['File', 'Rectangle Average'])
		excel_path = os.path.join(output_directory, "rectangle_averages.xlsx")
		if not test_mode:	
			df.to_excel(excel_path, index=False)

def visualize_vis_nir1_nir2_pixel_coordinates(
		vis_file_path: str,
		nir1_file_path: str,
		nir2_file_path: str,
		output_directory: str,
		pixel_coordinates: dict,
		helper_lines: dict,
		rectangle_coordinates: dict,
		shapes=[(1024, 1024), (518, 648), (518, 648)],
        show_helper_lines: bool = True,
		test_mode: bool = False,
	):
	
	print(f'Visualizing pixel coordinates for {vis_file_path}, {nir1_file_path}, {nir2_file_path}')

	vis_image  = load_image(vis_file_path, shapes[0])
	nir1_image = load_image(nir1_file_path, shapes[1])
	nir2_image = load_image(nir2_file_path, shapes[2])
	
	# Get titles using the provided get_title function
	vis_title  = get_title(vis_file_path, format=2)
	nir1_title = get_title(nir1_file_path, format=2)
	nir2_title = get_title(nir2_file_path, format=2)
	
	# Create subplots for the three images
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	
	# VIS image subplot
	im0 = axes[0].imshow(vis_image, cmap='gray', aspect='equal')
	axes[0].set_title(vis_title)
	axes[0].set_xlabel('Columns')
	axes[0].set_ylabel('Rows')
	plt.colorbar(im0, ax=axes[0], label='Intensity')
	
	# NIR1 image subplot
	im1 = axes[1].imshow(nir1_image, cmap='gray', aspect='equal')
	axes[1].set_title(nir1_title)
	axes[1].set_xlabel('Columns')
	axes[1].set_ylabel('Rows')
	plt.colorbar(im1, ax=axes[1], label='Intensity')
	
	# NIR2 image subplot
	im2 = axes[2].imshow(nir2_image, cmap='gray', aspect='equal')
	axes[2].set_title(nir2_title)
	axes[2].set_xlabel('Columns')
	axes[2].set_ylabel('Rows')
	plt.colorbar(im2, ax=axes[2], label='Intensity')
	
	if show_helper_lines:
		# Iterate through each image key in helper_lines
		for key, lines in helper_lines.items():
			# Map keys to subplot axes
			if key == 'VIS':
				current_ax = axes[0]
			elif key == 'NIR1':
				current_ax = axes[1]
			elif key == 'NIR2':
				current_ax = axes[2]
			else:
				continue
			
			for idx, ((x_start, y_start), (x_end, y_end)) in enumerate(lines):
				current_ax.plot([x_start, x_end], [y_start, y_end], color='blue', linewidth=1, linestyle='-')
				current_ax.text(x_start+5, y_start+5, f'Line {idx} start', c='blue', fontsize=8)
				current_ax.text(x_end+5, y_end+5, f'Line {idx} end', c='blue', fontsize=8)

	# Plot the provided markers on each subplot
	for i in range(len(pixel_coordinates['VIS'])):
		# Unpack coordinates for the three images.
		vis_coordinates, nir1_coordinates, nir2_coordinates = pixel_coordinates['VIS'][i], pixel_coordinates['NIR1'][i], pixel_coordinates['NIR2'][i]
		
		# Plot markers on each subplot using scatter.
		axes[0].scatter(vis_coordinates[0], vis_coordinates[1], marker='o', s=1, c='red', label='Marker' if i==0 else "")
		axes[1].scatter(nir1_coordinates[0], nir1_coordinates[1], marker='o', s=1, c='red', label='Marker' if i==0 else "")
		axes[2].scatter(nir2_coordinates[0], nir2_coordinates[1], marker='o', s=1, c='red', label='Marker' if i==0 else "")
		
		# Optionally annotate markers with their index.
		axes[0].text(vis_coordinates[0] + 5, vis_coordinates[1] + 5, str(i), color='red', fontsize=8)
		axes[1].text(nir1_coordinates[0] + 5, nir1_coordinates[1] + 5, str(i), color='red', fontsize=8)
		axes[2].text(nir2_coordinates[0] + 5, nir2_coordinates[1] + 5, str(i), color='red', fontsize=8)

	# Unpack rectangle coordinates for each image.
	vis_rect, nir1_rect, nir2_rect = rectangle_coordinates['VIS'], rectangle_coordinates['NIR1'], rectangle_coordinates['NIR2']
	
	# Create a polygon patch for each rectangle.
	vis_patch = Polygon(vis_rect, closed=True, fill=False, edgecolor='blue', linewidth=1, label='Rectangle')
	nir1_patch = Polygon(nir1_rect, closed=True, fill=False, edgecolor='blue', linewidth=1, label='Rectangle')
	nir2_patch = Polygon(nir2_rect, closed=True, fill=False, edgecolor='blue', linewidth=1, label='Rectangle')
	
	# Add the rectangle patches to the corresponding axes.
	axes[0].add_patch(vis_patch)
	axes[1].add_patch(nir1_patch)
	axes[2].add_patch(nir2_patch)
	
	plt.tight_layout()
	plot_path = os.path.join(output_directory, "coordinate_locations.png")
	if test_mode:
		plt.show()
	else:
		plt.savefig(plot_path)
	
def visualize_pixel_coordinates(
		file_path: str,
		calibration_file_path: str,
		output_directory: str,
		pixel_coordinates: list,
		helper_lines: list,
		rectangle_coordinates: tuple,
		averaged_spectra_square_coordinates: tuple,
		image_shape: tuple, # new = [(1024, 1024), (506, 636), (506, 636)], original = [(1024, 1024), (518, 648), (518, 648)],
		calibration_image_shape: tuple,
		channel: str,
        show_helper_lines: bool = True,
		show_calibration_squares_in_calibration_file: bool = True,
		show_calibration_squares_in_main_image: bool = False,
		test_mode: bool = False,
	):

	if not test_mode:
		os.makedirs(output_directory, exist_ok=True)
	
	print(f'Visualizing pixel coordinates for {file_path}')

	image  = load_image(file_path, image_shape)
	
	# Get titles using the provided get_title function
	title  = get_title(file_path, format=0)
	
	# Create subplots for the three images
	fig, axes = plt.subplots(1, 1, figsize=(10, 6))
	
	# VIS image subplot
	im0 = axes.imshow(image, cmap='gray', aspect='equal')
	axes.set_title(title)
	plt.colorbar(im0, ax=axes, label='Intensity')

	if show_calibration_squares_in_main_image:
		patch = Polygon(rectangle_coordinates, closed=True, fill=False, edgecolor='blue', linewidth=1, label='Rectangle')
		axes.add_patch(patch)
	
	if show_helper_lines and not averaged_spectra_square_coordinates:
		for idx, ((x_start, y_start), (x_end, y_end)) in enumerate(helper_lines):
			axes.plot([x_start, x_end], [y_start, y_end], color='blue', linewidth=1, linestyle='-')
			axes.text(x_start+5, y_start+5, f'Line {idx} start', c='blue', fontsize=8)
			axes.text(x_end+5, y_end+5, f'Line {idx} end', c='blue', fontsize=8)

	if not averaged_spectra_square_coordinates:
		# Plot the provided markers on each subplot
		for i in range(len(pixel_coordinates)):
			# Unpack coordinates for the three images.
			coordinates = pixel_coordinates[i]
			
			# Plot markers on each subplot using scatter.
			axes.scatter(coordinates[0], coordinates[1], marker='o', s=1, c='red', label='Marker' if i==0 else "")
			
			# Optionally annotate markers with their index.
			axes.text(coordinates[0] + 5, coordinates[1] + 5, str(i), color='red', fontsize=8)

	if averaged_spectra_square_coordinates:
		patch = Polygon(averaged_spectra_square_coordinates, closed=True, fill=False, edgecolor='blue', linewidth=1, label='Rectangle')
		axes.add_patch(patch)
	
	plt.tight_layout()
	file_name = get_title(file_path, format=4)
	plot_path = os.path.join(output_directory, f"{file_name}_spectra_locations.png")
	if test_mode:
		plt.show()
	else:
		plt.savefig(plot_path)
		plt.close(fig)
		
	if show_calibration_squares_in_calibration_file:
		if channel == 'vis':
			calibration_image  = load_image(os.path.join(calibration_file_path, 'dc_0_exp_005.bin'), calibration_image_shape)
		elif channel == 'nir1':
			calibration_image  = load_image(os.path.join(calibration_file_path, 'dc_1_exp_005.bin'), calibration_image_shape)
		elif channel == 'nir2':
			calibration_image  = load_image(os.path.join(calibration_file_path, 'dc_2_exp_005.bin'), calibration_image_shape)
			
		title  = get_title(calibration_file_path, format=0)

		fig, axes = plt.subplots(1, 1, figsize=(10, 6))
		im0 = axes.imshow(calibration_image, cmap='gray', aspect='equal')
		axes.set_title(title)
		plt.colorbar(im0, ax=axes, label='Intensity')
		
		if type(rectangle_coordinates) == tuple:
			# Create a polygon patch for each rectangle.
			patch = Polygon(rectangle_coordinates, closed=True, fill=False, edgecolor='blue', linewidth=1, label='Rectangle')
			
			# Add the rectangle patches to the corresponding axes.
			axes.add_patch(patch)
		elif type(rectangle_coordinates) == list:
			for i in range(len(rectangle_coordinates)):
				coordinates = rectangle_coordinates[i]
				axes.scatter(coordinates[0], coordinates[1], marker='o', s=1, c='red', label='Marker' if i==0 else "")
				axes.text(coordinates[0] + 5, coordinates[1] + 5, str(i), color='red', fontsize=8)
		elif not rectangle_coordinates:
			for i in range(len(pixel_coordinates)):
				coordinates = pixel_coordinates[i]
				axes.scatter(coordinates[0], coordinates[1], marker='o', s=1, c='red', label='Marker' if i==0 else "")
				axes.text(coordinates[0] + 5, coordinates[1] + 5, str(i), color='red', fontsize=8)

		plt.tight_layout()
		file_name = get_title(calibration_file_path, format=3)
		plot_path = os.path.join(output_directory, f"{file_name}_calibration_locations.png")
		if test_mode:
			plt.show()
		else:
			plt.savefig(plot_path)
			plt.close(fig)

def find_target_pixels(coordinate_helper_lines, pixel_coordinates_based_on_helper_lines):
	target_pixels_dict = {}
	
	for channel, lines in coordinate_helper_lines.items():
		percentages = pixel_coordinates_based_on_helper_lines[channel]
		# List to store the target information for each line
		band_targets = []
		
		for line, percentage_list in zip(lines, percentages):
			for percent in percentage_list:
				# Unpack the start and end coordinate for the current line
				(x_start, y_start), (x_end, y_end) = line
				
				# Convert to NumPy arrays for vectorized operations (if needed)
				start = np.array([x_start, y_start], dtype=float)
				end = np.array([x_end, y_end], dtype=float)
				
				# Compute the target coordinate using linear interpolation:
				# target = start + t * (end - start)
				target_coord = start + percent * (end - start)
				x_target, y_target = target_coord
				
				# To index into the image array, round to the nearest integer.
				# Note: NumPy image arrays typically use row, col order, so use y then x.
				x_int = int(round(x_target))
				y_int = int(round(y_target))
				
				# Append the computed information for this helper line
				band_targets.append((x_int, y_int))
			
		target_pixels_dict[channel] = band_targets
	
	return target_pixels_dict

def find_target_pixels_of_single_image(
		coordinate_helper_lines: list, # [((387,677),(568,535)), ((554,714),(458,470)), ((554,714),(420,612))]
		pixel_coordinates_based_on_helper_lines: list # [[0.29, 0.525, 0.7, 0.83], [0.73, 0.88], [0.4, 0.555]]
	):
	if coordinate_helper_lines == None or pixel_coordinates_based_on_helper_lines == None:
		return None
	
	target_pixels = []
	
	for i, line in enumerate(coordinate_helper_lines): # ((387,677),(568,535))
		percentages = pixel_coordinates_based_on_helper_lines
		# List to store the target information for each line
		for percent in percentages[i]: # 0.29
			# Unpack the start and end coordinate for the current line
			(x_start, y_start), (x_end, y_end) = line
			
			# Convert to NumPy arrays for vectorized operations (if needed)
			start = np.array([x_start, y_start], dtype=float)
			end = np.array([x_end, y_end], dtype=float)
			
			# Compute the target coordinate using linear interpolation:
			# target = start + t * (end - start)
			target_coord = start + percent * (end - start)
			x_target, y_target = target_coord
			
			# To index into the image array, round to the nearest integer.
			# Note: NumPy image arrays typically use row, col order, so use y then x.
			x_int = int(round(x_target))
			y_int = int(round(y_target))
			
			# Append the computed information for this helper line
			target_pixels.append((x_int, y_int))
	
	return target_pixels

def match_channels(acquisition_folders):
	exposures = [[1250, 5000, 5000], [1875, 7500, 7500], [2500, 10000, 10000]] # [625, 2500, 2500],
	watts = ['200w', '400w', '600w']
	orders = ['lo', 'ho']

	groups = {}
	for w in watts:
		for vis_e, nir1_e, nir2_e in exposures:
			vis_candidates = [
				f for f in acquisition_folders
				if f"_{w}_{vis_e}" in f and "vis" in f and "nir" not in f
			]
			nir1_candidates = [
				f for f in acquisition_folders
				if f"_{w}_{nir1_e}" in f and "nir1" in f
			]
			nir2_candidates = [
				f for f in acquisition_folders
				if f"_{w}_{nir2_e}" in f and "nir2" in f
			]
			if not vis_candidates and not nir1_candidates:
				continue

			key = f'{w}_exposures_{vis_e}-{nir1_e}-{nir2_e}'
			groups[key] = {'lo': [], 'ho': []}

			lo_vis_match = next((f for f in vis_candidates if f"_lo_" in f), None)
			lo_nir1_match = next((f for f in nir1_candidates if f"_lo_" in f), None)
			lo_nir2_match = next((f for f in nir2_candidates if f"_lo_" in f), None)
			ho_vis_match = next((f for f in vis_candidates if f"_ho_" in f), None)
			ho_nir1_match = next((f for f in nir1_candidates if f"_ho_" in f), None)
			ho_nir2_match = next((f for f in nir2_candidates if f"_ho_" in f), None)
			
			if lo_vis_match and lo_nir1_match and lo_nir2_match:
				groups[key]['lo'] = [lo_vis_match, lo_nir1_match, lo_nir2_match]
			elif lo_nir1_match and lo_nir2_match:
				groups[key]['lo'] = [lo_nir1_match, lo_nir2_match]
			if ho_vis_match and ho_nir1_match and ho_nir2_match:
				groups[key]['ho'] = [ho_vis_match, ho_nir1_match, ho_nir2_match]
	return groups

# This is deprecated and related to the process_ASPECT_TESTS_25_7 section below

# coordinate_helper_lines = {
# 	'VIS': [((387,677),(568,535)), ((554,714),(458,470)), ((554,714),(420,612))],
# 	'NIR1': [((420-6,294-6),(260-6,165-6)), ((273-6,333-6),(363-6,109-6)), ((273-6,333-6),(394-6,229-6))],
# 	'NIR2': [((357-6,243-6),(513-6,360-6)), ((502-6,199-6),(414-6,431-6)), ((502-6,199-6),(381-6,297-6))]
# }
# pixel_coordinates_based_on_helper_lines = { # Percentage of line length from start
# 	'VIS': [[0.29, 0.525, 0.7, 0.83], [0.73, 0.88], [0.4, 0.555]],
# 	'NIR1': [[0.29, 0.525, 0.7, 0.83], [0.73, 0.88], [0.4, 0.555]],
# 	'NIR2': [[0.29, 0.525, 0.7, 0.83], [0.73, 0.88], [0.4, 0.555]]
# }
# rectangle_coordinates = {
# 	'VIS': ((445,400), (450,400), (450,405), (445,405)),
# 	'NIR1': ((365,45), (370,45), (370,50), (365,50)),
# 	'NIR2': ((405,480), (410,480), (410,485), (405,485))
# }

# This is deprecated!

# process_ASPECT_TESTS_25_7 = False
# if process_ASPECT_TESTS_25_7:
# 	timestamp = time.time()
# 	local_time = time.localtime(timestamp)
# 	formatted_time = time.strftime("%Y-%m-%dT%H:%M:%S", local_time)

# 	target_pixels = find_target_pixels(
# 		coordinate_helper_lines,
# 		pixel_coordinates_based_on_helper_lines
# 	)

# 	grouped_acquisition_folders = match_channels(ASPECT_TESTS_25_7_non_dark)

# 	process_directories(
# 		'/home/sysa/HERA/test_data/ASPECT_TESTS_25.7',
# 		'/home/sysa/HERA/test_data/test_outputs/Dark_after-subtracted(ASPECT_TESTS_25.7)',
# 		grouped_acquisition_folders,
# 		f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_pixel_comparisons',
# 		target_pixels,#{'VIS': [(480,506),(530,677)], 'NIR1': [(347,132),(295,298)], 'NIR2': [(438,397),(481,242)]},
# 		rectangle_coordinates,
# 		test_mode=False
# 	)
# 	visualize_vis_nir1_nir2_pixel_coordinates(
# 		# '/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_12-04-30_vis_h_vis_ho_600w_2500/acq_000/dc_0_exp_007.bin',
# 		# '/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_14-23-40_nir1_h_nir1_ho_600w_10000/acq_000/dc_1_exp_007.bin',
# 		# '/home/sysa/HERA/test_data/ASPECT_TESTS_25.7/2024-07-25_15-25-02_nir2_h_nir2_ho_600w_10000/acq_000/dc_2_exp_007.bin',
# 		'/home/sysa/HERA/test_data/test_outputs/Dark_after-subtracted(ASPECT_TESTS_25.7)/2024-07-25_12-04-30_vis_h_vis_ho_600w_2500/dc_0_exp_007.bin',
# 		'/home/sysa/HERA/test_data/test_outputs/Dark_after-subtracted(ASPECT_TESTS_25.7)/2024-07-25_14-23-40_nir1_h_nir1_ho_600w_10000/dc_1_exp_007.bin',
# 		'/home/sysa/HERA/test_data/test_outputs/Dark_after-subtracted(ASPECT_TESTS_25.7)/2024-07-25_15-25-02_nir2_h_nir2_ho_600w_10000/dc_2_exp_007.bin',
# 		f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_pixel_comparisons',
# 		target_pixels,
# 		coordinate_helper_lines,
# 		rectangle_coordinates,
# 		# shapes = [(1024, 1024), (518, 648), (518, 648)],
# 		shapes = [(1024, 1024), (506, 636), (506, 636)],
# 		show_helper_lines=True,
# 		test_mode=False
# 	)

def process_ESA_meteorite_directory(
		metadata_directory: str,
		actual_data_directory: str,
		matching_dark_directory: str,
		output_directory: str,
		pixel_coordinates: list,# [(480,506),(530,677)] or None
		rectangle_coordinates: tuple,
		averaged_spectra_square_coordinates: tuple,
		calibration_data_directory: str,
		save_plot: bool = True,
		test_mode: bool = False,
	):
	if not test_mode:
		os.makedirs(output_directory, exist_ok=True)

	calib_path = os.path.join(metadata_directory, "calib.json")
	config_path = os.path.join(metadata_directory, "config.json")

	channel = read_channel(calib_path)

	vis_height = 1024
	vis_width = 1024
	nir_height = 506#518
	nir_width = 636#648

	if channel.lower() == 'vis':
		height = vis_height
		width = vis_width
		original_height = vis_height
		original_width = vis_width
	elif channel.lower() in ['nir1', 'nir2']:
		height = nir_height
		width = nir_width
		original_height = 518
		original_width = 648

	vis_edge_correction = 0
	nir1_edge_correction = 0
	nir2_edge_correction = -36
	
	# df_fs, wl_lookup = parse_PFM_and_FS_default_wl_and_setpoints('/home/sysa/HERA/test_data/PFM_and_FS_default_wl_and_setpoints.xlsx') # Old

	acquisition_wavelengths = []
	task_file_name = channel.lower() + 'TaskFile'
	with open(config_path, 'r') as file:
		data = json.load(file)
	task_file = data[task_file_name]
	task_file_sequences = [task_file[i:i+8] for i in range(0, len(task_file), 8)]
	order = 'unknown'
	if channel.lower() == 'vis':
		if task_file_sequences[0][1] == 17549:
			order = 'lo'
		elif task_file_sequences[0][1] == 19939:
			order = 'ho'
	elif channel.lower() == 'nir1':
		if task_file_sequences[0][1] == 17555:
			order = 'lo'
		elif task_file_sequences[0][1] == 20628:
			order = 'ho'
	elif channel.lower() == 'nir2':
		if task_file_sequences[0][1] == 18955:
			order = 'lo'
		elif task_file_sequences[0][1] == 23332:
			order = 'ho'
		# elif task_file_sequences[0][1] > 23000:
		# 	order = 'ho'
	if order == 'unknown':
		raise ValueError(f"Could not conclude order (lo/ho) from first setpoint: {task_file_sequences[0][1]}")
	for sequence in task_file_sequences:
		# acquisition_wavelengths.append(wl_lookup[channel][order][sequence[1]]) # Old
		if channel.lower() == 'vis':
			if order == 'lo':
				wl = vis_lo_setpoint_to_wavelength(sequence[1], vis_edge_correction)
			elif order == 'ho':
				wl = vis_ho_setpoint_to_wavelength(sequence[1], vis_edge_correction)
		elif channel.lower() == 'nir1':
			if order == 'lo':
				wl = nir1_lo_setpoint_to_wavelength(sequence[1], nir1_edge_correction)
			elif order == 'ho':
				wl = nir1_ho_setpoint_to_wavelength(sequence[1], nir1_edge_correction)
		elif channel.lower() == 'nir2':
			if order == 'lo':
				wl = nir2_lo_setpoint_to_wavelength(sequence[1], nir2_edge_correction)
			elif order == 'ho':
				wl = nir2_ho_setpoint_to_wavelength(sequence[1], nir2_edge_correction)
		acquisition_wavelengths.append(wl)

	bin_files = sorted([f for f in os.listdir(actual_data_directory) if f.endswith('.bin')])

	# Rectangles
	calibration_bin_files = sorted([f for f in os.listdir(calibration_data_directory) if f.endswith('.bin')])
	
	actual_data_directory_naming_key = get_title(actual_data_directory, format=3)
	
	# rectangle_averages = get_rectangle_averages(
	# 	bin_files,
	# 	actual_data_directory,
	# 	rectangle_coordinates,
	# 	actual_data_directory_naming_key,
	# 	height,
	# 	width
	# )

	if not rectangle_coordinates:
		calibration_pixel_values = get_calibration_pixel_values_from_different_file(
			bin_files,
			calibration_bin_files,
			calibration_data_directory,
			pixel_coordinates,
			actual_data_directory_naming_key,
			height,
			width
		)
	if type(rectangle_coordinates) == list:
		calibration_pixel_values = get_calibration_pixel_values_from_different_file(
			bin_files,
			calibration_bin_files,
			calibration_data_directory,
			rectangle_coordinates,
			actual_data_directory_naming_key,
			height,
			width
		)
	elif type(rectangle_coordinates) == tuple:
		rectangle_averages = get_rectangle_averages_from_different_file(
			bin_files,
			calibration_bin_files,
			calibration_data_directory,
			rectangle_coordinates,
			actual_data_directory_naming_key,
			height,
			width
		)

	# Pixel Values
	if not pixel_coordinates:
		num_coordinates = 1
	else:
		num_coordinates = len(pixel_coordinates)
	fig, axes = plt.subplots(num_coordinates, 1, figsize=(7, 2*num_coordinates), sharex=True)
	if num_coordinates == 1:
		axes = [axes]
	min_y_value, max_y_value = float('inf'), float('-inf')
	all_coordinates_pixel_values = []
	all_coordinates_binning_pixel_values = []

	if not pixel_coordinates:
		pixel_values = []
		binning_pixel_values = []
		for j, bin_file in enumerate(bin_files):
			file_path = os.path.join(actual_data_directory, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((height, width))
				if len(averaged_spectra_square_coordinates) != 4:
					raise ValueError("averaged_spectra_square_coordinates: Rectangle coordinates must contain exactly four points.")
				xs = [point[0] for point in averaged_spectra_square_coordinates]
				ys = [point[1] for point in averaged_spectra_square_coordinates]
				x_min = min(xs)
				x_max = max(xs)
				y_min = min(ys)
				y_max = max(ys)
				cropped_region = img_array[y_min:y_max+1, x_min:x_max+1]
				average_value = np.mean(cropped_region)
				pixel_values.append(average_value)
				binning_pixel_values.append(average_value)

		ax = axes[0]
		all_coordinates_pixel_values.append(pixel_values)
		all_coordinates_binning_pixel_values.append(binning_pixel_values)
		ax.plot(range(len(bin_files)), pixel_values, c='blue')
		ax.set_xticks(range(len(bin_files)))
		ax.set_xticklabels(bin_files, rotation=40, ha='right', fontsize=12)
		ax.set_ylabel("Pixel Value")
		ax.set_title(f"Average")
		ax.grid(True, axis='y')
		if min(pixel_values) < min_y_value:
			min_y_value = min(pixel_values)
		if max(pixel_values) > max_y_value:
			max_y_value = max(pixel_values)
	else:
		for i, pixel_coordinate in enumerate(pixel_coordinates):
			pixel_values = []
			binning_pixel_values = []
			for j, bin_file in enumerate(bin_files):
				file_path = os.path.join(actual_data_directory, bin_file)
				with open(file_path, 'rb') as file:
					bin_data = file.read()
					img_array = np.frombuffer(bin_data, dtype=np.uint16)
					img_array = img_array.reshape((height, width))
					pixel1 = img_array[pixel_coordinate[1], pixel_coordinate[0]]
					pixel_values.append(pixel1)
					pixel2 = img_array[pixel_coordinate[1], pixel_coordinate[0]+1]
					pixel3 = img_array[pixel_coordinate[1]+1, pixel_coordinate[0]]
					pixel4 = img_array[pixel_coordinate[1]+1, pixel_coordinate[0]+1]
					average_pixel = (np.uint32(pixel1) + np.uint32(pixel2) + np.uint32(pixel3) + np.uint32(pixel4)) / 4
					binning_pixel_values.append(average_pixel)

			ax = axes[i]
			all_coordinates_pixel_values.append(pixel_values)
			all_coordinates_binning_pixel_values.append(binning_pixel_values)
			ax.plot(range(len(bin_files)), pixel_values, c='blue')
			ax.set_xticks(range(len(bin_files)))
			ax.set_xticklabels(bin_files, rotation=40, ha='right', fontsize=12)
			ax.set_ylabel("Pixel Value")
			ax.set_title(f"Coordinate {i}")
			ax.grid(True, axis='y')
			if min(pixel_values) < min_y_value:
				min_y_value = min(pixel_values)
			if max(pixel_values) > max_y_value:
				max_y_value = max(pixel_values)
	
	if num_coordinates == 1:
		axes[0].set_ylim(0, max_y_value+max_y_value*0.05)
	else:
		for axr in axes.flatten():
			axr.set_ylim(0, max_y_value+max_y_value*0.05) # Optionally min_y_value instead of 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"{actual_data_directory_naming_key}_pixel_values.png")
	if test_mode:
		# pass
		plt.show()
	elif save_plot:
		plt.savefig(plot_path)
	plt.close(fig)
	min_y_pixel_value = min_y_value
	max_y_pixel_value = max_y_value
	
	# Reflectances
	unique_idxs = [
		i for i in range(len(acquisition_wavelengths)) if i == len(acquisition_wavelengths)-1 or acquisition_wavelengths[i] != acquisition_wavelengths[i+1]
	]
	filtered_wls = [acquisition_wavelengths[i] for i in unique_idxs]
	filtered_binfiles = [bin_files[i] for i in unique_idxs]
	x_labels = [f"{wl}" for wl, bf in zip(filtered_wls, filtered_binfiles)]
	
	fig_ref, axes_ref = plt.subplots(num_coordinates, 1, figsize=(7, 2*num_coordinates), sharex=True)
	min_y_value, max_y_value = float('inf'), float('-inf')
	if num_coordinates == 1:
		axes_ref = [axes_ref]

	# Low Order
	# x_labels
	n_wls = len(x_labels)
	n_coords = num_coordinates
	all_coordinates_pixel_reflectances = []
	all_coordinates_binning_pixel_reflectances = []
	# all_coordinates_ho_pixel_reflectances = []
	# all_coordinates_ho_binning_pixel_reflectances = []
	reflectance_matrix = np.zeros((n_coords, n_wls), dtype=float)
	binning_reflectance_matrix = np.zeros((n_coords, n_wls), dtype=float)
	
	if not pixel_coordinates:
		pixel_reflectances = []
		binning_pixel_reflectances = []
		for wl, bin_file in zip(filtered_wls, filtered_binfiles):
			path = os.path.join(actual_data_directory, bin_file)
			with open(path, 'rb') as f:
				buf = f.read()
			img = np.frombuffer(buf, dtype=np.uint16).reshape((height, width))
			if len(averaged_spectra_square_coordinates) != 4:
				raise ValueError("averaged_spectra_square_coordinates: Rectangle coordinates must contain exactly four points.")
			xs = [point[0] for point in averaged_spectra_square_coordinates]
			ys = [point[1] for point in averaged_spectra_square_coordinates]
			x_min = min(xs)
			x_max = max(xs)
			y_min = min(ys)
			y_max = max(ys)
			cropped_region = img[y_min:y_max+1, x_min:x_max+1]
			average_value = np.mean(cropped_region)
			pixel_reflectance = reflectance(average_value, rectangle_averages[os.path.join(actual_data_directory_naming_key, bin_file)])
			binning_pixel_reflectance = reflectance(average_value, rectangle_averages[os.path.join(actual_data_directory_naming_key, bin_file)])
			binning_pixel_reflectances.append(binning_pixel_reflectance)
			pixel_reflectances.append(pixel_reflectance)
		
		all_coordinates_pixel_reflectances.append(pixel_reflectances)
		all_coordinates_binning_pixel_reflectances.append(binning_pixel_reflectances)
		reflectance_matrix[0, :] = pixel_reflectances
		binning_reflectance_matrix[0, :] = binning_pixel_reflectances
		axr = axes_ref[0]
		axr.plot(range(len(x_labels)), binning_pixel_reflectances)
		axr.set_xticks(range(len(x_labels)))
		axr.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=12)
		axr.set_ylabel("Pixel Reflectance")
		axr.set_title(f"Average")
		axr.grid(True, axis='y')
		if min(binning_pixel_reflectances) < min_y_value:
			min_y_value = min(binning_pixel_reflectances)
		if max(binning_pixel_reflectances) > max_y_value:
			max_y_value = max(binning_pixel_reflectances)
	else:
		binning = 3 # Set main meteorite binning size, example: 3 for 3x3 or None for single pixel
		for i, (x_coord, y_coord) in enumerate(pixel_coordinates):
			pixel_reflectances = []
			binning_pixel_reflectances = []
			for wl, bin_file in zip(filtered_wls, filtered_binfiles):
				path = os.path.join(actual_data_directory, bin_file)
				with open(path, 'rb') as f:
					buf = f.read()
				img = np.frombuffer(buf, dtype=np.uint16).reshape((height, width))
				pix1 = img[y_coord, x_coord]
				pix2 = img[y_coord+1, x_coord]
				pix3 = img[y_coord, x_coord+1]
				pix4 = img[y_coord+1, x_coord+1]
				average_pix = (np.uint32(pix1) + np.uint32(pix2) + np.uint32(pix3) + np.uint32(pix4)) / 4
				
				if binning:
					(top_left_x, top_left_y), (top_right_x, top_right_y), (bottom_right_x, bottom_right_y), (bottom_left_x, bottom_left_y) = binning_coordinates((x_coord, y_coord), binning)
					cropped_region = img[top_left_y:bottom_left_y+1, top_left_x:top_right_x+1]
					average_value = np.mean(cropped_region)
				else:
					average_value = img[y_coord, x_coord]
				if type(rectangle_coordinates) == tuple:
					pixel_reflectance = reflectance(average_value, rectangle_averages[os.path.join(actual_data_directory_naming_key, bin_file)]) # Binning or not?
					binning_pixel_reflectance = reflectance(average_pix, rectangle_averages[os.path.join(actual_data_directory_naming_key, bin_file)]) # Old
				else:
					pixel_reflectance = reflectance(average_value, calibration_pixel_values[os.path.join(actual_data_directory_naming_key, bin_file)][i]) # Binning or not?
					binning_pixel_reflectance = reflectance(average_pix, calibration_pixel_values[os.path.join(actual_data_directory_naming_key, bin_file)][i]) # Old
				binning_pixel_reflectances.append(binning_pixel_reflectance)
				pixel_reflectances.append(pixel_reflectance)
			
			all_coordinates_pixel_reflectances.append(pixel_reflectances)
			all_coordinates_binning_pixel_reflectances.append(binning_pixel_reflectances)
			reflectance_matrix[i, :] = pixel_reflectances
			binning_reflectance_matrix[i, :] = binning_pixel_reflectances
			axr = axes_ref[i]
			axr.plot(range(len(x_labels)), pixel_reflectances)
			axr.set_xticks(range(len(x_labels)))
			axr.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=12)
			axr.set_ylabel("Pixel Reflectance")
			axr.set_title(f"Coordinate {i}")
			axr.grid(True, axis='y')
			if min(pixel_reflectances) < min_y_value:
				min_y_value = min(pixel_reflectances)
			if max(pixel_reflectances) > max_y_value:
				max_y_value = max(pixel_reflectances)

	df = pd.DataFrame(reflectance_matrix, columns=x_labels)
	excel_path = os.path.join(output_directory, f"{actual_data_directory_naming_key}_pixel_reflectances(single_pixel).xlsx")
	if not test_mode:
		df.to_excel(excel_path, index=False)
	df = pd.DataFrame(binning_reflectance_matrix, columns=x_labels)
	excel_path = os.path.join(output_directory, f"{actual_data_directory_naming_key}_pixel_reflectances(4-pixel_binning).xlsx")
	if not test_mode:
		df.to_excel(excel_path, index=False)

	if num_coordinates == 1:
		axes_ref[0].set_ylim(min_y_value, max_y_value+max_y_value*0.05)
	else:
		for axr in axes_ref.flatten():
			axr.set_ylim(min_y_value, max_y_value+max_y_value*0.05)

	plt.tight_layout()
	out_ref = os.path.join(output_directory, f"{actual_data_directory_naming_key}_pixel_reflectances.png")
	if test_mode:
		# pass
		plt.show()
	elif save_plot:
		fig_ref.savefig(out_ref)
	plt.close(fig_ref)
	min_y_reflectance = min_y_value
	max_y_reflectance = max_y_value

	# Signal to Noise Ratio
	fig, axes = plt.subplots(num_coordinates, 1, figsize=(7, 2*num_coordinates), sharex=True)
	if num_coordinates == 1:
		axes = [axes]
	min_y_value, max_y_value = float('inf'), float('-inf')
	all_coordinates_signal_to_noise_ratios = []
	all_coordinates_binning_signal_to_noise_ratios = []
	# Low Order
	if not pixel_coordinates:
		signal_to_noise_ratios = []
		binning_signal_to_noise_ratios = []
		for j, bin_file in enumerate(filtered_binfiles):
			file_path = os.path.join(actual_data_directory, bin_file)
			with open(file_path, 'rb') as file:
				bin_data = file.read()
				img_array = np.frombuffer(bin_data, dtype=np.uint16)
				img_array = img_array.reshape((height, width))
				if len(averaged_spectra_square_coordinates) != 4:
					raise ValueError("averaged_spectra_square_coordinates: Rectangle coordinates must contain exactly four points.")
				xs = [point[0] for point in averaged_spectra_square_coordinates]
				ys = [point[1] for point in averaged_spectra_square_coordinates]
				x_min = min(xs)
				x_max = max(xs)
				y_min = min(ys)
				y_max = max(ys)
				cropped_region = img_array[y_min:y_max+1, x_min:x_max+1]
				average_value = np.mean(cropped_region)
				standard_deviation = ESA_meteorite_average_std_from_excel(matching_dark_directory)
				signal_to_noise_ratio = calculate_signal_to_noise_ratio(average_value, standard_deviation)
				binning_signal_to_noise_ratio = calculate_signal_to_noise_ratio(average_value, standard_deviation)
				binning_signal_to_noise_ratios.append(binning_signal_to_noise_ratio)
				signal_to_noise_ratios.append(signal_to_noise_ratio)

		ax = axes[0]
		all_coordinates_signal_to_noise_ratios.append(signal_to_noise_ratios)
		all_coordinates_binning_signal_to_noise_ratios.append(binning_signal_to_noise_ratios)
		ax.plot(range(len(x_labels)), binning_signal_to_noise_ratios, c='blue')
		ax.set_xticks(range(len(x_labels)))
		ax.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=12)
		ax.set_ylabel("Signal to Noise Ratio")
		ax.set_title(f"Average")
		ax.grid(True, axis='y')
		if min(binning_signal_to_noise_ratios) < min_y_value:
			min_y_value = min(binning_signal_to_noise_ratios)
		if max(binning_signal_to_noise_ratios) > max_y_value:
			max_y_value = max(binning_signal_to_noise_ratios)
	else:
		for i, pixel_coordinate in enumerate(pixel_coordinates):
			signal_to_noise_ratios = []
			binning_signal_to_noise_ratios = []
			for j, bin_file in enumerate(filtered_binfiles):
				file_path = os.path.join(actual_data_directory, bin_file)
				with open(file_path, 'rb') as file:
					bin_data = file.read()
					img_array = np.frombuffer(bin_data, dtype=np.uint16)
					img_array = img_array.reshape((height, width))
					pixel1 = img_array[pixel_coordinate[1], pixel_coordinate[0]]
					pixel2 = img_array[pixel_coordinate[1], pixel_coordinate[0]+1]
					pixel3 = img_array[pixel_coordinate[1]+1, pixel_coordinate[0]]
					pixel4 = img_array[pixel_coordinate[1]+1, pixel_coordinate[0]+1]
					average_pixel = (np.uint32(pixel1) + np.uint32(pixel2) + np.uint32(pixel3) + np.uint32(pixel4)) / 4
					standard_deviation = ESA_meteorite_average_std_from_excel(matching_dark_directory)
					signal_to_noise_ratio = calculate_signal_to_noise_ratio(pixel1, standard_deviation)
					binning_signal_to_noise_ratio = calculate_signal_to_noise_ratio(average_pixel, standard_deviation)
					binning_signal_to_noise_ratios.append(binning_signal_to_noise_ratio)
					signal_to_noise_ratios.append(signal_to_noise_ratio)

			ax = axes[i]
			all_coordinates_signal_to_noise_ratios.append(signal_to_noise_ratios)
			all_coordinates_binning_signal_to_noise_ratios.append(binning_signal_to_noise_ratios)
			ax.plot(range(len(x_labels)), binning_signal_to_noise_ratios, c='blue')
			ax.set_xticks(range(len(x_labels)))
			ax.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=12)
			ax.set_ylabel("Signal to Noise Ratio")
			ax.set_title(f"Coordinate {i}")
			ax.grid(True, axis='y')
			if min(binning_signal_to_noise_ratios) < min_y_value:
				min_y_value = min(binning_signal_to_noise_ratios)
			if max(binning_signal_to_noise_ratios) > max_y_value:
				max_y_value = max(binning_signal_to_noise_ratios)
	
	if num_coordinates == 1:
		axes[0].set_ylim(0, max_y_value+max_y_value*0.05)
	else:
		for axr in axes.flatten():
			axr.set_ylim(0, max_y_value+max_y_value*0.05) # Optionally min_y_value instead of 0
	plt.tight_layout()
	plot_path = os.path.join(output_directory, f"{actual_data_directory_naming_key}_signal_to_noise_ratio.png")
	if test_mode:
		# pass
		plt.show()
	elif save_plot:
		plt.savefig(plot_path)
	plt.close(fig)
	min_y_signal_to_noise_ratio = min_y_value
	max_y_signal_to_noise_ratio = max_y_value

	if type(rectangle_coordinates) == tuple:
		return rectangle_averages, all_coordinates_pixel_values, all_coordinates_pixel_reflectances, all_coordinates_signal_to_noise_ratios, x_labels, num_coordinates, min_y_pixel_value, max_y_pixel_value, min_y_reflectance, max_y_reflectance, min_y_signal_to_noise_ratio, max_y_signal_to_noise_ratio
	else:
		return calibration_pixel_values, all_coordinates_pixel_values, all_coordinates_pixel_reflectances, all_coordinates_signal_to_noise_ratios, x_labels, num_coordinates, min_y_pixel_value, max_y_pixel_value, min_y_reflectance, max_y_reflectance, min_y_signal_to_noise_ratio, max_y_signal_to_noise_ratio

def process_directories_with_specified_metadata(
		metadata_directories: list,
		actual_data_directories: list,
		matching_dark_directories: list,
		output_directories: str,
		pixel_coordinates: list,
		rectangle_coordinates: list,
		test_mode: bool = False,
	):
	
	for metadata_directory, actual_data_directory, matching_dark_directory, output_directory, pixel_coordinate, rectangle_coordinate in zip(metadata_directories, actual_data_directories, matching_dark_directories, output_directories, pixel_coordinates, rectangle_coordinates):
		print(f'processing directory {actual_data_directory}')
		rectangle_averages, all_coordinates_pixel_values, all_coordinates_pixel_reflectances, all_coordinates_signal_to_noise_ratios, x_labels, num_coordinates, min_y_pixel_value, max_y_pixel_value, min_y_reflectance, max_y_reflectance, min_y_signal_to_noise_ratio, max_y_signal_to_noise_ratio = process_ESA_meteorite_directory(
				metadata_directory,
				actual_data_directory,
				matching_dark_directory,
				output_directory,
				pixel_coordinate,
				rectangle_coordinate,
				save_plot=True,
				test_mode=test_mode
			)
		print(f'finished processing directory {actual_data_directory}')
		print(f'Rectangle averages: {rectangle_averages}')
		print(f'pixel values: {all_coordinates_pixel_values}')
		print(f'pixel reflectances: {all_coordinates_pixel_reflectances}')
		print(f'signal to noise ratios: {all_coordinates_signal_to_noise_ratios}')
		print(f'x_labels: {x_labels}')
		print(f'num_coordinates: {num_coordinates}')

def plot_combined_spectra(
		title: str,
		y_label: str,
		x_labels: list,
		data: list,
		output_directory: str,
		min_y_value: float,
		max_y_value: float,
		test_mode: bool = False,
		save_plot: bool = True,
	):
	num_coordinates = len(data)
	fig, axes = plt.subplots(num_coordinates, 1, figsize=(20, 6*num_coordinates), sharex=False)
	if num_coordinates == 1:
		axes = [axes]
	for i, coordinate_data in enumerate(data):
		ax = axes[i]
		ax.plot(range(len(coordinate_data)), coordinate_data, c='blue')
		ax.set_xticks(range(len(x_labels)))
		ax.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=10)
		ax.set_ylabel(y_label)
		if num_coordinates == 1:
			ax.set_title(f"Average")
			df = pd.DataFrame([coordinate_data], columns=x_labels)
			df.to_excel(os.path.join(output_directory, f'{title}.xlsx'), index=False)
		else:
			ax.set_title(f"Coordinate {i}")
		ax.grid(True, axis='y')
		
	if num_coordinates == 1:
		axes[0].set_ylim(min_y_value, max_y_value)
	else:
		for axr in axes.flatten():
			axr.set_ylim(min_y_value, max_y_value)
	# plt.tight_layout()
	plot_path = os.path.join(output_directory, f"{title}.png")
	if test_mode:
		# pass
		plt.show()
	if save_plot:
		plt.savefig(plot_path)
	plt.close(fig)
	if num_coordinates != 1:
		df = pd.DataFrame(data, columns=x_labels)
		df.to_excel(os.path.join(output_directory, f'{title}.xlsx'), index=False)

def spectra_post_calibration(
		combined_all_coordinates_values, # [coordinate[spectra_with_all_channels]]]
		combined_x_labels, # [labels]
		nir_spectra_shifting,
		num_vis_x_ticks,
		num_nir1_x_ticks,
		num_nir2_x_ticks,
		vis_offset
	):

	combined_x_labels_int = [int(x) for x in combined_x_labels]

	nir1_nir2_border_wavelength = combined_x_labels_int[num_vis_x_ticks+num_nir1_x_ticks-1] + 0.5 * (combined_x_labels_int[num_vis_x_ticks+num_nir1_x_ticks] - combined_x_labels_int[num_vis_x_ticks+num_nir1_x_ticks-1])
	nir1_nir2_border_wavelength = round(nir1_nir2_border_wavelength)

	corrected_combined_all_coordinates_values = []

	if nir_spectra_shifting == 'nir1':
		for coordinate_spectra in combined_all_coordinates_values:
			corrected_nir1, offset = nir1_offset_correction(
				combined_x_labels_int[num_vis_x_ticks:num_vis_x_ticks+num_nir1_x_ticks],
				coordinate_spectra[num_vis_x_ticks:num_vis_x_ticks+num_nir1_x_ticks],
				combined_x_labels_int[num_vis_x_ticks+num_nir1_x_ticks:num_vis_x_ticks+num_nir1_x_ticks+num_nir2_x_ticks],
				coordinate_spectra[num_vis_x_ticks+num_nir1_x_ticks:num_vis_x_ticks+num_nir1_x_ticks+num_nir2_x_ticks],
				nir1_nir2_border_wavelength
			)
			new_vis_spectra = [x + vis_offset for x in coordinate_spectra[:num_vis_x_ticks]]
			new_coordinate_spectra = new_vis_spectra + list(corrected_nir1) + coordinate_spectra[num_vis_x_ticks+num_nir1_x_ticks:]
			new_coordinate_spectra.pop(num_vis_x_ticks-1)
			corrected_combined_all_coordinates_values.append(new_coordinate_spectra)

	elif nir_spectra_shifting == 'nir2':
		for coordinate_spectra in combined_all_coordinates_values:
			corrected_nir2, offset = nir2_offset_correction(
				combined_x_labels_int[num_vis_x_ticks:num_vis_x_ticks+num_nir1_x_ticks],
				coordinate_spectra[num_vis_x_ticks:num_vis_x_ticks+num_nir1_x_ticks],
				combined_x_labels_int[num_vis_x_ticks+num_nir1_x_ticks:num_vis_x_ticks+num_nir1_x_ticks+num_nir2_x_ticks],
				coordinate_spectra[num_vis_x_ticks+num_nir1_x_ticks:num_vis_x_ticks+num_nir1_x_ticks+num_nir2_x_ticks],
				nir1_nir2_border_wavelength
			)
			new_vis_spectra = [x + vis_offset for x in coordinate_spectra[:num_vis_x_ticks]]
			new_coordinate_spectra = new_vis_spectra + coordinate_spectra[num_vis_x_ticks:num_vis_x_ticks+num_nir1_x_ticks] + list(corrected_nir2)
			new_coordinate_spectra.pop(num_vis_x_ticks-1)
			corrected_combined_all_coordinates_values.append(new_coordinate_spectra)

	elif nir_spectra_shifting == None:
		for coordinate_spectra in combined_all_coordinates_values:
			new_vis_spectra = [x + vis_offset for x in coordinate_spectra[:num_vis_x_ticks]]
			new_coordinate_spectra = new_vis_spectra + coordinate_spectra[num_vis_x_ticks:]
			new_coordinate_spectra.pop(num_vis_x_ticks-1)
			corrected_combined_all_coordinates_values.append(new_coordinate_spectra)
	
	combined_x_labels.pop(num_vis_x_ticks-1)
	
	return corrected_combined_all_coordinates_values, combined_x_labels

def process_grouped_directories_with_specified_metadata(
		vis_metadata_directories: list,
		nir1_metadata_directories: list,
		nir2_metadata_directories: list,
		vis_actual_data_directories: list,
		nir1_actual_data_directories: list,
		nir2_actual_data_directories: list,
		vis_matching_dark_directories: list,
		nir1_matching_dark_directories: list,
		nir2_matching_dark_directories: list,
		output_directories: str,
		vis_pixel_coordinates: list,
		nir1_pixel_coordinates: list,
		nir2_pixel_coordinates: list,
		vis_rectangle_coordinates: list,
		nir1_rectangle_coordinates: list,
		nir2_rectangle_coordinates: list,
		vis_averaged_spectra_square_coordinates: list,
		nir1_averaged_spectra_square_coordinates: list,
		nir2_averaged_spectra_square_coordinates: list,
		vis_calibration_data_directories: list,
		nir1_calibration_data_directories: list,
		nir2_calibration_data_directories: list,
		nir_spectra_shifting_list: list,
		vis_offsets: list,
		test_mode: bool = False,
	):
	
	for vis_metadata_directory, nir1_metadata_directory, nir2_metadata_directory, vis_actual_data_directory, nir1_actual_data_directory, nir2_actual_data_directory, vis_matching_dark_directory, nir1_matching_dark_directory, nir2_matching_dark_directory, output_directory, vis_pixel_coordinate, nir1_pixel_coordinate, nir2_pixel_coordinate, vis_rectangle_coordinate, nir1_rectangle_coordinate, nir2_rectangle_coordinate, vis_averaged_spectra_square_coordinate, nir1_averaged_spectra_square_coordinate, nir2_averaged_spectra_square_coordinate, vis_calibration_data_directory, nir1_calibration_data_directory, nir2_calibration_data_directory, nir_spectra_shifting, vis_offset in zip(vis_metadata_directories, nir1_metadata_directories, nir2_metadata_directories, vis_actual_data_directories, nir1_actual_data_directories, nir2_actual_data_directories, vis_matching_dark_directories, nir1_matching_dark_directories, nir2_matching_dark_directories, output_directories, vis_pixel_coordinates, nir1_pixel_coordinates, nir2_pixel_coordinates, vis_rectangle_coordinates, nir1_rectangle_coordinates, nir2_rectangle_coordinates, vis_averaged_spectra_square_coordinates, nir1_averaged_spectra_square_coordinates, nir2_averaged_spectra_square_coordinates, vis_calibration_data_directories, nir1_calibration_data_directories, nir2_calibration_data_directories, nir_spectra_shifting_list, vis_offsets):

		print(f'Output directory name: {get_title(output_directory, format=3)}\nProcessing directories:\n{vis_actual_data_directory}\n{nir1_actual_data_directory}\n{nir2_actual_data_directory}\nMetadata directory:\n{get_title(vis_metadata_directory, format=4)}\n{get_title(nir1_metadata_directory, format=4)}\n{get_title(nir2_metadata_directory, format=4)}\nDark image directory:\n{vis_matching_dark_directory}\n{nir1_matching_dark_directory}\n{nir2_matching_dark_directory}\nCalibration directories:\n{vis_calibration_data_directory}\n{nir1_calibration_data_directory}\n{nir2_calibration_data_directory}\n')

		min_y_reflectance_of_all = float('inf')
		max_y_reflectance_of_all = float('-inf')
	
		if vis_actual_data_directory:
			# print(f'processing directory {vis_actual_data_directory}')
			directory_name = get_title(vis_actual_data_directory, format=3)
			single_directory_output_directory = os.path.join(output_directory, directory_name)
			vis_rectangle_averages, vis_all_coordinates_pixel_values, vis_all_coordinates_pixel_reflectances, vis_all_coordinates_signal_to_noise_ratios, vis_x_labels, vis_num_coordinates, min_y_pixel_value, max_y_pixel_value, min_y_reflectance, max_y_reflectance, min_y_signal_to_noise_ratio, max_y_signal_to_noise_ratio = process_ESA_meteorite_directory(
					vis_metadata_directory,
					vis_actual_data_directory,
					vis_matching_dark_directory,
					single_directory_output_directory,
					vis_pixel_coordinate,#[(480,506),(530,677)]
					vis_rectangle_coordinate,
					vis_averaged_spectra_square_coordinate,
					vis_calibration_data_directory,
					save_plot=True,
					test_mode=False
				)
			if min_y_reflectance < min_y_reflectance_of_all:
				min_y_reflectance_of_all = min_y_reflectance
			if max_y_reflectance > max_y_reflectance_of_all:
				max_y_reflectance_of_all = max_y_reflectance
		else:
			vis_rectangle_averages, vis_all_coordinates_pixel_values, vis_all_coordinates_pixel_reflectances, vis_all_coordinates_signal_to_noise_ratios, vis_x_labels, vis_num_coordinates = (None, None, None, None, None, None)
		if nir1_actual_data_directory:
			# print(f'processing directory {nir1_actual_data_directory}')
			directory_name = get_title(nir1_actual_data_directory, format=3)
			single_directory_output_directory = os.path.join(output_directory, directory_name)
			nir1_rectangle_averages, nir1_all_coordinates_pixel_values, nir1_all_coordinates_pixel_reflectances, nir1_all_coordinates_signal_to_noise_ratios, nir1_x_labels, nir1_num_coordinates, min_y_pixel_value, max_y_pixel_value, min_y_reflectance, max_y_reflectance, min_y_signal_to_noise_ratio, max_y_signal_to_noise_ratio = process_ESA_meteorite_directory(
					nir1_metadata_directory,
					nir1_actual_data_directory,
					nir1_matching_dark_directory,
					single_directory_output_directory,
					nir1_pixel_coordinate,
					nir1_rectangle_coordinate,
					nir1_averaged_spectra_square_coordinate,
					nir1_calibration_data_directory,
					save_plot=True,
					test_mode=False
				)
			if min_y_reflectance < min_y_reflectance_of_all:
				min_y_reflectance_of_all = min_y_reflectance
			if max_y_reflectance > max_y_reflectance_of_all:
				max_y_reflectance_of_all = max_y_reflectance
		else:
			nir1_rectangle_averages, nir1_all_coordinates_pixel_values, nir1_all_coordinates_pixel_reflectances, nir1_all_coordinates_signal_to_noise_ratios, nir1_x_labels, nir1_num_coordinates = (None, None, None, None, None, None)
		if nir2_actual_data_directory:
			# print(f'processing directory {nir2_actual_data_directory}')
			directory_name = get_title(nir2_actual_data_directory, format=3)
			single_directory_output_directory = os.path.join(output_directory, directory_name)
			nir2_rectangle_averages, nir2_all_coordinates_pixel_values, nir2_all_coordinates_pixel_reflectances, nir2_all_coordinates_signal_to_noise_ratios, nir2_x_labels, nir2_num_coordinates, min_y_pixel_value, max_y_pixel_value, min_y_reflectance, max_y_reflectance, min_y_signal_to_noise_ratio, max_y_signal_to_noise_ratio = process_ESA_meteorite_directory(
					nir2_metadata_directory,
					nir2_actual_data_directory,
					nir2_matching_dark_directory,
					single_directory_output_directory,
					nir2_pixel_coordinate,
					nir2_rectangle_coordinate,
					nir2_averaged_spectra_square_coordinate,
					nir2_calibration_data_directory,
					save_plot=True,
					test_mode=False
				)
			if min_y_reflectance < min_y_reflectance_of_all:
				min_y_reflectance_of_all = min_y_reflectance
			if max_y_reflectance > max_y_reflectance_of_all:
				max_y_reflectance_of_all = max_y_reflectance
		else:
			nir2_rectangle_averages, nir2_all_coordinates_pixel_values, nir2_all_coordinates_pixel_reflectances, nir2_all_coordinates_signal_to_noise_ratios, nir2_x_labels, nir2_num_coordinates = (None, None, None, None, None, None)

		# if vis_x_labels:
		# 	num_x_ticks = len(vis_x_labels)
		# elif nir1_x_labels:
		# 	num_x_ticks = len(nir1_x_labels)
		# elif nir2_x_labels:
		# 	num_x_ticks = len(nir2_x_labels)
		# else:
		# 	num_x_ticks = None


		if vis_pixel_coordinate:
			num_coordinates = len(vis_pixel_coordinate)
		elif nir1_pixel_coordinate:
			num_coordinates = len(nir1_pixel_coordinate)
		elif nir2_pixel_coordinate:
			num_coordinates = len(nir2_pixel_coordinate)
		else:
			num_coordinates = 1

		num_vis_x_ticks = 10
		num_nir1_x_ticks = 14
		num_nir2_x_ticks = 14
		
		combined_all_coordinates_values = [] # [coordinate[spectra_with_all_channels]]]
		for coordinate_index in range(num_coordinates):
			coordinate_spectra = []
			if vis_all_coordinates_pixel_reflectances:
				coordinate_spectra = coordinate_spectra + vis_all_coordinates_pixel_reflectances[coordinate_index]
				num_vis_x_ticks = len(vis_all_coordinates_pixel_reflectances[coordinate_index])
			else:
				coordinate_spectra = coordinate_spectra + [0]*num_vis_x_ticks
			if nir1_all_coordinates_pixel_reflectances:
				coordinate_spectra = coordinate_spectra + nir1_all_coordinates_pixel_reflectances[coordinate_index]
				num_nir1_x_ticks = len(nir1_all_coordinates_pixel_reflectances[coordinate_index])
			else:
				coordinate_spectra = coordinate_spectra + [0]*num_nir1_x_ticks
			if nir2_all_coordinates_pixel_reflectances:
				coordinate_spectra = coordinate_spectra + nir2_all_coordinates_pixel_reflectances[coordinate_index]
				num_nir2_x_ticks = len(nir2_all_coordinates_pixel_reflectances[coordinate_index])
			else:
				coordinate_spectra = coordinate_spectra + [0]*num_nir2_x_ticks
			combined_all_coordinates_values.append(coordinate_spectra)

		combined_x_labels = []
		if vis_x_labels:
			combined_x_labels = combined_x_labels + vis_x_labels
		else:
			combined_x_labels = combined_x_labels + ['']*num_vis_x_ticks
		if nir1_x_labels:
			combined_x_labels = combined_x_labels + nir1_x_labels
		else:
			combined_x_labels = combined_x_labels + ['']*num_nir1_x_ticks
		if nir2_x_labels:
			combined_x_labels = combined_x_labels + nir2_x_labels
		else:
			combined_x_labels = combined_x_labels + ['']*num_nir2_x_ticks

		title = get_title(output_directory, format=3)

		plot_combined_spectra(
			f'{title}_reflectances',
			'',
			combined_x_labels,
			combined_all_coordinates_values,
			output_directory,
			min_y_reflectance_of_all-min_y_reflectance_of_all*0.05,
			max_y_reflectance_of_all+max_y_reflectance_of_all*0.05,
			test_mode=test_mode,
		)

		combined_all_coordinates_values, combined_x_labels = spectra_post_calibration(combined_all_coordinates_values, combined_x_labels, nir_spectra_shifting, num_vis_x_ticks, num_nir1_x_ticks, num_nir2_x_ticks, vis_offset)

		plot_combined_spectra(
			f'{title}_smoothed_reflectances',
			'',
			combined_x_labels,
			combined_all_coordinates_values,
			output_directory,
			min_y_reflectance_of_all-min_y_reflectance_of_all*0.05,
			max_y_reflectance_of_all+max_y_reflectance_of_all*0.05,
			test_mode=test_mode,
		)

# ESA meteorite processing config begin from here ===============================================================================================================================
'''
This configuration section describes for example all the file paths,
pixel coordinates, and setting we want to use for processing.
For some cases there might still be some hardcoded paths or settings
in the functions themselves, but I try to mention them in the documentation,
at the beginning of each function.
'''

# timestamp for naming purposes
timestamp = time.time()
local_time = time.localtime(timestamp)
formatted_time = time.strftime("%Y-%m-%dT%H:%M:%S", local_time)

# These options specify which configuration lists to use for processing
'''
Back when I wrote this code, it was better to make new lists for new configurations
because I didn't want to lose the previous configuration I made.
You can specify which lists/cnfiguration you want to you by selecting writing
True or False to these variables. Make sure to only select one.

You can make your own set of lists for new data.
'''
averaged_spectra = True
uniform_spots = False

'''
This is the beginning of configuration lists for data processing.
The configuration lists are fed to the processing function, and the
function matches the configuration (lists) based in index.

Here is an example:
config_list_1 = [A, B, C]
config_list_2 = [1, 2, 3]
The processing function will process A with 1, B with 2, and C with 3.
'''

# Define the metadata directories for each acquisition
vis_metadata_directories = [
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-41-10_vis_20_vis_2800/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-41-10_vis_20_vis_2800/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-30_15-00-42_vis_Iran0006_vis_1400/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-30_15-11-36_vis_ElHammami_vis_1400/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-30_15-17-13_vis_Allende_vis_2800/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-22-37_vis_SaU001_vis_2800/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-27-59_vis_SaU001_vis_1400/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-41-10_vis_20_vis_2800/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-47-39_vis_20_vis_1400/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/meta', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-00-42_vis_Iran0006_vis_1400/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-11-36_vis_ElHammami_vis_1400/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-17-13_vis_Allende_vis_2800/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-22-37_vis_SaU001_vis_2800/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-27-59_vis_SaU001_vis_1400/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/meta',
]
if averaged_spectra:
	vis_metadata_directories = [
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/meta', # Averaged spectra (copied from above)
		'/home/sysa/HERA/ESA meteorites/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_15-00-42_vis_Iran0006_vis_1400/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_15-11-36_vis_ElHammami_vis_1400/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_15-17-13_vis_Allende_vis_2800/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-22-37_vis_SaU001_vis_2800/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-27-59_vis_SaU001_vis_1400/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/meta',
	]
if uniform_spots:
	vis_metadata_directories = [
		'/home/sysa/HERA/ESA meteorites/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_15-00-42_vis_Iran0006_vis_1400/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_15-11-36_vis_ElHammami_vis_1400/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_15-17-13_vis_Allende_vis_2800/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_15-22-37_vis_SaU001_vis_2800/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_15-27-59_vis_SaU001_vis_1400/meta',
	]
nir1_metadata_directories = [
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_11-51-32_nir1_20_nir1_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_11-51-32_nir1_20_nir1_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-30_12-31-12_nir1_Allende_nir1_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_11-51-32_nir1_20_nir1_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-43-27_nir1_20_nir1_5000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/meta', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-31-12_nir1_Allende_nir1_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/meta',
]
if averaged_spectra:
	nir1_metadata_directories = [
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/meta', # Averaged spectra (copied from above)
		'/home/sysa/HERA/ESA meteorites/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_12-31-12_nir1_Allende_nir1_15000/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/meta',
	]
if uniform_spots:
	nir1_metadata_directories = [
		'/home/sysa/HERA/ESA meteorites/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_12-31-12_nir1_Allende_nir1_15000/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/meta',
	]
nir2_metadata_directories = [
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_13-32-44_nir2_20_alb_target_tint25000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_14-36-57_nir2_20_nir2_25000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_14-43-36_nir2_Kilabo_nir2_25000/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-29_15-03-03_nir2_Iran0006_nir2_15000/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000/meta',
	'/home/sysa/HERA/ESA meteorites/2024-08-29_15-21-37_nir2_Allende_nir2_25000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-33-06_nir2_20_nir2_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-35-50_nir2_20_nir2_10000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_14-43-36_nir2_Kilabo_nir2_25000/meta', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-03-03_nir2_Iran0006_nir2_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-21-37_nir2_Allende_nir2_25000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000/meta',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000/meta',
]
if averaged_spectra:
	nir2_metadata_directories = [
		# '/home/sysa/HERA/ESA meteorites/2024-08-29_14-43-36_nir2_Kilabo_nir2_25000/meta', # Averaged spectra (copied from above)
		'/home/sysa/HERA/ESA meteorites/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-29_15-03-03_nir2_Iran0006_nir2_15000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-29_15-21-37_nir2_Allende_nir2_25000/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000/meta',
	]
if uniform_spots:
	nir2_metadata_directories = [
		'/home/sysa/HERA/ESA meteorites/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-29_15-03-03_nir2_Iran0006_nir2_15000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-29_15-21-37_nir2_Allende_nir2_25000/meta',
		# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/meta',
		'/home/sysa/HERA/ESA meteorites/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/meta',
	]
vis_actual_data_directories = [
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-00-42_vis_Iran0006_vis_1400',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-11-36_vis_ElHammami_vis_1400',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-17-13_vis_Allende_vis_2800',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-22-37_vis_SaU001_vis_2800',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-27-59_vis_SaU001_vis_1400',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-06-16_vis_ElHammami_vis_1400',#Duplicate?
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-00-42_vis_Iran0006_vis_1400',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-11-36_vis_ElHammami_vis_1400',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-17-13_vis_Allende_vis_2800',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-22-37_vis_SaU001_vis_2800',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-27-59_vis_SaU001_vis_1400',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400',
]
if averaged_spectra:
	vis_actual_data_directories = [
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400', # Averaged spectra (copied from above)
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-00-42_vis_Iran0006_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-11-36_vis_ElHammami_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-17-13_vis_Allende_vis_2800',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-22-37_vis_SaU001_vis_2800',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-27-59_vis_SaU001_vis_1400',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400',
	]
if uniform_spots:
	vis_actual_data_directories = [
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-00-42_vis_Iran0006_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-11-36_vis_ElHammami_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-17-13_vis_Allende_vis_2800',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-22-37_vis_SaU001_vis_2800',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-27-59_vis_SaU001_vis_1400',
	]
nir1_actual_data_directories = [
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-31-12_nir1_Allende_nir1_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-21-27_nir1_Iran0006_nir1_5000',#Duplicate?
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-18-38_nir1_Iran0006_nir1_5000',#Duplicate?
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-34-28_nir1_SaU001_nir1_15000',#Duplicate?
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-31-12_nir1_Allende_nir1_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000',
]
if averaged_spectra:
	nir1_actual_data_directories = [
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000', # Averaged spectra (copied from above)
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-31-12_nir1_Allende_nir1_15000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000', # Somewhat saturated
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000', # Somewhat saturated
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000',
	]
if uniform_spots:
	nir1_actual_data_directories = [
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-31-12_nir1_Allende_nir1_15000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000',
	]
nir2_actual_data_directories = [
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_13-32-44_nir2_20_alb_target_tint25000',# #2024-08-29_13-32-44_nir2_20_alb_target_tint25000 has weird setpoints and not matching with anything
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-43-36_nir2_Kilabo_nir2_25000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-03-03_nir2_Iran0006_nir2_15000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-21-37_nir2_Allende_nir2_25000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-35-50_nir2_20_nir2_10000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-07-52_nir2_Iran0006_nir2_15000',#Duplicate?
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-43-36_nir2_Kilabo_nir2_25000', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-03-03_nir2_Iran0006_nir2_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-21-37_nir2_Allende_nir2_25000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000',
]
if averaged_spectra:
	nir2_actual_data_directories = [
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-43-36_nir2_Kilabo_nir2_25000', # Saturated # Averaged spectra (copied from above)
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-03-03_nir2_Iran0006_nir2_15000', # Out of focus
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000', # Out of focus
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-21-37_nir2_Allende_nir2_25000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000', # Saturated
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000', # Saturated
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000', # Somewhat saturated
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000', # Somewhat saturated
	]
if uniform_spots:
	nir2_actual_data_directories = [
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-03-03_nir2_Iran0006_nir2_15000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-21-37_nir2_Allende_nir2_25000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000',
	]
vis_pixel_location_bins = [
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-00-42_vis_Iran0006_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-11-36_vis_ElHammami_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-17-13_vis_Allende_vis_2800/dc_0_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-22-37_vis_SaU001_vis_2800/dc_0_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-27-59_vis_SaU001_vis_1400/dc_0_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-06-16_vis_ElHammami_vis_1400/dc_0_exp_009.bin',#Duplicate?
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_009.bin', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-00-42_vis_Iran0006_vis_1400/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-11-36_vis_ElHammami_vis_1400/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-17-13_vis_Allende_vis_2800/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-22-37_vis_SaU001_vis_2800/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-27-59_vis_SaU001_vis_1400/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/dc_0_exp_009.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/dc_0_exp_009.bin',
]
if averaged_spectra:
	vis_pixel_location_bins = [
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_005.bin', # Averaged spectra (copied from above)
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-00-42_vis_Iran0006_vis_1400/dc_0_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-11-36_vis_ElHammami_vis_1400/dc_0_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-17-13_vis_Allende_vis_2800/dc_0_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-22-37_vis_SaU001_vis_2800/dc_0_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-27-59_vis_SaU001_vis_1400/dc_0_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/dc_0_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/dc_0_exp_005.bin',
	]
if uniform_spots:
	vis_pixel_location_bins = [
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-00-42_vis_Iran0006_vis_1400/dc_0_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-11-36_vis_ElHammami_vis_1400/dc_0_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-17-13_vis_Allende_vis_2800/dc_0_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-22-37_vis_SaU001_vis_2800/dc_0_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-27-59_vis_SaU001_vis_1400/dc_0_exp_005.bin',
	]
nir1_pixel_location_bins = [
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-31-12_nir1_Allende_nir1_15000/dc_1_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/dc_1_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/dc_1_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-18-38_nir1_Iran0006_nir1_5000/dc_1_exp_013.bin',#Duplicate?
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-21-27_nir1_Iran0006_nir1_5000/dc_1_exp_013.bin',#Duplicate?
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-34-28_nir1_SaU001_nir1_15000/dc_1_exp_013.bin',#Duplicate?
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_013.bin', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-31-12_nir1_Allende_nir1_15000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/dc_1_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/dc_1_exp_013.bin',
]
if averaged_spectra:
	nir1_pixel_location_bins = [
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_005.bin', # Averaged spectra (copied from above)
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000/dc_1_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000/dc_1_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-31-12_nir1_Allende_nir1_15000/dc_1_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/dc_1_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/dc_1_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/dc_1_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/dc_1_exp_005.bin',
	]
if uniform_spots:
	nir1_pixel_location_bins = [
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000/dc_1_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000/dc_1_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-31-12_nir1_Allende_nir1_15000/dc_1_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/dc_1_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/dc_1_exp_005.bin',
	]
nir2_pixel_location_bins = [
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_13-32-44_nir2_20_alb_target_tint25000/dc_2_exp_000.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-43-36_nir2_Kilabo_nir2_25000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-07-52_nir2_Iran0006_nir2_15000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-21-37_nir2_Allende_nir2_25000/dc_2_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/dc_2_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/dc_2_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-35-50_nir2_20_nir2_10000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-03-03_nir2_Iran0006_nir2_15000/dc_2_exp_013.bin',#Duplicate?
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-43-36_nir2_Kilabo_nir2_25000/dc_2_exp_013.bin', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-07-52_nir2_Iran0006_nir2_15000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-21-37_nir2_Allende_nir2_25000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000/dc_2_exp_013.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000/dc_2_exp_013.bin',
]
if averaged_spectra:
	nir2_pixel_location_bins = [
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-43-36_nir2_Kilabo_nir2_25000/dc_2_exp_005.bin', # Averaged spectra (copied from above)
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000/dc_2_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000/dc_2_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-07-52_nir2_Iran0006_nir2_15000/dc_2_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000/dc_2_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-21-37_nir2_Allende_nir2_25000/dc_2_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/dc_2_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/dc_2_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000/dc_2_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000/dc_2_exp_005.bin',
	]
if uniform_spots:
	nir2_pixel_location_bins = [
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000/dc_2_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000/dc_2_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-07-52_nir2_Iran0006_nir2_15000/dc_2_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000/dc_2_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-21-37_nir2_Allende_nir2_25000/dc_2_exp_005.bin',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/dc_2_exp_005.bin',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/dc_2_exp_005.bin',
	]
vis_matching_dark_directories = [
	# '2024-08-30_15-51-17_vis_vis_dark_2800',
	# '2024-08-30_15-51-17_vis_vis_dark_2800',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-51-17_vis_vis_dark_2800',
	# '2024-08-30_15-51-17_vis_vis_dark_2800',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	# '2024-08-30_15-51-17_vis_vis_dark_2800',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	# '2024-08-30_15-54-10_vis_vis_dark_1400', # Averaged spectra (copied from above)
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	# '2024-08-30_15-51-17_vis_vis_dark_2800',
	# '2024-08-30_15-51-17_vis_vis_dark_2800',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
]
if averaged_spectra:
	vis_matching_dark_directories = [
		# '2024-08-30_15-54-10_vis_vis_dark_1400', # Averaged spectra (copied from above)
		'2024-08-30_15-54-10_vis_vis_dark_1400',
		'2024-08-30_15-54-10_vis_vis_dark_1400',
		'2024-08-30_15-54-10_vis_vis_dark_1400',
		'2024-08-30_15-54-10_vis_vis_dark_1400',
		'2024-08-30_15-51-17_vis_vis_dark_2800',
		# '2024-08-30_15-51-17_vis_vis_dark_2800',
		# '2024-08-30_15-54-10_vis_vis_dark_1400',
		# '2024-08-30_15-54-10_vis_vis_dark_1400',
		'2024-08-30_15-54-10_vis_vis_dark_1400',
	]
if uniform_spots:
	vis_matching_dark_directories = [
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	# '2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-51-17_vis_vis_dark_2800',
	# '2024-08-30_15-51-17_vis_vis_dark_2800',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	]
nir1_matching_dark_directories = [
	# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	'2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000', # Averaged spectra (copied from above)
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
]
if averaged_spectra:
	nir1_matching_dark_directories = [
		# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000', # Averaged spectra (copied from above)
		'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
		'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
		'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
		'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
		'2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
		# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
		# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
		# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
		'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	]
if uniform_spots:
	nir1_matching_dark_directories = [
		'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
		# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
		'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
		'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
		'2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
		# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
		'2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	]
nir2_matching_dark_directories = [
	# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
	# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
	# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
	'2024-08-29_15-57-12_nir2_dark_nir2_15000',
	'2024-08-29_15-55-34_nir2_dark_nir2_10000',
	'2024-08-29_15-57-12_nir2_dark_nir2_15000',
	'2024-08-29_15-57-12_nir2_dark_nir2_15000',
	'2024-08-29_15-58-38_nir2_dark_nir2_25000',
	# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
	# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
	# '2024-08-29_15-57-12_nir2_dark_nir2_15000',
	# '2024-08-29_15-55-34_nir2_dark_nir2_10000',
	# '2024-08-29_15-57-12_nir2_dark_nir2_15000',
	# '2024-08-29_15-55-34_nir2_dark_nir2_10000',
	# '2024-08-29_15-58-38_nir2_dark_nir2_25000', # Averaged spectra (copied from above)
	# '2024-08-29_15-57-12_nir2_dark_nir2_15000',
	# '2024-08-29_15-55-34_nir2_dark_nir2_10000',
	# '2024-08-29_15-57-12_nir2_dark_nir2_15000',
	# '2024-08-29_15-57-12_nir2_dark_nir2_15000',
	# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
	# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
	# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
	# '2024-08-29_15-57-12_nir2_dark_nir2_15000',
	# '2024-08-29_15-55-34_nir2_dark_nir2_10000',
]
if averaged_spectra:
	nir2_matching_dark_directories = [
		# '2024-08-29_15-58-38_nir2_dark_nir2_25000', # Averaged spectra (copied from above)
		'2024-08-29_15-57-12_nir2_dark_nir2_15000',
		'2024-08-29_15-55-34_nir2_dark_nir2_10000',
		'2024-08-29_15-57-12_nir2_dark_nir2_15000',
		'2024-08-29_15-57-12_nir2_dark_nir2_15000',
		'2024-08-29_15-58-38_nir2_dark_nir2_25000',
		# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
		# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
		# '2024-08-29_15-57-12_nir2_dark_nir2_15000',
		'2024-08-29_15-55-34_nir2_dark_nir2_10000',
	]
if uniform_spots:
	nir2_matching_dark_directories = [
		'2024-08-29_15-57-12_nir2_dark_nir2_15000',
		# '2024-08-29_15-55-34_nir2_dark_nir2_10000',
		'2024-08-29_15-57-12_nir2_dark_nir2_15000',
		'2024-08-29_15-57-12_nir2_dark_nir2_15000',
		'2024-08-29_15-58-38_nir2_dark_nir2_25000',
		# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
		'2024-08-29_15-58-38_nir2_dark_nir2_25000',
	]
output_directories = [
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/20_vis_2800_nir1_15000_nir2_tint_25000',#2024-08-29_13-32-44_nir2_20_alb_target_tint25000 has weird setpoints
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/20_vis_2800_nir1_15000_nir2_25000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Kilabo_vis_1400_nir1_5000_nir2_25000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Kilabo_vis_1400_nir1_5000_nir2_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Kilabo_vis_1400_nir1_5000_nir2_10000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Iran0006_vis_1400_nir1_5000_nir2_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/ElHammami_vis_1400_nir1_5000_nir2_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Allende_vis_2800_nir1_15000_nir2_25000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/SaU001_vis_2800_nir1_15000_nir2_25000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/SaU001_vis_1400_nir1_15000_nir2_25000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/20_vis_2800_nir1_15000_nir2_15000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/20_vis_1400_nir1_5000_nir2_10000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/ElHammamiB_vis_1400_nir1_5000_nir2_15000', # Bad images
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/ElHammamiB_vis_1400_nir1_5000_nir2_10000', # Bad images
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Kilabo_vis_1400_nir1_5000_nir2_25000', # Averaged spectra (copied from above)
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Kilabo_vis_1400_nir1_5000_nir2_15000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Kilabo_vis_1400_nir1_5000_nir2_10000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Iran0006_vis_1400_nir1_5000_nir2_15000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/ElHammami_vis_1400_nir1_5000_nir2_15000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Allende_vis_2800_nir1_15000_nir2_25000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/SaU001_vis_2800_nir1_15000_nir2_25000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/SaU001_vis_1400_nir1_15000_nir2_25000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/ElHammamiB_vis_1400_nir1_5000_nir2_15000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/ElHammamiB_vis_1400_nir1_5000_nir2_10000',
]
if averaged_spectra:
	output_directories = [
		# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Kilabo_vis_1400_nir1_5000_nir2_25000', # Averaged spectra (copied from above)
		f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Kilabo_vis_1400_nir1_5000_nir2_15000',
		f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Kilabo_vis_1400_nir1_5000_nir2_10000',
		f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Iran0006_vis_1400_nir1_5000_nir2_15000',
		f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/ElHammami_vis_1400_nir1_5000_nir2_15000',
		f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Allende_vis_2800_nir1_15000_nir2_25000',
		# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/SaU001_vis_2800_nir1_15000_nir2_25000',
		# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/SaU001_vis_1400_nir1_15000_nir2_25000',
		# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/ElHammamiB_vis_1400_nir1_5000_nir2_15000',
		f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/ElHammamiB_vis_1400_nir1_5000_nir2_10000',
	]
if uniform_spots:
	output_directories = [
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_reflectances/Kilabo_vis_1400_nir1_5000_nir2_15000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/Kilabo_vis_1400_nir1_5000_nir2_10000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_reflectances/Iran0006_vis_1400_nir1_5000_nir2_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_reflectances/ElHammami_vis_1400_nir1_5000_nir2_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_reflectances/Allende_vis_2800_nir1_15000_nir2_25000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_comparison/SaU001_vis_2800_nir1_15000_nir2_25000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_reflectances/SaU001_vis_1400_nir1_15000_nir2_25000',
	]
vis_pixel_coordinates = []
nir1_pixel_coordinates = []
nir2_pixel_coordinates = []
vis_rectangle_coordinates = [ # Calibration squares are taken from ESA 20 images
	# ((620,11), (625,11), (625,16), (620,16)),#20_vis_2800
	# ((620,11), (625,11), (625,16), (620,16)),#20_vis_2800
	# ((250,200), (700,200), (700,800), (250,800)),#Kilaboo_vis_1400
	((250,200), (700,200), (700,800), (250,800)),#Kilaboo_vis_1400
	((250,200), (700,200), (700,800), (250,800)),#Kilaboo_vis_1400
	((250,200), (700,200), (700,800), (250,800)),#Iran0006_vis_1400
	((250,200), (700,200), (700,800), (250,800)),#15-11-36_ElHammami_vis_1400
	((250,200), (700,200), (700,800), (250,800)),#Allende_vis_2800
	# ((250,200), (700,200), (700,800), (250,800)),#SaU001_vis_2800
	# ((250,200), (700,200), (700,800), (250,800)),#SaU001_vis_1400
	# ((620,11), (625,11), (625,16), (620,16)),#20_vis_2800
	# ((620,11), (625,11), (625,16), (620,16)),#20_vis_1400
	# ((250,200), (700,200), (700,800), (250,800)),#ElHammamiB_vis_1400
	# ((250,200), (700,200), (700,800), (250,800)),#ElHammamiB_vis_1400
	# ((980,30), (985,30), (985,35), (980,35)),#15-06-16_ElHammami_vis_1400#Duplicate?
	# ((250,200), (700,200), (700,800), (250,800)),#Kilaboo_vis_1400 # Averaged spectra (copied from above)
	# ((250,200), (700,200), (700,800), (250,800)),#Kilaboo_vis_1400
	# ((250,200), (700,200), (700,800), (250,800)),#Kilaboo_vis_1400
	# ((250,200), (700,200), (700,800), (250,800)),#Iran0006_vis_1400
	# ((250,200), (700,200), (700,800), (250,800)),#15-11-36_ElHammami_vis_1400
	# ((250,200), (700,200), (700,800), (250,800)),#Allende_vis_2800
	# ((250,200), (700,200), (700,800), (250,800)),#SaU001_vis_2800
	# ((250,200), (700,200), (700,800), (250,800)),#SaU001_vis_1400
	# ((250,200), (700,200), (700,800), (250,800)),#ElHammamiB_vis_1400
	# ((250,200), (700,200), (700,800), (250,800)),#ElHammamiB_vis_1400
]
if averaged_spectra:
	vis_rectangle_coordinates = [
		# ((250,200), (700,200), (700,800), (250,800)),#Kilaboo_vis_1400 # Averaged spectra (copied from above) # Old calibration squares
		# ((250,200), (700,200), (700,800), (250,800)),#Kilaboo_vis_1400
		# ((250,200), (700,200), (700,800), (250,800)),#Kilaboo_vis_1400
		# ((250,200), (700,200), (700,800), (250,800)),#Iran0006_vis_1400
		# ((250,200), (700,200), (700,800), (250,800)),#15-11-36_ElHammami_vis_1400
		# ((250,200), (700,200), (700,800), (250,800)),#Allende_vis_2800
		# ((250,200), (700,200), (700,800), (250,800)),#SaU001_vis_2800
		# ((250,200), (700,200), (700,800), (250,800)),#SaU001_vis_1400
		# ((250,200), (700,200), (700,800), (250,800)),#ElHammamiB_vis_1400
		# ((250,200), (700,200), (700,800), (250,800)),#ElHammamiB_vis_1400

		# ((359,210), (771,210), (771,628), (359,628)),#Kilaboo_vis_1400 # Averaged spectra # Identical calibration squares to the averaged spectra squares
		# ((359,210), (771,210), (771,628), (359,628)),#Kilaboo_vis_1400
		# ((359,210), (771,210), (771,628), (359,628)),#Kilaboo_vis_1400
		# ((377,126), (724,126), (724,364), (377,364)),#Iran0006_vis_1400
		# ((203,347), (568,347), (568,603), (203,603)),#15-11-36_ElHammami_vis_1400
		# ((306,292), (619,292), (619,591), (306,591)),#Allende_vis_2800
		# ((134,370), (726,370), (726,849), (134,849)),#SaU001_vis_2800
		# ((171,398), (800,398), (800,886), (171,886)),#SaU001_vis_1400
		# ((238,384), (822,384), (822,974), (238,974)),#ElHammamiB_vis_1400
		# ((238,384), (822,384), (822,974), (238,974)),#ElHammamiB_vis_1400

		# ((423,463), (787,463), (787,597), (423,597)),#Kilaboo_vis_1400 # Averaged spectra # specific area where possible
		((423,463), (787,463), (787,597), (423,597)),#Kilaboo_vis_1400
		((420,315), (787,315), (787,603), (420,603)),#Kilaboo_vis_1400
		((400,250), (600,250), (600,400), (400,400)),#Iran0006_vis_1400
		((183,347), (538,347), (538,603), (183,603)),#15-11-36_ElHammami_vis_1400
		((388,509), (638,509), (638,571), (388,571)),#Allende_vis_2800
		# ((134,370), (726,370), (726,849), (134,849)),#SaU001_vis_2800
		# ((171,398), (800,398), (800,886), (171,886)),#SaU001_vis_1400
		# ((263,513), (566,513), (566,786), (263,786)),#ElHammamiB_vis_1400
		((263,513), (566,513), (566,786), (263,786)),#ElHammamiB_vis_1400
	]
if uniform_spots:
	vis_rectangle_coordinates = [
		# ((250,200), (700,200), (700,800), (250,800)),#Kilaboo_vis_1400
		# # ((250,200), (700,200), (700,800), (250,800)),#Kilaboo_vis_1400
		# ((429,179), (715,179), (715,372), (429,372)),#Iran0006_vis_1400
		# ((306,366), (570,366), (570,583), (306,583)),#15-11-36_ElHammami_vis_1400
		# ((480,487), (660,487), (660,605), (480,605)),#Allende_vis_2800
		# # ((250,200), (700,200), (700,800), (250,800)),#SaU001_vis_2800
		# ((250,200), (700,200), (700,800), (250,800)),#SaU001_vis_1400

		None,
		# None,
		None,
		None,
		None,
		# None,
		None,
	]
nir1_rectangle_coordinates = [
	# ((625,5), (630,5), (630,10), (625,10)),#20_nir1_15000
	# ((625,5), (630,5), (630,10), (625,10)),#20_nir1_15000
	# ((40,40), (600,40), (600,475), (40,475)),#Kilaboo_nir1_5000
	((40,40), (600,40), (600,475), (40,475)),#Kilaboo_nir1_5000
	((40,40), (600,40), (600,475), (40,475)),#Kilaboo_nir1_5000
	((40,40), (600,40), (600,475), (40,475)),#12-24-32_nir1_Iran0006_nir1_5000
	((40,40), (600,40), (600,475), (40,475)),#ElHammami_nir1_5000
	((40,40), (600,40), (600,475), (40,475)),#Allende_nir1_15000
	# ((40,40), (600,40), (600,475), (40,475)),#12-36-59_nir1_SaU001_nir1_15000
	# ((40,40), (600,40), (600,475), (40,475)),#12-36-59_nir1_SaU001_nir1_15000
	# ((625,5), (630,5), (630,10), (625,10)),#20_nir1_15000
	# ((1,1), (6,1), (6,6), (1,6)),#20_nir1_5000
	# ((40,40), (600,40), (600,475), (40,475)),#ElHammamiB_nir1_5000
	# ((40,40), (600,40), (600,475), (40,475)),#ElHammamiB_nir1_5000
	# ((590,305), (595,305), (595,310), (590,310)),#12-18-38_nir1_Iran0006_nir1_5000#Duplicate?
	# ((590,305), (595,305), (595,310), (590,310)),#12-21-27_nir1_Iran0006_nir1_5000#Duplicate?
	# ((591,19), (596,19), (596,24), (591,24)),#12-34-28_nir1_SaU001_nir1_15000#Duplicate?
	# ((40,40), (600,40), (600,475), (40,475)),#Kilaboo_nir1_5000 # Averaged spectra (copied from above)
	# ((40,40), (600,40), (600,475), (40,475)),#Kilaboo_nir1_5000
	# ((40,40), (600,40), (600,475), (40,475)),#Kilaboo_nir1_5000
	# ((40,40), (600,40), (600,475), (40,475)),#12-24-32_nir1_Iran0006_nir1_5000
	# ((40,40), (600,40), (600,475), (40,475)),#ElHammami_nir1_5000
	# ((40,40), (600,40), (600,475), (40,475)),#Allende_nir1_15000
	# ((40,40), (600,40), (600,475), (40,475)),#12-36-59_nir1_SaU001_nir1_15000
	# ((40,40), (600,40), (600,475), (40,475)),#12-36-59_nir1_SaU001_nir1_15000
	# ((40,40), (600,40), (600,475), (40,475)),#ElHammamiB_nir1_5000
	# ((40,40), (600,40), (600,475), (40,475)),#ElHammamiB_nir1_5000
]
if averaged_spectra:
	nir1_rectangle_coordinates = [
		# ((40,40), (600,40), (600,475), (40,475)),#Kilaboo_nir1_5000 # Averaged spectra (copied from above)
		# ((40,40), (600,40), (600,475), (40,475)),#Kilaboo_nir1_5000
		# ((40,40), (600,40), (600,475), (40,475)),#Kilaboo_nir1_5000
		# ((40,40), (600,40), (600,475), (40,475)),#12-24-32_nir1_Iran0006_nir1_5000
		# ((40,40), (600,40), (600,475), (40,475)),#ElHammami_nir1_5000
		# ((40,40), (600,40), (600,475), (40,475)),#Allende_nir1_15000
		# ((40,40), (600,40), (600,475), (40,475)),#12-36-59_nir1_SaU001_nir1_15000
		# ((40,40), (600,40), (600,475), (40,475)),#12-36-59_nir1_SaU001_nir1_15000
		# ((40,40), (600,40), (600,475), (40,475)),#ElHammamiB_nir1_5000
		# ((40,40), (600,40), (600,475), (40,475)),#ElHammamiB_nir1_5000

		# ((210,40), (586,40), (586,486), (210,486)),#Kilaboo_nir1_5000 # Averaged spectra
		# ((210,40), (586,40), (586,486), (210,486)),#Kilaboo_nir1_5000
		# ((210,40), (586,40), (586,486), (210,486)),#Kilaboo_nir1_5000
		# ((58,124), (433,124), (433,310), (58,310)),#12-24-32_nir1_Iran0006_nir1_5000
		# ((205,102), (565,102), (565,328), (205,328)),#ElHammami_nir1_5000
		# ((55,92), (455,92), (455,342), (55,342)),#Allende_nir1_15000
		# ((21,27), (603,27), (603,474), (21,474)),#12-36-59_nir1_SaU001_nir1_15000
		# ((21,27), (603,27), (603,474), (21,474)),#12-36-59_nir1_SaU001_nir1_15000
		# ((78,44), (585,44), (585,462), (78,462)),#ElHammamiB_nir1_5000
		# ((78,44), (585,44), (585,462), (78,462)),#ElHammamiB_nir1_5000

		# ((531,348), (193,348), (193,459), (531,459)),#Kilaboo_nir1_5000 # Averaged spectra
		((531,348), (193,348), (193,459), (531,459)),#Kilaboo_nir1_5000
		((533,458), (196,458), (196,193), (533,193)),#Kilaboo_nir1_5000
		((112,206), (346,206), (346,83), (112,83)),#12-24-32_nir1_Iran0006_nir1_5000
		((205,102), (565,102), (565,328), (205,328)),#ElHammami_nir1_5000
		((350,273), (125,273), (125,318), (350,318)),#Allende_nir1_15000
		# ((21,27), (603,27), (603,474), (21,474)),#12-36-59_nir1_SaU001_nir1_15000
		# ((21,27), (603,27), (603,474), (21,474)),#12-36-59_nir1_SaU001_nir1_15000
		# ((264,93), (517,93), (517,332), (264,332)),#ElHammamiB_nir1_5000
		((264,93), (517,93), (517,332), (264,332)),#ElHammamiB_nir1_5000
	]
if uniform_spots:
	nir1_rectangle_coordinates = [
		# ((40,40), (600,40), (600,475), (40,475)),#Kilaboo_nir1_5000
		# # ((40,40), (600,40), (600,475), (40,475)),#Kilaboo_nir1_5000
		# ((131,171), (361,171), (361,346), (131,346)),#12-24-32_nir1_Iran0006_nir1_5000
		# ((223,137), (447,137), (447,358), (223,358)),#ElHammami_nir1_5000
		# ((106,263), (264,263), (264,357), (106,357)),#Allende_nir1_15000
		# # ((40,40), (600,40), (600,475), (40,475)),#12-36-59_nir1_SaU001_nir1_15000
		# ((40,40), (600,40), (600,475), (40,475)),#12-36-59_nir1_SaU001_nir1_15000

		None,
		# None,
		None,
		None,
		None,
		# None,
		None,
	]
nir2_rectangle_coordinates = [
	# ((626,5), (631,5), (631,10), (626,10)),#20_alb_target_tint25000
	# ((626,5), (631,5), (631,10), (626,10)),#20_nir2_25000
	# ((40,40), (600,40), (600,475), (40,475)),#Kilabo_nir2_25000
	((40,40), (500,40), (500,250), (40,250)),#Kilabo_nir2_15000
	((40,40), (500,40), (500,250), (40,250)),#Kilabo_nir2_10000
	((40,40), (500,40), (500,250), (40,250)),#15-07-52_nir2_Iran0006_nir2_15000
	((40,40), (500,40), (500,250), (40,250)),#ElHammami_nir2_15000
	((40,40), (600,40), (600,475), (40,475)),#Allende_nir2_25000
	# ((40,40), (600,40), (600,475), (40,475)),#SaU001_nir2_25000
	# ((40,40), (600,40), (600,475), (40,475)),#SaU001_nir2_25000
	# ((550,355), (555,355), (555,360), (550,360)),#20_nir2_15000
	# ((550,355), (555,355), (555,360), (550,360)),#20_nir2_10000
	# ((40,40), (500,40), (500,250), (40,250)),#ElHammamiB_nir2_15000
	# ((40,40), (500,40), (500,250), (40,250)),#ElHammamiB_nir2_10000
	# ((580,400), (585,400), (585,405), (580,405)),#15-03-03_nir2_Iran0006_nir2_15000#Duplicate?
	# ((40,40), (600,40), (600,475), (40,475)),#Kilabo_nir2_25000 # Averaged spectra (copied from above)
	# ((40,40), (500,40), (500,250), (40,250)),#Kilabo_nir2_15000
	# ((40,40), (500,40), (500,250), (40,250)),#Kilabo_nir2_10000
	# ((40,40), (500,40), (500,250), (40,250)),#15-07-52_nir2_Iran0006_nir2_15000
	# ((40,40), (500,40), (500,250), (40,250)),#ElHammami_nir2_15000
	# ((40,40), (600,40), (600,475), (40,475)),#Allende_nir2_25000
	# ((40,40), (600,40), (600,475), (40,475)),#SaU001_nir2_25000
	# ((40,40), (600,40), (600,475), (40,475)),#SaU001_nir2_25000
	# ((40,40), (500,40), (500,250), (40,250)),#ElHammamiB_nir2_15000
	# ((40,40), (500,40), (500,250), (40,250)),#ElHammamiB_nir2_10000
]
if averaged_spectra:
	nir2_rectangle_coordinates = [
		# ((40,40), (600,40), (600,475), (40,475)),#Kilabo_nir2_25000 # Averaged spectra (copied from above)
		# ((40,40), (500,40), (500,250), (40,250)),#Kilabo_nir2_15000
		# ((40,40), (500,40), (500,250), (40,250)),#Kilabo_nir2_10000
		# ((40,40), (500,40), (500,250), (40,250)),#15-07-52_nir2_Iran0006_nir2_15000
		# ((40,40), (500,40), (500,250), (40,250)),#ElHammami_nir2_15000
		# ((40,40), (600,40), (600,475), (40,475)),#Allende_nir2_25000
		# ((40,40), (600,40), (600,475), (40,475)),#SaU001_nir2_25000
		# ((40,40), (600,40), (600,475), (40,475)),#SaU001_nir2_25000
		# ((40,40), (500,40), (500,250), (40,250)),#ElHammamiB_nir2_15000
		# ((40,40), (500,40), (500,250), (40,250)),#ElHammamiB_nir2_10000

		# ((92,190), (400,190), (400,473), (92,473)),#Kilabo_nir2_25000 # Averaged spectra
		# ((80,180), (420,180), (420,480), (80,480)),#Kilabo_nir2_15000
		# ((75,30), (475,30), (475,449), (75,449)),#Kilabo_nir2_10000
		# ((105,125), (420,125), (420,373), (105,373)),#15-07-52_nir2_Iran0006_nir2_15000
		# ((140,92), (500,92), (500,358), (140,358)),#ElHammami_nir2_15000
		# ((150,180), (505,180), (505,430), (150,430)),#Allende_nir2_25000
		# ((29,27), (570,27), (570,315), (29,315)),#SaU001_nir2_25000
		# ((29,27), (570,27), (570,315), (29,315)),#SaU001_nir2_25000
		# ((65,25), (260,25), (260,375), (65,375)),#ElHammamiB_nir2_15000
		# ((72,28), (266,28), (266,471), (72,471)),#ElHammamiB_nir2_10000

		# ((138,286), (466,286), (466,369), (138,369)),#Kilabo_nir2_25000 # Averaged spectra
		((138,286-100), (466,286-100), (466,369-100), (138,369-100)),#Kilabo_nir2_15000
		((135,62), (464,62), (464,300), (135,300)),#Kilabo_nir2_10000
		((437-100,238-100), (227-100,238-100), (227-100,396-100), (437-100,396-100)),#15-07-52_nir2_Iran0006_nir2_15000
		((140,92), (500,92), (500,200), (140,200)),#ElHammami_nir2_15000
		((271,190), (493,190), (493,142), (271,142)),#Allende_vis_2800
		# ((29,27), (570,27), (570,315), (29,315)),#SaU001_nir2_25000
		# ((29,27), (570,27), (570,315), (29,315)),#SaU001_nir2_25000
		# ((87,1), (335,1), (335,149), (87,149)),#ElHammamiB_nir2_15000
		((93,23), (344,23), (344,264), (93,264)),#ElHammamiB_nir2_10000
	]
if uniform_spots:
	nir2_rectangle_coordinates = [
		# ((40,40), (500,40), (500,250), (40,250)),#Kilabo_nir2_15000
		# # ((40,40), (500,40), (500,250), (40,250)),#Kilabo_nir2_10000
		# ((176,125), (412,125), (412,269), (176,269)),#15-07-52_nir2_Iran0006_nir2_15000
		# ((225,100), (500,100), (500,250), (225,250)),#ElHammami_nir2_15000
		# ((340,100), (516,100), (516,174), (340,174)),#Allende_nir2_25000
		# # ((40,40), (600,40), (600,475), (40,475)),#SaU001_nir2_25000
		# ((40,40), (600,40), (600,475), (40,475)),#SaU001_nir2_25000

		[(451,304), (374,332), (244,344), (118,288)],
		# None,
		None,
		None,
		None,
		# None,
		None,
	]
vis_coordinate_helper_lines = [
	# [((387,214),(779,734)), ((387,214),(459,923))],#20_vis_2800
	# [((387,214),(779,734)), ((387,214),(459,923))],#20_vis_2800
	# [((422,461),(828,599))],#Kilaboo_vis_1400
	[((422,461),(840,603))],#Kilaboo_vis_1400
	[((422,461),(840,603))],#Kilaboo_vis_1400
	[((697,198),(296,403)), ((697,198),(842,161))],#Iran0006_vis_1400
	[((560,570),(144,532)), ((560,570),(199,294))],#15-11-36_ElHammami_vis_1400
	[((642,503),(334,675)), ((642,503),(240,589))],#Allende_vis_2800
	# [((392,540),(285,837))],#SaU001_vis_2800
	# [((392,540),(285,837))],#SaU001_vis_1400
	# [((387,214),(779,734)), ((387,214),(459,923))],#20_vis_2800
	# [((387,214),(779,734)), ((387,214),(459,923))],#20_vis_1400
	# [((404,652),(638,321)), ((404,652),(801,355))],#ElHammamiB_vis_1400
	# [((404,652),(638,321)), ((404,652),(801,355))],#ElHammamiB_vis_1400
	# [((637,306),(338,333)), ((637,306),(349,257))],#15-06-16_ElHammami_vis_1400#Duplicate?
	# None, # Averaged spectra
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
]
if averaged_spectra:
	vis_coordinate_helper_lines = [
		# None, # Averaged spectra
		None,
		None,
		None,
		None,
		None,
		# None,
		# None,
		# None,
		None,
	]
if uniform_spots:
	vis_coordinate_helper_lines = [
		[((837,581),(422,461)), ((837,581),(394,540))],#Kilaboo_vis_1400
		# [((422,461),(840,603))],#Kilaboo_vis_1400
		[((694,194),(296,403)), ((694,194),(511,413))],#Iran0006_vis_1400
		[((557,567),(199,294))],#15-11-36_ElHammami_vis_1400
		[((648,498),(507,587)), ((648,498),(240,589))],#Allende_vis_2800
		# [((392,540),(285,837))],#SaU001_vis_2800
		[((550,550),(398,876))],#SaU001_vis_1400
	]
nir1_coordinate_helper_lines = [
	# [((100,100),(550,450))],#20_nir1_15000
	# [((100,100),(550,450))],#20_nir1_15000
	# [((533,345),(148,457))],#Kilaboo_nir1_5000
	[((533,345),(148,457))],#Kilaboo_nir1_5000
	[((533,345),(148,457))],#Kilaboo_nir1_5000
	[((145,177),(473,406)), ((145,177),(20,127))],#12-24-32_nir1_Iran0006_nir1_5000
	[((248,344),(628,278)), ((248,344),(561,66))],#ElHammami_nir1_5000
	[((122,273),(420,395)), ((122,273),(493,324))],#Allende_nir1_15000
	# [((211,336),(118,84))],#12-36-59_nir1_SaU001_nir1_15000
	# [((211,336),(118,84))],#12-36-59_nir1_SaU001_nir1_15000
	# [((100,100),(550,450))],#20_nir1_15000
	# [((387,214),(47,291)), ((387,214),(81,481))],#20_nir1_5000
	# [((387,214),(409,7)), ((387,214),(554,16))],#ElHammamiB_nir1_5000
	# [((387,214),(409,7)), ((387,214),(554,16))],#ElHammamiB_nir1_5000
	# [((529,170),(52,239)), ((529,170),(469,322))],#12-18-38_nir1_Iran0006_nir1_5000#Duplicate?
	# [((529,214),(56,283)), ((529,214),(470,367))],#12-21-27_nir1_Iran0006_nir1_5000#Duplicate?
	# [((211,451),(118,197))],#12-34-28_nir1_SaU001_nir1_15000#Duplicate?
	# None, # Averaged spectra
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
]
if averaged_spectra:
	nir1_coordinate_helper_lines = [
		# None, # Averaged spectra
		None,
		None,
		None,
		None,
		None,
		# None,
		# None,
		# None,
		None,
	]
if uniform_spots:
	nir1_coordinate_helper_lines = [
		[((148,431),(533,345)), ((148,431),(560,427))],#Kilaboo_nir1_5000
		# [((533,345),(148,457))],#Kilaboo_nir1_5000
		[((145,177),(473,406)), ((145,177),(294,398))],#12-24-32_nir1_Iran0006_nir1_5000
		[((248,344),(561,66))],#ElHammami_nir1_5000
		[((112,270),(246,340)), ((112,270),(493,324))],#Allende_nir1_15000
		# [((211,336),(118,84))],#12-36-59_nir1_SaU001_nir1_15000
		[((362,325),(199,40))],#12-36-59_nir1_SaU001_nir1_15000
	]
nir2_coordinate_helper_lines = [
	# [((100,100),(550,450))],#20_alb_target_tint25000
	# [((100,100),(550,450))],#20_nir2_25000
	# [((134,369),(525,282))],#Kilabo_nir2_25000
	[((134,369),(525,282))],#Kilabo_nir2_15000
	[((133,148),(525,55))],#Kilabo_nir2_10000
	[((403,262),(40,100)), ((403,262),(540,288))],#15-07-52_nir2_Iran0006_nir2_15000
	[((478,131),(91,158)), ((478,131),(137,377))],#ElHammami_nir2_15000
	[((498,153),(187,70)), ((498,153),(123,178))],#Allende_nir2_25000
	# [((323,5),(418,261))],#SaU001_nir2_25000
	# [((323,5),(418,261))],#SaU001_nir2_25000
	# [((374,157),(451,269)), ((374,157),(310,342))],#20_nir2_15000
	# [((374,157),(451,269)), ((374,157),(310,342))],#20_nir2_10000
	# [((215,31),(425,348)), ((215,31),(565,325))],#ElHammamiB_nir2_15000
	# [((217,145),(427,460)), ((217,145),(566,439))],#ElHammamiB_nir2_10000
	# [((308,166),(92,457)), ((308,166),(541,383))],#15-03-03_nir2_Iran0006_nir2_15000#Duplicate?
	# None, # Averaged spectra
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
]
if averaged_spectra:
	nir2_coordinate_helper_lines = [
		# None, # Averaged spectra
		None,
		None,
		None,
		None,
		None,
		# None,
		# None,
		# None,
		None,
	]
if uniform_spots:
	nir2_coordinate_helper_lines = [
		[((523,308),(134,365)), ((523,308),(120,290))],#Kilabo_nir2_15000
		# [((133,148),(525,55))],#Kilabo_nir2_10000
		[((403,262),(40,100)), ((403,262),(225,75))],#15-07-52_nir2_Iran0006_nir2_15000
		[((478,131),(137,377))],#ElHammami_nir2_15000
		[((507,152),(361,114)), ((507,152),(123,178))],#Allende_nir2_25000
		# [((323,5),(418,261))],#SaU001_nir2_25000
		[((175,10),(327,298))],#SaU001_nir2_25000
	]
vis_pixel_coordinates_based_on_helper_lines = [
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#20_vis_2800
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#20_vis_2800
	# [[0.1, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilaboo_vis_1400
	[[0.0001, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilaboo_vis_1400
	[[0.0001, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilaboo_vis_1400
	[[0.05, 0.3, 0.5, 0.83], [0.001, 0.4, 0.555, 0.8]],#Iran0006_vis_1400
	[[0.001, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#15-11-36_ElHammami_vis_1400
	[[0.08, 0.25, 0.45, 0.83], [0.4, 0.555, 0.8, 0.93]],#Allende_vis_2800
	# [[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],#SaU001_vis_2800
	# [[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],#SaU001_vis_1400
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#20_vis_2800
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#20_vis_1400
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#ElHammamiB_vis_1400
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#ElHammamiB_vis_1400
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#15-06-16_ElHammami_vis_1400#Duplicate?
	# None, # Averaged spectra
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
]
if averaged_spectra:
	vis_pixel_coordinates_based_on_helper_lines = [
		# None, # Averaged spectra
		None,
		None,
		None,
		None,
		None,
		# None,
		# None,
		# None,
		None,
	]
if uniform_spots:
	vis_pixel_coordinates_based_on_helper_lines = [
		[[0.167, 0.385, 0.72], [1]],#Kilaboo_vis_1400
		# [[0.0001, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilaboo_vis_1400
		[[0, 0.6], [0.7]],#Iran0006_vis_1400
		[[0, 0.43, 0.65, 0.705]],#15-11-36_ElHammami_vis_1400
		[[0, 1], [0.25]],#Allende_vis_2800
		# [[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],#SaU001_vis_2800
		[[0.55, 0.86]],#SaU001_vis_1400
	]
nir1_pixel_coordinates_based_on_helper_lines = [
	# [[0.29, 0.4, 0.525, 0.555, 0.7, 0.73, 0.83, 0.88]],#20_nir1_15000
	# [[0.29, 0.4, 0.525, 0.555, 0.7, 0.73, 0.83, 0.88]],#20_nir1_15000
	# [[0.1, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilaboo_nir1_5000
	[[0.0001, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilaboo_nir1_5000
	[[0.0001, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilaboo_nir1_5000
	[[0.05, 0.3, 0.5, 0.83], [0.001, 0.4, 0.555, 0.8]],#12-24-32_nir1_Iran0006_nir1_5000
	[[0.001, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#ElHammami_nir1_5000
	[[0.08, 0.25, 0.45, 0.83], [0.4, 0.555, 0.8, 0.93]],#Allende_nir1_15000
	# [[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],#12-36-59_nir1_SaU001_nir1_15000
	# [[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],#12-36-59_nir1_SaU001_nir1_15000
	# [[0.29, 0.4, 0.525, 0.555, 0.7, 0.73, 0.83, 0.88]],#20_nir1_15000
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#20_nir1_5000
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#ElHammamiB_nir1_5000
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#ElHammamiB_nir1_5000
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#12-18-38_nir1_Iran0006_nir1_5000#Duplicate?
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#12-21-27_nir1_Iran0006_nir1_5000#Duplicate?
	# [[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],#12-34-28_nir1_SaU001_nir1_15000#Duplicate?
	# None, # Averaged spectra
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
]
if averaged_spectra:
	nir1_pixel_coordinates_based_on_helper_lines = [
		# None, # Averaged spectra
		None,
		None,
		None,
		None,
		None,
		# None,
		# None,
		# None,
		None,
	]
if uniform_spots:
	nir1_pixel_coordinates_based_on_helper_lines = [
		[[0.167, 0.385, 0.72], [1]],#Kilaboo_nir1_5000
		# [[0.0001, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilaboo_nir1_5000
		[[0, 0.6], [0.7]],#12-24-32_nir1_Iran0006_nir1_5000
		[[0, 0.43, 0.65, 0.705]],#ElHammami_nir1_5000
		[[0, 1], [0.25]],#Allende_nir1_15000
		# [[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],#12-36-59_nir1_SaU001_nir1_15000
		[[0.55, 0.86]],#12-36-59_nir1_SaU001_nir1_15000
	]
nir2_pixel_coordinates_based_on_helper_lines = [
	# [[0.29, 0.4, 0.525, 0.555, 0.7, 0.73, 0.83, 0.88]],#20_alb_target_tint25000
	# [[0.29, 0.4, 0.525, 0.555, 0.7, 0.73, 0.83, 0.88]],#20_nir2_25000
	# [[0.1, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilabo_nir2_25000
	[[0.0001, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilabo_nir2_15000
	[[0.0001, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilabo_nir2_10000
	[[0.05, 0.3, 0.5, 0.83], [0.001, 0.4, 0.555, 0.8]],#15-07-52_nir2_Iran0006_nir2_15000
	[[0.001, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#ElHammami_nir2_15000
	[[0.08, 0.25, 0.45, 0.83], [0.4, 0.555, 0.8, 0.93]],#Allende_nir2_25000
	# [[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],#SaU001_nir2_25000
	# [[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],#SaU001_nir2_25000
	# [[0.01, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],#20_nir2_15000
	# [[0.01, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],#20_nir2_10000
	# [[0.01, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],#ElHammamiB_nir2_15000
	# [[0.01, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],#ElHammamiB_nir2_10000
	# [[0.08, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],#15-03-03_nir2_Iran0006_nir2_15000#Duplicate?
	# None, # Averaged spectra
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
	# None,
]
if averaged_spectra:
	nir2_pixel_coordinates_based_on_helper_lines = [
		# None, # Averaged spectra
		None,
		None,
		None,
		None,
		None,
		# None,
		# None,
		# None,
		None,
	]
if uniform_spots:
	nir2_pixel_coordinates_based_on_helper_lines = [
		[[0.167, 0.385, 0.72], [1]],#Kilabo_nir2_15000
		# [[0.0001, 0.2, 0.25, 0.4, 0.5, 0.555, 0.83, 0.9]],#Kilabo_nir2_10000
		[[0, 0.6], [0.7]],#15-07-52_nir2_Iran0006_nir2_15000
		[[0, 0.43, 0.65, 0.705]],#ElHammami_nir2_15000
		[[0, 1], [0.25]],#Allende_nir2_25000
		# [[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],#SaU001_nir2_25000
		[[0.55, 0.86]],#SaU001_nir2_25000
	]
vis_averaged_spectra_square_coordinates = [
	# None,
	None,
	None,
	None,
	None,
	None,
	# None,
	# None,
	# ((359,210), (771,210), (771,628), (359,628)),#Kilaboo_vis_1400 # Averaged spectra
	# ((359,210), (771,210), (771,628), (359,628)),#Kilaboo_vis_1400
	# ((359,210), (771,210), (771,628), (359,628)),#Kilaboo_vis_1400
	# ((377,126), (724,126), (724,364), (377,364)),#Iran0006_vis_1400
	# ((203,347), (568,347), (568,603), (203,603)),#15-11-36_ElHammami_vis_1400
	# ((306,292), (619,292), (619,591), (306,591)),#Allende_vis_2800
	# ((134,370), (726,370), (726,849), (134,849)),#SaU001_vis_2800
	# ((171,398), (800,398), (800,886), (171,886)),#SaU001_vis_1400
	# ((238,384), (822,384), (822,974), (238,974)),#ElHammamiB_vis_1400
	# ((238,384), (822,384), (822,974), (238,974)),#ElHammamiB_vis_1400
]
if averaged_spectra:
	vis_averaged_spectra_square_coordinates = [
		# ((359,210), (771,210), (771,628), (359,628)),#Kilaboo_vis_1400 # Averaged spectra # entire asteroid
		# ((359,210), (771,210), (771,628), (359,628)),#Kilaboo_vis_1400
		# ((359,210), (771,210), (771,628), (359,628)),#Kilaboo_vis_1400
		# ((377,126), (724,126), (724,364), (377,364)),#Iran0006_vis_1400
		# ((203,347), (568,347), (568,603), (203,603)),#15-11-36_ElHammami_vis_1400
		# ((306,292), (619,292), (619,591), (306,591)),#Allende_vis_2800
		# ((134,370), (726,370), (726,849), (134,849)),#SaU001_vis_2800
		# ((171,398), (800,398), (800,886), (171,886)),#SaU001_vis_1400
		# ((238,384), (822,384), (822,974), (238,974)),#ElHammamiB_vis_1400
		# ((238,384), (822,384), (822,974), (238,974)),#ElHammamiB_vis_1400
		
		# ((423,463), (787,463), (787,597), (423,597)),#Kilaboo_vis_1400 # Averaged spectra # specific area where possible
		((423,463), (787,463), (787,597), (423,597)),#Kilaboo_vis_1400
		((420,315), (787,315), (787,603), (420,603)),#Kilaboo_vis_1400
		((728,224), (496,224), (496,67), (728,67)),#Iran0006_vis_1400
		((183,347), (538,347), (538,603), (183,603)),#15-11-36_ElHammami_vis_1400
		((388,509), (638,509), (638,571), (388,571)),#Allende_vis_2800
		# ((134,370), (726,370), (726,849), (134,849)),#SaU001_vis_2800
		# ((171,398), (800,398), (800,886), (171,886)),#SaU001_vis_1400
		# ((263,513), (566,513), (566,786), (263,786)),#ElHammamiB_vis_1400
		((263,513), (566,513), (566,786), (263,786)),#ElHammamiB_vis_1400
	]
if uniform_spots:
	vis_averaged_spectra_square_coordinates = [
		None,
		# None,
		None,
		None,
		None,
		# None,
		None,
	]
nir1_averaged_spectra_square_coordinates = [
	# None,
	None,
	None,
	None,
	None,
	None,
	# None,
	# None,
	# ((210,40), (586,40), (586,486), (210,486)),#Kilaboo_nir1_5000 # Averaged spectra
	# ((210,40), (586,40), (586,486), (210,486)),#Kilaboo_nir1_5000
	# ((210,40), (586,40), (586,486), (210,486)),#Kilaboo_nir1_5000
	# ((58,124), (433,124), (433,310), (58,310)),#12-24-32_nir1_Iran0006_nir1_5000
	# ((205,102), (565,102), (565,328), (205,328)),#ElHammami_nir1_5000
	# ((55,92), (455,92), (455,342), (55,342)),#Allende_nir1_15000
	# ((21,27), (603,27), (603,474), (21,474)),#12-36-59_nir1_SaU001_nir1_15000
	# ((21,27), (603,27), (603,474), (21,474)),#12-36-59_nir1_SaU001_nir1_15000
	# ((78,44), (585,44), (585,462), (78,462)),#ElHammamiB_nir1_5000
	# ((78,44), (585,44), (585,462), (78,462)),#ElHammamiB_nir1_5000
]
if averaged_spectra:
	nir1_averaged_spectra_square_coordinates = [
		# ((210,40), (586,40), (586,486), (210,486)),#Kilaboo_nir1_5000 # Averaged spectra
		# ((210,40), (586,40), (586,486), (210,486)),#Kilaboo_nir1_5000
		# ((210,40), (586,40), (586,486), (210,486)),#Kilaboo_nir1_5000
		# ((58,124), (433,124), (433,310), (58,310)),#12-24-32_nir1_Iran0006_nir1_5000
		# ((205,102), (565,102), (565,328), (205,328)),#ElHammami_nir1_5000
		# ((55,92), (455,92), (455,342), (55,342)),#Allende_nir1_15000
		# ((21,27), (603,27), (603,474), (21,474)),#12-36-59_nir1_SaU001_nir1_15000
		# ((21,27), (603,27), (603,474), (21,474)),#12-36-59_nir1_SaU001_nir1_15000
		# ((78,44), (585,44), (585,462), (78,462)),#ElHammamiB_nir1_5000
		# ((78,44), (585,44), (585,462), (78,462)),#ElHammamiB_nir1_5000

		# ((531,348), (193,348), (193,459), (531,459)),#Kilaboo_nir1_5000 # Averaged spectra
		((531,348), (193,348), (193,459), (531,459)),#Kilaboo_nir1_5000
		((533,458), (196,458), (196,193), (533,193)),#Kilaboo_nir1_5000
		((112,206), (346,206), (346,83), (112,83)),#12-24-32_nir1_Iran0006_nir1_5000
		((205,102), (565,102), (565,328), (205,328)),#ElHammami_nir1_5000
		((350,273), (125,273), (125,318), (350,318)),#Allende_nir1_15000
		# ((21,27), (603,27), (603,474), (21,474)),#12-36-59_nir1_SaU001_nir1_15000
		# ((21,27), (603,27), (603,474), (21,474)),#12-36-59_nir1_SaU001_nir1_15000
		# ((264,93), (517,93), (517,332), (264,332)),#ElHammamiB_nir1_5000
		((264,93), (517,93), (517,332), (264,332)),#ElHammamiB_nir1_5000
	]
if uniform_spots:
	nir1_averaged_spectra_square_coordinates = [
		None,
		# None,
		None,
		None,
		None,
		# None,
		None,
	]
nir2_averaged_spectra_square_coordinates = [
	# None,
	None,
	None,
	None,
	None,
	None,
	# None,
	# None,
	# ((92,190), (400,190), (400,473), (92,473)),#Kilabo_nir2_25000 # Averaged spectra
	# ((80,180), (420,180), (420,480), (80,480)),#Kilabo_nir2_15000
	# ((75,30), (475,30), (475,449), (75,449)),#Kilabo_nir2_10000
	# ((105,125), (420,125), (420,373), (105,373)),#15-07-52_nir2_Iran0006_nir2_15000
	# ((140,92), (500,92), (500,358), (140,358)),#ElHammami_nir2_15000
	# ((150,180), (505,180), (505,430), (150,430)),#Allende_nir2_25000
	# ((29,27), (570,27), (570,315), (29,315)),#SaU001_nir2_25000
	# ((29,27), (570,27), (570,315), (29,315)),#SaU001_nir2_25000
	# ((65,25), (260,25), (260,375), (65,375)),#ElHammamiB_nir2_15000
	# ((72,28), (266,28), (266,471), (72,471)),#ElHammamiB_nir2_10000
]
if averaged_spectra:
	nir2_averaged_spectra_square_coordinates = [
		# ((92,190), (400,190), (400,473), (92,473)),#Kilabo_nir2_25000 # Averaged spectra
		# ((80,180), (420,180), (420,480), (80,480)),#Kilabo_nir2_15000
		# ((75,30), (475,30), (475,449), (75,449)),#Kilabo_nir2_10000
		# ((105,125), (420,125), (420,373), (105,373)),#15-07-52_nir2_Iran0006_nir2_15000
		# ((140,92), (500,92), (500,358), (140,358)),#ElHammami_nir2_15000
		# ((150,180), (505,180), (505,430), (150,430)),#Allende_nir2_25000
		# ((29,27), (570,27), (570,315), (29,315)),#SaU001_nir2_25000
		# ((29,27), (570,27), (570,315), (29,315)),#SaU001_nir2_25000
		# ((65,25), (260,25), (260,375), (65,375)),#ElHammamiB_nir2_15000
		# ((72,28), (266,28), (266,471), (72,471)),#ElHammamiB_nir2_10000

		# ((138,286), (466,286), (466,369), (138,369)),#Kilabo_nir2_25000 # Averaged spectra
		((138,286), (466,286), (466,369), (138,369)),#Kilabo_nir2_15000
		((135,62), (464,62), (464,325), (135,325)),#Kilabo_nir2_10000
		((437,238), (227,238), (227,396), (437,396)),#15-07-52_nir2_Iran0006_nir2_15000
		((140,92), (500,92), (500,358), (140,358)),#ElHammami_nir2_15000
		((271,190), (493,190), (493,142), (271,142)),#Allende_vis_2800
		# ((29,27), (570,27), (570,315), (29,315)),#SaU001_nir2_25000
		# ((29,27), (570,27), (570,315), (29,315)),#SaU001_nir2_25000
		# ((87,1), (335,1), (335,149), (87,149)),#ElHammamiB_nir2_15000
		((93,23), (344,23), (344,264), (93,264)),#ElHammamiB_nir2_10000
	]
if uniform_spots:
	nir2_averaged_spectra_square_coordinates = [
		None,
		# None,
		None,
		None,
		None,
		# None,
		None,
	]
vis_calibration_data_directories = [
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-47-39_vis_20_vis_1400/acq_000', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-47-39_vis_20_vis_1400/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-47-39_vis_20_vis_1400/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-47-39_vis_20_vis_1400/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-47-39_vis_20_vis_1400/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-41-10_vis_20_vis_2800/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-41-10_vis_20_vis_2800/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-47-39_vis_20_vis_1400/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-47-39_vis_20_vis_1400/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_14-47-39_vis_20_vis_1400/acq_000',
]
if averaged_spectra:
	vis_calibration_data_directories = [
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400', # Averaged spectra (copied from above)
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
	]
if uniform_spots:
	vis_calibration_data_directories = [
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400',
	]
nir1_calibration_data_directories = [
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-43-27_nir1_20_nir1_5000/acq_000', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-43-27_nir1_20_nir1_5000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-43-27_nir1_20_nir1_5000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-43-27_nir1_20_nir1_5000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-43-27_nir1_20_nir1_5000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_11-51-32_nir1_20_nir1_15000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_11-51-32_nir1_20_nir1_15000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_11-51-32_nir1_20_nir1_15000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-43-27_nir1_20_nir1_5000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-30_12-43-27_nir1_20_nir1_5000/acq_000',
]
if averaged_spectra:
	nir1_calibration_data_directories = [
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000', # Averaged spectra (copied from above)
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
	]
if uniform_spots:
	nir1_calibration_data_directories = [
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000',
	]
nir2_calibration_data_directories = [
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-35-50_nir2_20_nir2_10000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_14-36-57_nir2_20_nir2_25000/acq_000', # Averaged spectra (copied from above)
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-33-06_nir2_20_nir2_15000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-35-50_nir2_20_nir2_10000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-33-06_nir2_20_nir2_15000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-33-06_nir2_20_nir2_15000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_14-36-57_nir2_20_nir2_25000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_14-36-57_nir2_20_nir2_25000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_14-36-57_nir2_20_nir2_25000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-33-06_nir2_20_nir2_15000/acq_000',
	# '/home/sysa/HERA/ESA meteorites/2024-08-29_15-35-50_nir2_20_nir2_10000/acq_000',
]
if averaged_spectra:
	nir2_calibration_data_directories = [
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000', # Averaged spectra (copied from above)
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-35-50_nir2_20_nir2_10000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-35-50_nir2_20_nir2_10000',
	]
if uniform_spots:
	nir2_calibration_data_directories = [
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-35-50_nir2_20_nir2_10000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000',
		# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000',
		'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000',
	]
nir_spectra_shifting_list = [
	None,
	None,
	None,
	None,
	None,
]
if averaged_spectra:
	nir_spectra_shifting_list = [
		'nir1', # Kilabo nir2 15000
		'nir1', # Kilabo nir2 10000
		'nir1', # Iran
		'nir1', # ElHammami
		'nir2', # Allende
		'nir2', # ElHammamiB
	]
if uniform_spots:
	nir_spectra_shifting_list = [
		'nir1', # Kilabo
		'nir2', # Iran
		'nir2', # ElHammami
		'nir1', # Allende
		None, # SaU
	]
vis_offsets = [
	0,
	0,
	0,
	0,
	0,
]
if averaged_spectra:
	vis_offsets = [
		0, # Kilabo nir2 15000
		0, # Kilabo nir2 10000
		0.08, # Iran
		0, # ElHammami
		0, # Allende
		0, # ElHammamiB
	]
if uniform_spots:
	vis_offsets = [
		0, # Kilabo
		0, # Iran
		0, # ElHammami
		0, # Allende
		0, # SaU
	]

for target_index in range(len(vis_coordinate_helper_lines)):
	if vis_coordinate_helper_lines[target_index] is None:
		vis_pixel_coordinates.append(None)
		nir1_pixel_coordinates.append(None)
		nir2_pixel_coordinates.append(None)
	else:
		vis_target_pixels = find_target_pixels_of_single_image(# old = {'VIS': [(480,506),(530,677)], 'NIR1': [(347,132),(295,298)], 'NIR2': [(438,397),(481,242)]}, new = [(480,506),(530,677)]
			vis_coordinate_helper_lines[target_index],
			vis_pixel_coordinates_based_on_helper_lines[target_index]
		)
		nir1_target_pixels = find_target_pixels_of_single_image(# old = {'VIS': [(480,506),(530,677)], 'NIR1': [(347,132),(295,298)], 'NIR2': [(438,397),(481,242)]}, new = [(480,506),(530,677)]
			nir1_coordinate_helper_lines[target_index],
			nir1_pixel_coordinates_based_on_helper_lines[target_index]
		)
		nir2_target_pixels = find_target_pixels_of_single_image(# old = {'VIS': [(480,506),(530,677)], 'NIR1': [(347,132),(295,298)], 'NIR2': [(438,397),(481,242)]}, new = [(480,506),(530,677)]
			nir2_coordinate_helper_lines[target_index],
			nir2_pixel_coordinates_based_on_helper_lines[target_index]
		)
		vis_pixel_coordinates.append(vis_target_pixels)
		nir1_pixel_coordinates.append(nir1_target_pixels)
		nir2_pixel_coordinates.append(nir2_target_pixels)

ESA_meteorite_processing = False
if ESA_meteorite_processing:
	
	process_grouped_directories_with_specified_metadata(
		vis_metadata_directories,
		nir1_metadata_directories,
		nir2_metadata_directories,
		vis_actual_data_directories,
		nir1_actual_data_directories,
		nir2_actual_data_directories,
		vis_matching_dark_directories,
		nir1_matching_dark_directories,
		nir2_matching_dark_directories,
		output_directories,
		vis_pixel_coordinates, # [[(480,506),(530,677)], None, None]
		nir1_pixel_coordinates,
		nir2_pixel_coordinates,
		vis_rectangle_coordinates,
		nir1_rectangle_coordinates,
		nir2_rectangle_coordinates,
		vis_averaged_spectra_square_coordinates,
		nir1_averaged_spectra_square_coordinates,
		nir2_averaged_spectra_square_coordinates,
		vis_calibration_data_directories,
		nir1_calibration_data_directories,
		nir2_calibration_data_directories,
		nir_spectra_shifting_list,
		vis_offsets,
		test_mode=False
	)
	
	for example_index in range(len(vis_actual_data_directories)):
		if vis_pixel_location_bins[example_index]:
			visualize_pixel_coordinates(
				vis_pixel_location_bins[example_index],
				vis_calibration_data_directories[example_index],
				output_directories[example_index],
				vis_pixel_coordinates[example_index],
				vis_coordinate_helper_lines[example_index],
				vis_rectangle_coordinates[example_index],
				vis_averaged_spectra_square_coordinates[example_index],
				image_shape=(1024, 1024), # [(1024, 1024), (506, 636), (506, 636)],
				calibration_image_shape=(1024, 1024),
				channel='vis',
				show_helper_lines=True,
				show_calibration_squares_in_calibration_file=True,
				show_calibration_squares_in_main_image=False,
				test_mode=False
			)
		if nir1_pixel_location_bins[example_index]:
			visualize_pixel_coordinates(
				nir1_pixel_location_bins[example_index],
				nir1_calibration_data_directories[example_index],
				output_directories[example_index],
				nir1_pixel_coordinates[example_index],
				nir1_coordinate_helper_lines[example_index],
				nir1_rectangle_coordinates[example_index],
				nir1_averaged_spectra_square_coordinates[example_index],
				image_shape=(506, 636), # [(1024, 1024), (506, 636), (506, 636)],
				calibration_image_shape=(506, 636),
				channel='nir1',
				show_helper_lines=True,
				show_calibration_squares_in_calibration_file=True,
				show_calibration_squares_in_main_image=False,
				test_mode=False
			)
		if nir2_pixel_location_bins[example_index]:
			visualize_pixel_coordinates(
				nir2_pixel_location_bins[example_index],
				nir2_calibration_data_directories[example_index],
				output_directories[example_index],
				nir2_pixel_coordinates[example_index],
				nir2_coordinate_helper_lines[example_index],
				nir2_rectangle_coordinates[example_index],
				nir2_averaged_spectra_square_coordinates[example_index],
				image_shape=(506, 636), # [(1024, 1024), (506, 636), (506, 636)],
				calibration_image_shape=(506, 636),
				channel='nir2',
				show_helper_lines=True,
				show_calibration_squares_in_calibration_file=True,
				show_calibration_squares_in_main_image=False,
				test_mode=False
			)

# Pixel locations =====================================================================

example_dark_subtracted_paths = [
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_13-32-44_nir2_20_alb_target_tint25000/dc_2_exp_000.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-43-36_nir2_Kilabo_nir2_25000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-03-03_nir2_Iran0006_nir2_15000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-07-52_nir2_Iran0006_nir2_15000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-21-37_nir2_Allende_nir2_25000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-29-28_nir2_SaU001_nir2_25000/dc_2_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000/dc_2_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-35-50_nir2_20_nir2_10000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000/dc_2_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-18-38_nir1_Iran0006_nir1_5000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-21-27_nir1_Iran0006_nir1_5000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-31-12_nir1_Allende_nir1_15000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-34-28_nir1_SaU001_nir1_15000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-36-59_nir1_SaU001_nir1_15000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000/dc_1_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000/dc_1_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800/dc_0_exp_005.bin',
	# '/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-54-01_vis_Kilaboo_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-00-42_vis_Iran0006_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-06-16_vis_ElHammami_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-11-36_vis_ElHammami_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-17-13_vis_Allende_vis_2800/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-22-37_vis_SaU001_vis_2800/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-27-59_vis_SaU001_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_14-36-57_nir2_20_nir2_25000/dc_2_exp_005.bin', # Calibration images (ESA 20)
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-33-06_nir2_20_nir2_15000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-29_15-35-50_nir2_20_nir2_10000/dc_2_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_11-51-32_nir1_20_nir1_15000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_12-43-27_nir1_20_nir1_5000/dc_1_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-41-10_vis_20_vis_2800/dc_0_exp_005.bin',
	'/home/sysa/HERA/test_data/test_outputs/ESA meteorites dark subtracted (16-bit signed, NIR cropped to 636x506)/2024-08-30_14-47-39_vis_20_vis_1400/dc_0_exp_005.bin',
]
pixel_location_plot_output_directories = [
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_13-32-44_nir2_20_alb_target_tint25000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_14-36-57_nir2_20_nir2_25000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_14-43-36_nir2_Kilabo_nir2_25000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_14-51-53_nir2_Kilabo_nir2_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_14-55-51_nir2_Kilabo_nir2_10000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_15-03-03_nir2_Iran0006_nir2_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_15-07-52_nir2_Iran0006_nir2_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_15-12-43_nir2_ElHammami_nir2_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_15-21-37_nir2_Allende_nir2_25000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_15-29-28_nir2_SaU001_nir2_25000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_15-33-06_nir2_20_nir2_15000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_15-35-50_nir2_20_nir2_10000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_11-51-32_nir1_20_nir1_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_12-18-38_nir1_Iran0006_nir1_5000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_12-21-27_nir1_Iran0006_nir1_5000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_12-24-32_nir1_Iran0006_nir1_5000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_12-27-25_nir1_ElHammami_nir1_5000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_12-31-12_nir1_Allende_nir1_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_12-34-28_nir1_SaU001_nir1_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_12-36-59_nir1_SaU001_nir1_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_12-43-27_nir1_20_nir1_5000',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_14-41-10_vis_20_vis_2800',
	# f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_14-47-39_vis_20_vis_1400',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_14-54-01_vis_Kilaboo_vis_1400',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_15-00-42_vis_Iran0006_vis_1400',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_15-06-16_vis_ElHammami_vis_1400',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_15-11-36_vis_ElHammami_vis_1400',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_15-17-13_vis_Allende_vis_2800',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_15-22-37_vis_SaU001_vis_2800',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_15-27-59_vis_SaU001_vis_1400',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_15-33-30_vis_ElHammamiB_vis_1400',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_14-36-57_nir2_20_nir2_25000', # Calibration images (ESA 20)
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_15-33-06_nir2_20_nir2_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-29_15-35-50_nir2_20_nir2_10000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_11-51-32_nir1_20_nir1_15000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_12-43-27_nir1_20_nir1_5000',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_14-41-10_vis_20_vis_2800',
	f'/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorite_pixel_locations/2024-08-30_14-47-39_vis_20_vis_1400',
]
specific_target_pixels = []
coordinate_helper_lines = [
	# [((100,100),(550,450))],
	# [((100,100),(550,450))],
	[((47,333),(525,282)), ((47,333),(372,105))],
	[((47,333),(525,282)), ((47,333),(372,105))],
	[((44,107),(525,55)), ((44,107),(553,410))],
	[((308,166),(92,457)), ((308,166),(541,383))],
	[((309,76),(92,364)), ((309,76),(540,288))],
	[((546,88),(93,158)), ((546,88),(141,377))],
	[((498,153),(279,31)), ((498,153),(187,70))],
	[((323,5),(418,261))],
	# [((374,157),(451,269)), ((374,157),(310,342))],
	# [((374,157),(451,269)), ((374,157),(310,342))],
	[((215,31),(425,348)), ((215,31),(565,325))],
	[((217,145),(427,460)), ((217,145),(566,439))],
	# [((100,100),(550,450))],
	[((533,345),(98,187)), ((533,345),(148,457))],
	[((529,170),(52,239)), ((529,170),(469,322))],
	[((529,214),(56,283)), ((529,214),(470,367))],
	[((527,262),(57,331)), ((527,262),(469,415))],
	[((185,346),(628,278)), ((185,346),(561,66))],
	[((122,273),(420,395)), ((122,273),(493,324))],#allende_nir1
	[((211,451),(118,197))],
	[((211,336),(118,84))],
	[((387,214),(409,7)), ((387,214),(554,16))],
	# [((387,214),(47,291)), ((387,214),(81,481))],
	# [((387,214),(779,734)), ((387,214),(459,923))],
	# [((387,214),(779,734)), ((387,214),(459,923))],
	[((427,466),(910,312)), ((427,466),(828,599))],
	[((756,296),(527,372)), ((756,296),(552,302))],
	[((637,306),(338,333)), ((637,306),(349,257))],
	[((635,568),(222,571)), ((635,568),(255,444))],
	[((642,503),(334,675)), ((642,503),(240,589))],#Allende_vis
	[((392,540),(285,837))],
	[((392,540),(285,837))],
	[((404,652),(638,321)), ((404,652),(801,355))],#ElHammamiB_vis
	[((1,1),(1,50)), ((1,1),(50,1))], # Calibration images (ESA 20)
	[((1,1),(1,50)), ((1,1),(50,1))],
	[((1,1),(1,50)), ((1,1),(50,1))],
	[((1,1),(1,50)), ((1,1),(50,1))],
	[((1,1),(1,50)), ((1,1),(50,1))],
	[((1,1),(1,50)), ((1,1),(50,1))],
	[((1,1),(1,50)), ((1,1),(50,1))],
]
pixel_coordinates_based_on_helper_lines = [
	# [[0.29, 0.4, 0.525, 0.555, 0.7, 0.73, 0.83, 0.88]],
	# [[0.29, 0.4, 0.525, 0.555, 0.7, 0.73, 0.83, 0.88]],
	[[0.1, 0.35, 0.7, 0.83], [0.4, 0.555, 0.8, 0.93]],
	[[0.1, 0.35, 0.7, 0.83], [0.4, 0.555, 0.8, 0.93]],
	[[0.08, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],
	[[0.08, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],
	[[0.08, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],
	[[0.08, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],
	[[0.08, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],
	[[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],
	# [[0.01, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],
	# [[0.01, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],
	[[0.01, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],
	[[0.01, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],
	# [[0.29, 0.4, 0.525, 0.555, 0.7, 0.73, 0.83, 0.88]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.08, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],
	[[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],
	[[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	# [[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.08, 0.25, 0.5, 0.83], [0.4, 0.555, 0.8, 0.93]],#Allende_vis
	[[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],
	[[0.01, 0.1, 0.25, 0.4, 0.58, 0.63, 0.83, 0.93]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],#ElHammamiB_vis
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]], # Calibration images (ESA 20)
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
	[[0.1, 0.25, 0.5, 0.83], [0.2, 0.4, 0.555, 0.8]],
]
rectangle_coordinates = [
	# ((626,5), (631,5), (631,10), (626,10)),
	# ((626,5), (631,5), (631,10), (626,10)),
	((92,190), (400,190), (400,473), (92,473)),#Kilabo_nir2_25000
	((80,180), (420,180), (420,480), (80,480)),#Kilabo_nir2_15000
	((75,30), (475,30), (475,449), (75,449)),#Kilabo_nir2_10000
	((115,225), (432,225), (432,450), (115,450)),#2024-08-29_15-03-03_nir2_Iran0006_nir2_15000
	((105,125), (420,125), (420,373), (105,373)),#2024-08-29_15-07-52_nir2_Iran0006_nir2_15000
	((140,92), (500,92), (500,358), (140,358)),#ElHammami_nir2_15000
	((150,180), (505,180), (505,430), (150,430)),#Allende_nir2_25000
	((29,27), (570,27), (570,315), (29,315)),#SaU001_nir2_25000
	# ((550,355), (555,355), (555,360), (550,360)),
	# ((550,355), (555,355), (555,360), (550,360)),
	((65,25), (260,25), (260,375), (65,375)),#ElHammamiB_nir2_15000
	((72,28), (266,28), (266,471), (72,471)),#ElHammamiB_nir2_10000
	# ((625,5), (630,5), (630,10), (625,10)),
	((210,40), (586,40), (586,486), (210,486)),#Kilaboo_nir1_5000
	((69,28), (442,28), (442,220), (69,220)),#2024-08-30_12-18-38_nir1_Iran0006_nir1_5000
	((73,72), (443,72), (443,257), (73,257)),#2024-08-30_12-21-27_nir1_Iran0006_nir1_5000
	((58,124), (433,124), (433,310), (58,310)),#2024-08-30_12-24-32_nir1_Iran0006_nir1_5000
	((205,102), (565,102), (565,328), (205,328)),#ElHammami_nir1_5000
	((55,92), (455,92), (455,342), (55,342)),#Allende_nir1_15000
	((29,139), (600,139), (600,480), (29,480)),#2024-08-30_12-34-28_nir1_SaU001_nir1_15000
	((21,27), (603,27), (603,474), (21,474)),#2024-08-30_12-36-59_nir1_SaU001_nir1_15000
	((78,44), (585,44), (585,462), (78,462)),#ElHammamiB_nir1_5000
	# ((1,1), (6,1), (6,6), (1,6)),
	# ((620,11), (625,11), (625,16), (620,16)),
	# ((620,11), (625,11), (625,16), (620,16)),
	((359,210), (771,210), (771,628), (359,628)),#Kilaboo_vis_1400
	((377,126), (724,126), (724,364), (377,364)),#Iran0006_vis_1400
	((201,153), (588,153), (588,339), (201,339)),#2024-08-30_15-06-16_vis_ElHammami_vis_1400
	((203,347), (568,347), (568,603), (203,603)),#2024-08-30_15-11-36_vis_ElHammami_vis_1400
	((306,292), (619,292), (619,591), (306,591)),#Allende_vis_2800
	((134,370), (726,370), (726,849), (134,849)),#SaU001_vis_2800
	((171,398), (800,398), (800,886), (171,886)),#SaU001_vis_1400
	((238,384), (822,384), (822,974), (238,974)),#ElHammamiB_vis_1400
	((40,40), (600,40), (600,475), (40,475)), # Calibration images (ESA 20)
	((40,40), (500,40), (500,250), (40,250)),
	((40,40), (500,40), (500,250), (40,250)),
	((40,40), (600,40), (600,475), (40,475)),
	((40,40), (600,40), (600,475), (40,475)),
	((250,200), (700,200), (700,800), (250,800)),
	((250,200), (700,200), (700,800), (250,800)),
]
pixel_location_example_shapes = [
	# (506, 636),
	# (506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	# (506, 636),
	# (506, 636),
	(506, 636),
	(506, 636),
	# (506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	# (506, 636),
	# (1024, 1024),
	# (1024, 1024),
	(1024, 1024),
	(1024, 1024),
	(1024, 1024),
	(1024, 1024),
	(1024, 1024),
	(1024, 1024),
	(1024, 1024),
	(1024, 1024),
	(506, 636), # Calibration images (ESA 20)
	(506, 636),
	(506, 636),
	(506, 636),
	(506, 636),
	(1024, 1024),
	(1024, 1024),
]

for target_index in range(len(coordinate_helper_lines)):
	target_pixels = find_target_pixels_of_single_image(# old = {'VIS': [(480,506),(530,677)], 'NIR1': [(347,132),(295,298)], 'NIR2': [(438,397),(481,242)]}, new = [(480,506),(530,677)]
		coordinate_helper_lines[target_index],
		pixel_coordinates_based_on_helper_lines[target_index]
	)
	specific_target_pixels.append(target_pixels)

plot_pixel_locations = False
if plot_pixel_locations:
	for example_index in range(len(example_dark_subtracted_paths)):
		visualize_pixel_coordinates(
			example_dark_subtracted_paths[example_index],
			None,
			pixel_location_plot_output_directories[example_index],
			specific_target_pixels[example_index],
			coordinate_helper_lines[example_index],
			rectangle_coordinates[example_index],
			None,
			image_shape=pixel_location_example_shapes[example_index], # [(1024, 1024), (506, 636), (506, 636)],
			calibration_image_shape=None,
			channel=None,
			show_helper_lines=True,
			show_calibration_squares_in_calibration_file=False,
			show_calibration_squares_in_main_image=True,
			test_mode=True
		)