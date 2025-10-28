import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import pandas as pd
import os
import time
import json

def save_setpoint_calibration_graphs(
		output_folder: str,
		setpoint: np.ndarray,
		brightness: np.ndarray,
		rate_of_change: np.ndarray,
		fine_setpoints: np.ndarray,
		spline_line: np.ndarray,
		peak_setpoint: int
	):
	
	# Create subplots
	fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

	# Plot brightness
	axs[0].plot(setpoint, brightness, color='blue', label='Brightness')
	axs[0].set_ylabel('Brightness')
	axs[0].set_title('Brightness vs Setpoint')
	axs[0].grid(True)
	axs[0].legend()

	# Plot rate of change
	axs[1].plot(setpoint, rate_of_change, color='red', label='Rate of Change')
	axs[1].plot(fine_setpoints, spline_line, label='Spline Interpolation', linestyle='--')
	axs[1].plot([], [], ' ', label=f'Peak Spline Setpoint: {peak_setpoint}')
	axs[1].set_xlabel('Setpoint')
	axs[1].set_ylabel('d(Brightness)/d(Setpoint)')
	axs[1].set_title('Rate of Change of Brightness')
	axs[1].grid(True)
	axs[1].legend()

	# Layout adjustment
	plt.tight_layout()
	output_path = os.path.join(output_folder, f"setpoint_recalibration_graph.png")
	plt.savefig(output_path)
	plt.close(fig)

def calculate_peak_setpoint(path: str, output_folder: str, save: bool = True):
	data = np.loadtxt(path)
	setpoint = data[:, 0]
	brightness = data[:, 1]
	rate_of_change = np.gradient(brightness, setpoint)

	# Interpolate the derivative using a smoothing spline
	spline = UnivariateSpline(setpoint[3:], rate_of_change[3:], s=0)

	# Find the setpoint where the interpolated derivative is maximum
	res = minimize_scalar(lambda x: -spline(x), bounds=(setpoint[3:].min(), setpoint[3:].max()), method='bounded')
	peak_setpoint = res.x
	peak_value = spline(peak_setpoint)

	print(f"Peak of rate_of_change at setpoint: {peak_setpoint:.2f} (value: {peak_value:.2f})")
	
	# Graphs
	fine_setpoints = np.linspace(setpoint.min(), setpoint.max(), 1000)
	spline_line = spline(fine_setpoints)
	if save:
		save_setpoint_calibration_graphs(
			output_folder,
			setpoint,
			brightness,
			rate_of_change,
			fine_setpoints,
			spline_line,
			round(peak_setpoint)
		)
	
	return peak_setpoint

def calculate_peak_setpoint_from_lists(setpoint: list, brightness: list, output_folder: str, save: bool = True):
	setpoint = np.array(setpoint)
	brightness = np.array(brightness)
	rate_of_change = np.gradient(brightness, setpoint)

	# Interpolate the derivative using a smoothing spline
	# spline = UnivariateSpline(setpoint[3:], rate_of_change[3:], s=0)
	spline = UnivariateSpline(setpoint, rate_of_change, s=0)

	# Find the setpoint where the interpolated derivative is maximum
	res = minimize_scalar(lambda x: -spline(x), bounds=(setpoint.min(), setpoint.max()), method='bounded')
	peak_setpoint = res.x
	peak_value = spline(peak_setpoint)

	print(f"Peak of rate_of_change at setpoint: {peak_setpoint:.2f} (value: {peak_value:.2f})")
	
	# Graphs
	fine_setpoints = np.linspace(setpoint.min(), setpoint.max(), 1000)
	spline_line = spline(fine_setpoints)
	if save:
		save_setpoint_calibration_graphs(
			output_folder,
			setpoint,
			brightness,
			rate_of_change,
			fine_setpoints,
			spline_line,
			round(peak_setpoint)
		)
	
	return peak_setpoint

def setpoint_to_wavelength(channel: str, order: str, setpoint: int, edge_correction: int = 0):
	if channel == 'vis':
		if order == 'lo':
			wl = vis_lo_setpoint_to_wavelength(setpoint, edge_correction)
		elif order == 'ho':
			wl = vis_ho_setpoint_to_wavelength(setpoint, edge_correction)
	elif channel == 'nir1':
		if order == 'lo':
			wl = nir1_lo_setpoint_to_wavelength(setpoint, edge_correction)
		elif order == 'ho':
			wl = nir1_ho_setpoint_to_wavelength(setpoint, edge_correction)
	elif channel == 'nir2':
		if order == 'lo':
			wl = nir2_lo_setpoint_to_wavelength(setpoint, edge_correction)
		elif order == 'ho':
			wl = nir2_ho_setpoint_to_wavelength(setpoint, edge_correction)
	return wl

def vis_lo_setpoint_to_wavelength(setpoint: int, edge_correction: int = 0):
	return np.round(0.1256 * (setpoint - edge_correction) - 1511.3).astype(int)

def vis_ho_setpoint_to_wavelength(setpoint: int, edge_correction: int = 0):
	return np.round(((setpoint - edge_correction) - 12275) / 11.25).astype(int)

def nir1_lo_setpoint_to_wavelength(setpoint: int, edge_correction: int = 0):
	return np.round(0.2696 * (setpoint - edge_correction) - 3845.8).astype(int)

def nir1_ho_setpoint_to_wavelength(setpoint: int, edge_correction: int = 0):
	return np.round(((setpoint - edge_correction) - 13866) / 7.6772).astype(int)

def nir2_lo_setpoint_to_wavelength(setpoint: int, edge_correction: int = 0):
	return np.round(0.2687 * (setpoint - edge_correction) - 3872.6).astype(int)

def nir2_ho_setpoint_to_wavelength(setpoint: int, edge_correction: int = 0):
	return np.round(((setpoint - edge_correction) - 14922) / 6.875).astype(int)

def write_excel(
		path: str,
		output_folder: str,
		original_setpoints: np.ndarray,
		original_wl: np.ndarray,
		recalibrated_setpoints: np.ndarray,
		recalibrated_wl: np.ndarray
	):
	
	df = pd.DataFrame({
        "original_sp": original_setpoints,
        "original_wl": original_wl,
        "recalibrated_sp": recalibrated_setpoints,
        "recalibrated_wl": recalibrated_wl
    })

	df.to_excel(os.path.join(output_folder, f'setpoints_and_wavelengths.xlsx'), index=False)

def read_original_setpoints_wavelengths(path: str, channel: str, order: str):
	data = np.loadtxt(path)
	original_setpoints = data[:, 0].astype(int)
	if channel == 'vis':
		if order == 'lo':
			original_wl = vis_lo_setpoint_to_wavelength(original_setpoints, 0)
		elif order == 'ho':
			original_wl = vis_ho_setpoint_to_wavelength(original_setpoints, 0)
	elif channel == 'nir1':
		if order == 'lo':
			original_wl = nir1_lo_setpoint_to_wavelength(original_setpoints, 0)
		elif order == 'ho':
			original_wl = nir1_ho_setpoint_to_wavelength(original_setpoints, 0)
	elif channel == 'nir2':
		if order == 'lo':
			original_wl = nir2_lo_setpoint_to_wavelength(original_setpoints, 0)
		elif order == 'ho':
			original_wl = nir2_ho_setpoint_to_wavelength(original_setpoints, 0)
	return original_setpoints, original_wl
	

def recalibrate_folder(path: str, channel: str, order: str, output_folder: str, save: bool = True):
	timestamp = time.time()
	local_time = time.localtime(timestamp)
	formatted_time = time.strftime("%Y-%m-%dT%H:%M:%S", local_time)
	path_parts = path.strip('/').split('/')
	output_folder = os.path.join(output_folder, f'{formatted_time}_{path_parts[-3]}_recalibrated')
	if save:
		os.makedirs(output_folder, exist_ok=True)

	original_setpoints, original_wl = read_original_setpoints_wavelengths(path, channel, order)
	middle_setpoint_index = len(original_setpoints) // 2
	middle_setpoint = original_setpoints[middle_setpoint_index]

	peak_setpoint = calculate_peak_setpoint(
		path,
		output_folder,
		save=save
	)
	peak_wavelength = setpoint_to_wavelength(channel, order, peak_setpoint)
	edge_correction = peak_setpoint - middle_setpoint
	recalibrated_setpoints = original_setpoints + edge_correction
	recalibrated_setpoints = recalibrated_setpoints.astype(int)
	recalibrated_wl = setpoint_to_wavelength(channel, order, recalibrated_setpoints)
	
	if save:
		write_excel(path, output_folder, original_setpoints, original_wl, recalibrated_setpoints, recalibrated_wl)

	return edge_correction, peak_wavelength

# def calculate_bin_averages(input_folder: str, output_folder: str):
# 	"""
# 	Loads and calculates the average value of each bin file in the input_folder.
# 	The files are loaded in alphabetical order, and
# 	the averages are saved into a txt file {filename}_result.txt in the output_folder.
# 	The result.txt has one column where the lines represent the bin files.
# 	"""

# 	# Ensure output folder exists
# 	os.makedirs(output_folder, exist_ok=True)

# 	# Get sorted list of bin files
# 	bin_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith('.bin'))

# 	for file_name in bin_files:
# 		file_path = os.path.join(input_folder, file_name)

# 		# Load binary file as float32 (adjust dtype if necessary)
# 		data = np.fromfile(file_path, dtype=np.float32)

# 		# Calculate average
# 		avg_value = np.mean(data)

# 		# Create output file path
# 		result_file = os.path.join(output_folder, f"result.txt")

# 		# Save average to file
# 		with open(result_file, "a") as f:
# 			f.write(f"{avg_value}\n")

# calculate_bin_averages(
# 	'/home/sysa/HERA/test_data/2025-08-25_wavelength_calibration/acqseq_302_decompressed/acq_000',
# 	'/home/sysa/HERA/test_data/2025-08-25_wavelength_calibration/acqseq_302_decompressed/acq_000'
# )

def get_brightness_list(calibpath="/home/sysa/HERA/test_data/2025-08-25_wavelength_calibration/acqseq_301_decompressed/meta/calib.json", channel: str = 'vis'):
	# Load JSON file
	with open(calibpath, "r") as f:
		data = json.load(f)

	# Navigate to the VIS object (assuming single top-level and sub-level keys)
	first_level = next(iter(data.values()))
	second_level = next(iter(first_level.values()))
	third_level = next(iter(second_level.values()))

	# If channel is 'nir2', directly access the NIR2 object
	# This is because NIR1 and NIR2 were saved in the same calib.json file for some reason
	if channel == 'nir2':
		nir2_data = second_level["NIR2"]
		nir2_sorted_indices = sorted(nir2_data.keys(), key=lambda x: int(x))
		nir2_brightness_list = [nir2_data[i]["GRAY"] for i in nir2_sorted_indices]
		return nir2_brightness_list
	
	# Sort keys alphabetically as strings and extract brightness values
	sorted_indices = sorted(third_level.keys(), key=lambda x: int(x))
	brightness_list = [third_level[i]["GRAY"] for i in sorted_indices]

	return brightness_list

def get_setpoint_list(configpath="/home/sysa/HERA/test_data/2025-08-25_wavelength_calibration/acqseq_301_decompressed/meta/config.json", channel: str = 'vis'):
	# Load JSON file
	with open(configpath, "r") as f:
		data = json.load(f)

	task_file_name = channel.lower() + 'TaskFile'
	task_file = data[task_file_name]
	task_file_sequences = [task_file[i:i+8] for i in range(0, len(task_file), 8)]
	setpoints = [sequence[1] for sequence in task_file_sequences]
	return setpoints

# ESA meteorite wavelength recalibration ===============================================================================================

'''
I would recommend using the "2025-08-25 wavelength recalibration" section below instead of this one.

This section recalibrates the wavelengths from precomputed "results.txt" files
which contain the average brightnesses of each image. If result.txt files are not provided,
follow the 2025-08-25 wavelength recalibration section below. It retrieves these
values from the calib.json files.
'''

edge_results = [
	'/home/sysa/HERA/ESA meteorites/2024-08-29_15-50-26_nir2_edge_l_nir2_10000/acq_000/result.txt', # Wavelength difference over 10 nm
	'/home/sysa/HERA/ESA meteorites/2024-08-30_12-46-58_nir1_nir1_edge_l_20000/acq_000/result.txt', # Fine
	'/home/sysa/HERA/ESA meteorites/2024-08-30_15-41-29_vis_vis_edge_h_2800/acq_000/result.txt', # Fine
]
channels = [
	'nir2',
	'nir1',
	'vis',
]
orders = [
	'lo',
	'lo',
	'ho',
]
output_folder = '/home/sysa/HERA/test_data/test_outputs'

recalibrate_with_result_files = False
if recalibrate_with_result_files:
	for edge, channel, order in zip(edge_results, channels, orders):
		print(recalibrate_folder(edge, channel, order, output_folder, save=False))

# Results from the above section:

# Edge corrections
# NIR2: -36
# NIR1: 43
# VIS: 96

# Peak wavelengths
# NIR2: 1181
# NIR1: 851
# VIS: 649

# NIR2 lower edge 1200
# NIR1 lower edge 850
# VIS lower edge 650


# 2025-08-25 wavelength recalibration =====================================================================================================

'''
This section recalibrates the wavelengths from the brightness values similarly as above,
but retrieves the brightness values from the calib.json files instead of result.txt.

Take into consideration that sometimes the calib.json files are structured differently,
depending on the channel. NIR2 was combined with NIR1 in the same calib.json file,
so I made a special case for it in the get_brightness_list function. I'm not sure if this
will be the case for future calib.json files, so please check the structure of the files
and edit the function accordingly.

Another thing to take into consideration is that the recalibration image (.png file)
is currently saved with the name setpoint_recalibration_graph.png. If you run this
section multiple times, it will overwrite the previous file. You can change the name
in the save_setpoint_calibration_graphs function or change the name of the generated
file manually if you want to keep multiple files.
'''

'''
Instructions on how to apply the edge correction to the setpoint-to-wavelength calculations:
(This procedure is also described in the Belgian Meteorite Spectra Calibration document.)

After you have the resulting images from calculate_peak_setpoint_from_lists functions,
you can check the peak spline setpoint from them. This is the new middle setpoint that
I mentioned in the Belgian Meteorite Spectra Calibration document.

You subtract the original middle setpoint from this new middle setpoint to get the
edge correction value. (new setpoint - original setpoint = edge correction)

The peak spline setpoints are then converted to wavelengths by applying the edge correction
to the equation. These equations can be found in this file or in the Belgian Meteorite
Spectra Calibration document.

The new wavelength is then compared with the officially documented vis, nir1, and nir2
lower edge wavelengths (650, 850, 1200). If the differences between these new wavelengths
and the official wavelengths is more than 10 nm, the edge correction value must be used in
the setpoint-to-wavelength equations when deriving the wavelengths. If the difference is
less than 10 nm, the edge correction should be set to 0.

The edge correction values are hardcoded in the spectra calibration file that processes
the spectra. They can be found at the beginning of process_ESA_meteorite_directory function,
and are named vis_edge_correction, nir1_edge_correction, and nir2_edge_correction.
'''

recalibrate_with_calib_files = False

if recalibrate_with_calib_files:
	# VIS
	brightness = get_brightness_list(
		'/home/sysa/HERA/test_data/2025-08-25_wavelength_calibration/acqseq_301_decompressed/meta/calib.json',
		channel='vis'
	)
	print(brightness)
	setpoints = get_setpoint_list(
		'/home/sysa/HERA/test_data/2025-08-25_wavelength_calibration/acqseq_301_decompressed/meta/config.json',
		channel='vis'
	)
	print(setpoints)	
	calculate_peak_setpoint_from_lists(
		setpoints,
		brightness,
		'/home/sysa/HERA/test_data/test_outputs',
		save=True
	)

	# # NIR2 example
	# brightness = get_brightness_list(
	# 	'/home/sysa/HERA/test_data/2025-08-25_wavelength_calibration/acqseq_302_decompressed/meta/calib.json',
	# 	channel='nir2'
	# )
	# print(brightness)
	# setpoints = get_setpoint_list(
	# 	'/home/sysa/HERA/test_data/2025-08-25_wavelength_calibration/acqseq_302_decompressed/meta/config.json',
	# 	channel='nir2'
	# )
	# print(setpoints)	
	# calculate_peak_setpoint_from_lists(
	# 	setpoints,
	# 	brightness,
	# 	'/home/sysa/HERA/test_data/test_outputs',
	# 	save=True
	# )
