import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import time

"""
This file is for processing the ASPECT data to calculate the base
dark background. It has not been cleaned, so expect mess.

The following data structure is expected:

dirPath/             
├── acq_000/                
│   ├── dc_1_exp_000.bin         
│   ├── dc_1_exp_001.bin         
│   └── ...                
│
└── meta/                   
	├── calibration.json   
	├── config.json
	├── telemetry.json     
	└── ...
"""

def check_order(sp, channel):
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
def read_channel(calibPath):
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
def read_config(configPath, channel):
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

def histogram(data, title=None):
	flattened_data = data.flatten()
	plt.figure(figsize=(10, 5))
	plt.hist(flattened_data, bins=500, color='blue', alpha=0.7)
	if title:
		plt.title(title)
	plt.yscale("log")
	plt.ylabel("Frequency (log scale)")
	plt.show()

def compare_histograms(data1, data2, outputFolder=None, filename=None, title=None):
	flattened_data1 = data1.flatten()
	flattened_data2 = data2.flatten()

	# Calculate the histograms and get the maximum frequency for both datasets
	hist1, bins1 = np.histogram(flattened_data1, bins=500)
	hist2, bins2 = np.histogram(flattened_data2, bins=500)

	# Get the maximum y-value from both histograms
	max_y = max(np.max(hist1), np.max(hist2))
	max_x = max(np.max(flattened_data1), np.max(flattened_data2))
	min_x = min(np.min(flattened_data1), np.min(flattened_data2))

	plt.figure(figsize=(15, 7))

	# Create first subplot for the first histogram
	plt.subplot(1, 2, 1)
	plt.hist(flattened_data1, bins=bins1, color='blue', alpha=0.7)
	plt.yscale("log")
	plt.ylim(0.7, max_y)  # Set the same y-axis range
	plt.xlim(min_x-0.09*min_x, max_x+0.09*max_x)  # Set the same x-axis range
	plt.ylabel("Frequency (log scale)")
	plt.title("Histogram of Original Values (cropped 6 pixels from all sides)")
	
	# Create second subplot for the second histogram
	plt.subplot(1, 2, 2)
	plt.hist(flattened_data2, bins=bins2, color='green', alpha=0.7)
	plt.yscale("log")
	plt.ylim(0.7, max_y)  # Set the same y-axis range
	plt.xlim(min_x-0.09*min_x, max_x+0.09*max_x)  # Set the same x-axis range
	plt.ylabel("Frequency (log scale)")
	plt.title("Filtered All Unique Values And Values Over 12000")
	
	if title:
		plt.suptitle(title, fontsize=10, fontweight='bold')
	plt.tight_layout()
	# plt.show()

	save_path = os.path.join(outputFolder, f"{filename}.png")
	plt.savefig(save_path, dpi=300)
	plt.close()

	print(f"Histogram saved to: {save_path}")

def extract_diagnostics(image):
	# Define diagnostic pixel regions
	top = 6#5  # Lines at the top
	bottom = 6#3  # Lines at the bottom
	left = 6#6  # Columns on the left
	right = 6#4  # Columns on the right
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


def average_and_sd(parentDirPath, mainDirectory, outputFolder, make_histograms=False):
	acquisitionPath = os.path.join(parentDirPath, mainDirectory, "acq_000/") # path to acquisitions
	configPath = os.path.join(parentDirPath, mainDirectory, "meta/config.json") # path to config file
	calibPath = os.path.join(parentDirPath, mainDirectory, "meta/calib.json") # path to calib file

	if make_histograms:
		outputFolder = os.path.join(outputFolder, mainDirectory)
		os.makedirs(outputFolder, exist_ok=True)

	#Extract channel
	channel = read_channel(calibPath)
	#Read config files and extract (order, exposuretimes[])
	config = read_config(configPath, channel)

	#Create dimensions based on channel
	if channel == 'VIS':
		height = 1024
		width = 1024
	elif channel == 'NIR1' or channel == 'NIR2':
		height = 518
		width = 648
	
	#Read bin files from folder and short them in the right order
	binFiles = [f for f in os.listdir(acquisitionPath) if f.endswith('.bin')]
	binFiles.sort()

	if channel == 'SWIR':
		raise NotImplementedError("SWIR channel is not supported for average and standard deviation calculation.")

	file_names = []
	mean_values = []
	std_devs = []

	for i, binFile in enumerate(binFiles):
		file_names.append(binFile)
		print(f"\nProcessing file: {mainDirectory}/{binFile}")
		filePath = os.path.join(acquisitionPath, binFile)
		with open(filePath, 'rb') as file:
			BinData = file.read()
			imgArray = np.frombuffer(BinData, dtype=np.uint16)
			imgArray = imgArray.reshape((height, width))
			# print("Mean:", np.mean(imgArray))
			# print("Median:", np.median(imgArray))
			# print("Min:", np.min(imgArray))
			# print("Max:", np.max(imgArray))
			# print("Standard Deviation:", np.std(imgArray))
			# print("Sum:", np.sum(imgArray))
			print("Original Image Shape:", imgArray.shape)
			# histogram(imgArray)
			if channel == 'NIR1' or channel == 'NIR2':
				cleanedImage, diagnostics = extract_diagnostics(imgArray)
				print(f"Extracted diagnostic pixels. New image shape: {cleanedImage.shape}")
			else:
				cleanedImage = imgArray
			# histogram(cleanedImage)
			print("Mean:", np.mean(cleanedImage))
			print("Median:", np.median(cleanedImage))
			print("Min:", np.min(cleanedImage))
			print("Max:", np.max(cleanedImage))
			print("Standard Deviation:", np.std(cleanedImage))
			print("Cleaned Image Shape:", cleanedImage.shape)
			print("flattened shape:", cleanedImage.flatten().shape)

			unique_values, counts = np.unique(cleanedImage, return_counts=True)
			non_unique_values = unique_values[counts > 1]
			filtered_values = cleanedImage[np.isin(cleanedImage, non_unique_values)]
			print("no unique values shape:", filtered_values.shape)
			filtered_values = filtered_values[filtered_values < 12000]
			mean_value = np.mean(filtered_values)
			std_dev = np.std(filtered_values)
			mean_values.append(mean_value)
			std_devs.append(std_dev)
			# histogram(filtered_values)

			title = os.path.join(mainDirectory, binFile)

			if make_histograms:
				compare_histograms(cleanedImage, filtered_values, outputFolder=outputFolder, filename=binFile[:-4], title=title)

			print("Mean:", mean_value)
			print("Standard Deviation:", std_dev)
			print("Min:", np.min(filtered_values))
			print("Max:", np.max(filtered_values))

	return file_names, mean_values, std_devs

def mean_median_sd(parentDirPath, mainDirectory, outputFolder, make_histograms=False):
	acquisitionPath = os.path.join(parentDirPath, mainDirectory, "acq_000/") # path to acquisitions
	configPath = os.path.join(parentDirPath, mainDirectory, "meta/config.json") # path to config file
	calibPath = os.path.join(parentDirPath, mainDirectory, "meta/calib.json") # path to calib file

	if make_histograms:
		outputFolder = os.path.join(outputFolder, mainDirectory)
		os.makedirs(outputFolder, exist_ok=True)

	#Extract channel
	channel = read_channel(calibPath)
	#Read config files and extract (order, exposuretimes[])
	config = read_config(configPath, channel)

	#Create dimensions based on channel
	if channel == 'VIS':
		height = 1024
		width = 1024
	elif channel == 'NIR1' or channel == 'NIR2':
		height = 518
		width = 648
	
	#Read bin files from folder and short them in the right order
	binFiles = [f for f in os.listdir(acquisitionPath) if f.endswith('.bin')]
	binFiles.sort()

	if channel == 'SWIR':
		raise NotImplementedError("SWIR channel is not supported for average and standard deviation calculation.")

	file_names = []
	mean_values = []
	median_values = []
	std_devs = []

	for i, binFile in enumerate(binFiles):
		file_names.append(binFile)
		print(f"\nProcessing file: {mainDirectory}/{binFile}")
		filePath = os.path.join(acquisitionPath, binFile)
		with open(filePath, 'rb') as file:
			BinData = file.read()
			imgArray = np.frombuffer(BinData, dtype=np.uint16)
			imgArray = imgArray.reshape((height, width))
			# print("Mean:", np.mean(imgArray))
			# print("Median:", np.median(imgArray))
			# print("Min:", np.min(imgArray))
			# print("Max:", np.max(imgArray))
			# print("Standard Deviation:", np.std(imgArray))
			# print("Sum:", np.sum(imgArray))
			print("Original Image Shape:", imgArray.shape)
			# histogram(imgArray)
			if channel == 'NIR1' or channel == 'NIR2':
				cleanedImage, diagnostics = extract_diagnostics(imgArray)
				print(f"Extracted diagnostic pixels. New image shape: {cleanedImage.shape}")
			else:
				cleanedImage = imgArray
			# histogram(cleanedImage)
			print("Mean:", np.mean(cleanedImage))
			print("Median:", np.median(cleanedImage))
			print("Min:", np.min(cleanedImage))
			print("Max:", np.max(cleanedImage))
			print("Standard Deviation:", np.std(cleanedImage))
			print("Cleaned Image Shape:", cleanedImage.shape)
			print("flattened shape:", cleanedImage.flatten().shape)

			unique_values, counts = np.unique(cleanedImage, return_counts=True)
			non_unique_values = unique_values[counts > 1]
			filtered_values = cleanedImage[np.isin(cleanedImage, non_unique_values)]
			print("no unique values shape:", filtered_values.shape)
			filtered_values = filtered_values[filtered_values < 12000]
			mean_value = np.mean(filtered_values)
			median_value = np.median(filtered_values)
			std_dev = np.std(filtered_values)
			mean_values.append(mean_value)
			median_values.append(median_value)
			std_devs.append(std_dev)
			# histogram(filtered_values)

			title = os.path.join(mainDirectory, binFile)

			if make_histograms:
				compare_histograms(cleanedImage, filtered_values, outputFolder=outputFolder, filename=binFile[:-4], title=title)

			print("Median:", median_value)
			print("Standard Deviation:", std_dev)
			print("Min:", np.min(filtered_values))
			print("Max:", np.max(filtered_values))

	return file_names, mean_values, median_values, std_devs

def convert_jp2_to_bin(jp2_folder, bin_folder):
	os.makedirs(bin_folder, exist_ok=True)
	jp2_files = [f for f in os.listdir(jp2_folder) if f.endswith('.jp2')]
	for file in jp2_files:
		file_path = os.path.join(jp2_folder, file)
		image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
		binary_data = image.tobytes()
		bin_file_path = os.path.join(bin_folder, file.replace('.jp2', ''))
		with open(bin_file_path, 'wb') as bin_file:
			bin_file.write(binary_data)

def dataframe_average_and_sd(parent_directory, directories_to_process, output_folder, make_histograms=False):
	results = []

	for directory in directories_to_process:
		print(f"Processing: {directory}")
		try:
			file_names, mean_values, std_devs = average_and_sd(
				parent_directory,
				directory,
				output_folder,
				make_histograms=make_histograms
			)
			for file_name, mean_value, std_dev in zip(file_names, mean_values, std_devs):
				results.append({
					"File": f"{directory}/{file_name}",
					"Mean": mean_value,
					"Standard Deviation": std_dev
				})
		except Exception as e:
			print(f"Error processing {directory}: {e}")
			results.append({
				"File": f"error: {e}",
				"Mean": f"error: {e}",
				"Standard Deviation": f"error: {e}"
			})

	# Convert to DataFrame
	df = pd.DataFrame(results)
	return df

def dataframe_mean_median_sd(parent_directory, directories_to_process, output_folder, make_histograms=False):
	results = []

	for directory in directories_to_process:
		print(f"Processing: {directory}")
		try:
			file_names, mean_values, median_values, std_devs = mean_median_sd(
				parent_directory,
				directory,
				output_folder,
				make_histograms=make_histograms
			)
			for file_name, mean_value, median_value, std_dev in zip(file_names, mean_values, median_values, std_devs):
				results.append({
					"File": f"{directory}/{file_name}",
					"Mean": mean_value,
					"Median": median_value,
					"Standard Deviation": std_dev
				})
		except Exception as e:
			print(f"Error processing {directory}: {e}")
			results.append({
				"File": f"error: {e}",
				"Mean": f"error: {e}",
				"Median": f"error: {e}",
				"Standard Deviation": f"error: {e}"
			})

	# Convert to DataFrame
	df = pd.DataFrame(results)
	return df

def save_excel(df, output_folder):
	excel_output_folder = output_folder
	excel_output_file = "mean_median_sd_results.xlsx"
	excel_output_path = os.path.join(excel_output_folder, excel_output_file)
	os.makedirs(excel_output_folder, exist_ok=True)
	df.to_excel(excel_output_path, index=False)
	print(f"\n✅ Results written to Excel: {excel_output_path}")

def compare_bin_stats(parent_directory, directories_to_process, output_folder):
	os.makedirs(output_folder, exist_ok=True)

	for directory in directories_to_process:
		print(f"Processing: {directory}")
		results = []
		file_names, mean_values, median_values, std_devs = mean_median_sd(
			parent_directory,
			directory,
			output_folder,
			make_histograms=True
		)
		for file_name, mean_value, median_value, std_dev in zip(file_names, mean_values, median_values, std_devs):
			results.append({
				"File": f"{directory}/{file_name}",
				"Mean": mean_value,
				"Median": median_value,
				"Standard Deviation": std_dev
			})

		x_indices = list(range(len(file_names)))
		
		# Create the plot
		plt.figure(figsize=(10, 6))
		plt.errorbar(x_indices, mean_values, yerr=std_devs, fmt='o', capsize=5,
					 label='Mean ± STD', color='blue', ecolor='red')
		plt.xticks(x_indices, file_names, rotation=45, ha='right')
		plt.xlabel("Files")
		plt.ylabel("Mean Values (with STD)")
		plt.title(f"{directory}")
		plt.legend()
		plt.tight_layout()
		
		# Save the figure
		output_file = os.path.join(output_folder, f"{directory}.png")
		plt.savefig(output_file)
		plt.close()
		print(f"Graph saved to {output_file}")

def make_average_bins(parent_directory, directories_to_process, output_folder, filters):
	for directory, filter in zip(directories_to_process, filters):
		cleaned_images = []
		print(f"Processing: {directory}")

		acquisitionPath = os.path.join(parent_directory, directory, "acq_000/") # path to acquisitions
		configPath = os.path.join(parent_directory, directory, "meta/config.json") # path to config file
		calibPath = os.path.join(parent_directory, directory, "meta/calib.json") # path to calib file

		#Extract channel
		channel = read_channel(calibPath)
		#Read config files and extract (order, exposuretimes[])
		config = read_config(configPath, channel)

		#Create dimensions based on channel
		if channel == 'VIS':
			height = 1024
			width = 1024
		elif channel == 'NIR1' or channel == 'NIR2':
			height = 518#506
			width = 648#636
		
		#Read bin files from folder and short them in the right order
		binFiles = [f for f in os.listdir(acquisitionPath) if f.endswith('.bin')]
		binFiles.sort()

		print(f"Filtered files: {binFiles[:filter]}")
		binFiles = binFiles[filter:]

		if channel == 'SWIR':
			raise NotImplementedError("SWIR channel is not supported for average and standard deviation calculation.")

		for i, binFile in enumerate(binFiles):
			print(f"\nProcessing file: {directory}/{binFile}")
			filePath = os.path.join(acquisitionPath, binFile)
			with open(filePath, 'rb') as file:
				BinData = file.read()
				imgArray = np.frombuffer(BinData, dtype=np.uint16)
				imgArray = imgArray.reshape((height, width))
				print("Original Image Shape:", imgArray.shape)
				if channel == 'NIR1' or channel == 'NIR2':
					cleanedImage, diagnostics = extract_diagnostics(imgArray)
					print(f"Extracted diagnostic pixels. New image shape: {cleanedImage.shape}")
				else:
					cleanedImage = imgArray
				cleaned_images.append(cleanedImage)

		pixel1 = []
		pixel2 = []
		for image in cleaned_images:
			pixel1.append(image[3][33])
			pixel2.append(image[36][4])
		print("Pixel 1 average", np.mean(pixel1))
		print("Pixel 2 average", np.mean(pixel2))

		image_stack = np.stack(cleaned_images, axis=0)
		print(f"Stacked image shape: {image_stack.shape}")
		average_image = np.mean(image_stack, axis=0)
		
		# Option: Round the averaged image and convert back to the original data type.
		average_image = np.around(average_image).astype(np.uint16)
		print("Averaged image shape:", average_image.shape)

		print(f"pixel1 average: {average_image[3][33]}")
		print(f"pixel2 average: {average_image[36][4]}")
		
		os.makedirs(output_folder, exist_ok=True)
		
		# Create a filename for the averaged image
		output_filename = f"{directory}.bin"
		output_filepath = os.path.join(output_folder, output_filename)
		
		# Write the averaged image to a binary file
		with open(output_filepath, 'wb') as out_file:
			out_file.write(average_image.tobytes())
			
		print(f"Averaged binary image saved to: {output_filepath}")

def subtract_average_bin(parent_directory, directory_to_process, averaged_bin, output_folder):
	acquisitionPath = os.path.join(parent_directory, directory_to_process, "acq_000/") # path to acquisitions
	calibPath = os.path.join(parent_directory, directory_to_process, "meta/calib.json") # path to calib file

	os.makedirs(output_folder, exist_ok=True)

	#Extract channel
	channel = read_channel(calibPath)

	#Create dimensions based on channel
	if channel == 'VIS':
		height = 1024
		width = 1024
		averaged_height = 1024
		averaged_width = 1024
	elif channel == 'NIR1' or channel == 'NIR2':
		height = 518
		width = 648
		averaged_height = 506
		averaged_width = 636

	with open(averaged_bin, 'rb') as file:
		average_data = file.read()
		averagedImage = np.frombuffer(average_data, dtype=np.uint16)
		averagedImage = averagedImage.reshape((averaged_height, averaged_width))
		print("Loaded averaged image shape:", averagedImage.shape)

	binFiles = [f for f in os.listdir(acquisitionPath) if f.endswith('.bin')]
	binFiles.sort()

	for i, binFile in enumerate(binFiles):
		print(f"\nProcessing file: {directory_to_process}/{binFile}")
		filePath = os.path.join(acquisitionPath, binFile)
		with open(filePath, 'rb') as file:
			BinData = file.read()
			imgArray = np.frombuffer(BinData, dtype=np.uint16)
			imgArray = imgArray.reshape((height, width))
			print("Original Image Shape:", imgArray.shape)
			if channel == 'NIR1' or channel == 'NIR2':
				cleanedImage, diagnostics = extract_diagnostics(imgArray)
				print(f"Extracted diagnostic pixels. New image shape: {cleanedImage.shape}")
			else:
				cleanedImage = imgArray

			subtractedImage = cleanedImage.astype(np.int32) - averagedImage.astype(np.int32)

			# Optionally, inspect the range of values to ensure they are within np.int16 limits
			min_val, max_val = subtractedImage.min(), subtractedImage.max()
			print("Subtracted image pixel value range:", min_val, "to", max_val)

			subtractedImage_int16 = subtractedImage.astype(np.int16)
			print("Subtracted image head (after conversion):\n", subtractedImage_int16[:5])

			# Save the subtracted image
			outputPath = os.path.join(output_folder, directory_to_process, binFile)
			os.makedirs(os.path.dirname(outputPath), exist_ok=True)
			with open(outputPath, 'wb') as outFile:
				outFile.write(subtractedImage_int16.tobytes())
			print(f"Saved subtracted image to: {outputPath}")

def list_folders(path):
	return sorted([
		name for name in os.listdir(path)
		if os.path.isdir(os.path.join(path, name))
	])

# ASPECT_TESTS_25_7 folder =========================================================================================================================================================================================

ASPECT_TESTS_25_7_directories_to_process = [
	"2024-07-25_11-37-22_vis_h_dark_vis_2500",
	"2024-07-25_11-41-57_vis_h_dark_vis_1875",
	"2024-07-25_11-46-48_vis_h_dark_vis_1250",
	"2024-07-25_13-51-37_vis_h_dark_vis_2500",
	"2024-07-25_13-56-10_vis_h_dark_vis_1875",
	"2024-07-25_13-59-32_vis_h_dark_vis_1250",
	"2024-07-25_14-12-05_nir1_h_dark_nir1_10000",
	"2024-07-25_14-13-52_nir1_h_dark_nir1_7500",
	"2024-07-25_14-15-24_nir1_h_dark_nir1_5000",
	"2024-07-25_15-06-52_nir1_h_dark_nir1_10000",
	"2024-07-25_15-08-30_nir1_h_dark_nir1_7500",
	"2024-07-25_15-10-26_nir1_h_dark_nir1_5000",
	"2024-07-25_15-12-33_nir2_h_dark_nir2_10000",
	"2024-07-25_15-14-57_nir2_h_dark_nir2_7500",
	"2024-07-25_15-16-36_nir2_h_dark_nir2_5000",
	"2024-07-25_16-23-38_nir2_h_dark_nir2_10000",
	"2024-07-25_16-25-22_nir2_h_dark_nir2_7500",
	"2024-07-25_16-28-11_nir2_h_dark_nir2_5000"
]

ASPECT_TESTS_25_7_filters = [
	0,
	0,
	0,
	0,
	0,
	0,
	2,
	1,
	1,
	1,
	1,
	1,
	2,
	1,
	1,
	1,
	1,
	1
]

in_flight_dark_250225_directories_to_process = [
	"acqseq_104",
	"acqseq_105",
	"acqseq_106"
]

ASPECT_TESTS_25_7_folder = "/home/sysa/HERA/test_data/ASPECT_TESTS_25.7"
ASPECT_TESTS_25_7_output_folder = "/home/sysa/HERA/test_data/test_outputs/ASPECT_TESTS_25.7_test"
in_flight_dark_250225_folder = "/home/sysa/HERA/test_data/in-flight_dark_250225"
in_flight_dark_250225_output_folder = "/home/sysa/HERA/test_data/test_outputs/in-flight_dark_250225"

do_conversions_from_jp2_to_bin = False
if do_conversions_from_jp2_to_bin:
	for directory in in_flight_dark_250225_directories_to_process:
		convert_jp2_to_bin(
			os.path.join(in_flight_dark_250225_folder, directory, "acq_000"),
			os.path.join(in_flight_dark_250225_folder, directory, "acq_000")
		)
	print("✅ Conversion from .jp2 to .bin completed.")

process_ASPECT_TESTS_25_7 = False
if process_ASPECT_TESTS_25_7:
	df = dataframe_mean_median_sd(
		ASPECT_TESTS_25_7_folder,
		ASPECT_TESTS_25_7_directories_to_process,
		ASPECT_TESTS_25_7_output_folder,
		make_histograms=True
	)
	save_excel(df, ASPECT_TESTS_25_7_output_folder)

process_in_flight_dark_250225 = False
if process_in_flight_dark_250225:
	df = dataframe_mean_median_sd(
		in_flight_dark_250225_folder,
		in_flight_dark_250225_directories_to_process,
		in_flight_dark_250225_output_folder,
		make_histograms=True
	)
	save_excel(df, in_flight_dark_250225_output_folder)

compare_bins_ASPECT_TESTS_25_7 = False
if compare_bins_ASPECT_TESTS_25_7:
	compare_bin_stats(
		ASPECT_TESTS_25_7_folder,
		ASPECT_TESTS_25_7_directories_to_process,
		"/home/sysa/HERA/test_data/test_outputs/ASPECT_TESTS_25.7_bin_comparisons"
	)

make_average_bins_ASPECT_TESTS_25_7 = False
if make_average_bins_ASPECT_TESTS_25_7:
	make_average_bins(
		ASPECT_TESTS_25_7_folder,
		ASPECT_TESTS_25_7_directories_to_process,
		"/home/sysa/HERA/test_data/test_outputs/ASPECT_TESTS_25.7_averages",
		filters=ASPECT_TESTS_25_7_filters
	)

compare_bins_in_flight_dark_250225 = False
if compare_bins_in_flight_dark_250225:
	compare_bin_stats(
		in_flight_dark_250225_folder,
		in_flight_dark_250225_directories_to_process,
		"/home/sysa/HERA/test_data/test_outputs/in-flight_dark_250225_bin_comparisons"
	)

ASPECT_TESTS_25_7_to_be_subtracted = [
    '2024-07-25_11-59-06_vis_h_vis_lo_600w_2500',#
    '2024-07-25_12-04-30_vis_h_vis_ho_600w_2500',#
    '2024-07-25_12-37-11_vis_h_vis_lo_600w_1875',#
    '2024-07-25_12-46-50_vis_h_vis_ho_600w_1875',#
    '2024-07-25_12-55-01_vis_h_vis_lo_600w_1250',#
    '2024-07-25_13-01-06_vis_h_vis_ho_600w_1250',#
    # '2024-07-25_13-07-40_vis_h_vis_lo_600w_625',
    # '2024-07-25_13-13-49_vis_h_vis_ho_600w_625',
    '2024-07-25_13-22-13_vis_h_vis_lo_400w_2500',#
    '2024-07-25_13-28-18_vis_h_vis_ho_400w_2500',#
    '2024-07-25_13-35-53_vis_h_vis_lo_200w_2500',#
    '2024-07-25_13-42-25_vis_h_vis_ho_200w_2500',#
    '2024-07-25_14-20-42_nir1_h_nir1_lo_600w_10000',#
    '2024-07-25_14-23-40_nir1_h_nir1_ho_600w_10000',#
    '2024-07-25_14-27-03_nir1_h_nir1_lo_600w_7500',#
    '2024-07-25_14-30-18_nir1_h_nir1_ho_600w_7500',#
    '2024-07-25_14-34-04_nir1_h_nir1_lo_600w_5000',#
    '2024-07-25_14-37-12_nir1_h_nir1_ho_600w_5000',#
    # '2024-07-25_14-40-51_nir1_h_nir1_lo_600w_2500',
    # '2024-07-25_14-43-53_nir1_h_nir1_ho_600w_2500',
    '2024-07-25_14-48-27_nir1_h_nir1_lo_400w_10000',#
    '2024-07-25_14-52-39_nir1_h_nir1_ho_400w_10000',#
    '2024-07-25_14-57-21_nir1_h_nir1_lo_200w_10000',#
    '2024-07-25_15-00-41_nir1_h_nir1_ho_200w_10000',#
    '2024-07-25_15-19-38_nir2_h_nir2_lo_600w_10000',#
    '2024-07-25_15-25-02_nir2_h_nir2_ho_600w_10000',#
    '2024-07-25_15-36-06_nir2_h_nir2_lo_600w_7500',#
    '2024-07-25_15-41-18_nir2_h_nir2_ho_600w_7500',#
    '2024-07-25_15-45-37_nir2_h_nir2_lo_600w_5000',#
    '2024-07-25_15-50-56_nir2_h_nir2_ho_600w_5000',#
    # '2024-07-25_15-54-56_nir2_h_nir2_lo_600w_2500',
    # '2024-07-25_15-58-35_nir2_h_nir2_ho_600w_2500',
    '2024-07-25_16-05-19_nir2_h_nir2_lo_400w_10000',#
    '2024-07-25_16-08-47_nir2_h_nir2_ho_400w_10000',#
    '2024-07-25_16-14-10_nir2_h_nir2_lo_200w_10000',#
    '2024-07-25_16-18-11_nir2_h_nir2_ho_200w_10000',#
]

ASPECT_TESTS_25_7_averages_dark_before = [
    '2024-07-25_11-37-22_vis_h_dark_vis_2500',#
    '2024-07-25_11-37-22_vis_h_dark_vis_2500',#
    '2024-07-25_11-41-57_vis_h_dark_vis_1875',#
    '2024-07-25_11-41-57_vis_h_dark_vis_1875',#
    '2024-07-25_11-46-48_vis_h_dark_vis_1250',#
    '2024-07-25_11-46-48_vis_h_dark_vis_1250',#
    '2024-07-25_11-37-22_vis_h_dark_vis_2500',#
    '2024-07-25_11-37-22_vis_h_dark_vis_2500',#
    '2024-07-25_11-37-22_vis_h_dark_vis_2500',#
    '2024-07-25_11-37-22_vis_h_dark_vis_2500',#
    '2024-07-25_14-12-05_nir1_h_dark_nir1_10000',#
    '2024-07-25_14-12-05_nir1_h_dark_nir1_10000',#
    '2024-07-25_14-13-52_nir1_h_dark_nir1_7500',#
    '2024-07-25_14-13-52_nir1_h_dark_nir1_7500',#
    '2024-07-25_14-15-24_nir1_h_dark_nir1_5000',#
    '2024-07-25_14-15-24_nir1_h_dark_nir1_5000',#
    '2024-07-25_14-12-05_nir1_h_dark_nir1_10000',#
    '2024-07-25_14-12-05_nir1_h_dark_nir1_10000',#
    '2024-07-25_14-12-05_nir1_h_dark_nir1_10000',#
    '2024-07-25_14-12-05_nir1_h_dark_nir1_10000',#
    '2024-07-25_15-12-33_nir2_h_dark_nir2_10000',#
    '2024-07-25_15-12-33_nir2_h_dark_nir2_10000',#
    '2024-07-25_15-14-57_nir2_h_dark_nir2_7500',#
    '2024-07-25_15-14-57_nir2_h_dark_nir2_7500',#
    '2024-07-25_15-16-36_nir2_h_dark_nir2_5000',#
    '2024-07-25_15-16-36_nir2_h_dark_nir2_5000',#
    '2024-07-25_15-12-33_nir2_h_dark_nir2_10000',#
    '2024-07-25_15-12-33_nir2_h_dark_nir2_10000',#
    '2024-07-25_15-12-33_nir2_h_dark_nir2_10000',#
    '2024-07-25_15-12-33_nir2_h_dark_nir2_10000',#
]

ASPECT_TESTS_25_7_averages_dark_after = [
    '2024-07-25_13-51-37_vis_h_dark_vis_2500',#
    '2024-07-25_13-51-37_vis_h_dark_vis_2500',#
    '2024-07-25_13-56-10_vis_h_dark_vis_1875',#
    '2024-07-25_13-56-10_vis_h_dark_vis_1875',#
    '2024-07-25_13-59-32_vis_h_dark_vis_1250',#
    '2024-07-25_13-59-32_vis_h_dark_vis_1250',#
    '2024-07-25_13-51-37_vis_h_dark_vis_2500',#
    '2024-07-25_13-51-37_vis_h_dark_vis_2500',#
    '2024-07-25_13-51-37_vis_h_dark_vis_2500',#
    '2024-07-25_13-51-37_vis_h_dark_vis_2500',#
    '2024-07-25_15-06-52_nir1_h_dark_nir1_10000',#
    '2024-07-25_15-06-52_nir1_h_dark_nir1_10000',#
    '2024-07-25_15-08-30_nir1_h_dark_nir1_7500',#
    '2024-07-25_15-08-30_nir1_h_dark_nir1_7500',#
    '2024-07-25_15-10-26_nir1_h_dark_nir1_5000',#
    '2024-07-25_15-10-26_nir1_h_dark_nir1_5000',#
    '2024-07-25_15-06-52_nir1_h_dark_nir1_10000',#
    '2024-07-25_15-06-52_nir1_h_dark_nir1_10000',#
    '2024-07-25_15-06-52_nir1_h_dark_nir1_10000',#
    '2024-07-25_15-06-52_nir1_h_dark_nir1_10000',#
    '2024-07-25_16-23-38_nir2_h_dark_nir2_10000',#
    '2024-07-25_16-23-38_nir2_h_dark_nir2_10000',#
    '2024-07-25_16-25-22_nir2_h_dark_nir2_7500',#
    '2024-07-25_16-25-22_nir2_h_dark_nir2_7500',#
    '2024-07-25_16-28-11_nir2_h_dark_nir2_5000',#
    '2024-07-25_16-28-11_nir2_h_dark_nir2_5000',#
    '2024-07-25_16-23-38_nir2_h_dark_nir2_10000',#
    '2024-07-25_16-23-38_nir2_h_dark_nir2_10000',#
    '2024-07-25_16-23-38_nir2_h_dark_nir2_10000',#
    '2024-07-25_16-23-38_nir2_h_dark_nir2_10000',#
]

subtract_average_bins_ASPECT_TESTS_25_7 = False
if subtract_average_bins_ASPECT_TESTS_25_7:
	for to_be_subtracted, average_before, average_after in zip(
		ASPECT_TESTS_25_7_to_be_subtracted,
		ASPECT_TESTS_25_7_averages_dark_before,
		ASPECT_TESTS_25_7_averages_dark_after
	):		
		subtract_average_bin(
			ASPECT_TESTS_25_7_folder,
			to_be_subtracted,
			f"/home/sysa/HERA/test_data/test_outputs/ASPECT_TESTS_25.7_averages/{average_before}.bin",
			"/home/sysa/HERA/test_data/test_outputs/ASPECT_TESTS_25.7_dark_before-subtracted"
		)
		subtract_average_bin(
			ASPECT_TESTS_25_7_folder,
			to_be_subtracted,
			f"/home/sysa/HERA/test_data/test_outputs/ASPECT_TESTS_25.7_averages/{average_after}.bin",
			"/home/sysa/HERA/test_data/test_outputs/ASPECT_TESTS_25.7_dark_after-subtracted"
		)

# ESA meteorites folder =========================================================================================================================================================================================

ESA_meteorites_folder = '/home/sysa/HERA/ESA meteorites'

ESA_meteorites_directories_to_process = [
	'2024-08-29_15-55-34_nir2_dark_nir2_10000',
	'2024-08-29_15-57-12_nir2_dark_nir2_15000',
	'2024-08-29_15-58-38_nir2_dark_nir2_25000',
	'2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	'2024-08-30_12-57-11_nir1_nir1_dark_l_5000',
	'2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	'2024-08-30_15-51-17_vis_vis_dark_2800',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
]

compare_bins_ESA_meteorites = False
if compare_bins_ESA_meteorites:
	compare_bin_stats(
		ESA_meteorites_folder,
		ESA_meteorites_directories_to_process,
		"/home/sysa/HERA/test_data/test_outputs/test_ESA_meteorites_dark_bin_comparisons"
	)

ESA_meteorites_dark_filters = [ # Paired with ESA_meteorites_directories_to_process!! Acquired from analysing compare_bin_stats outputs
	1,
	1,
	1,
	1,
	1,
	1,
	0,
	0
]

make_average_bins_ESA_meteorites = False
if make_average_bins_ESA_meteorites:
	make_average_bins(
		ESA_meteorites_folder,
		ESA_meteorites_directories_to_process,
		"/home/sysa/HERA/test_data/test_outputs/ESA_meteorites_dark_averages",
		filters=ESA_meteorites_dark_filters
	)

# print(list_folders('/home/sysa/HERA/ESA meteorites'))

ESA_meteorites_folders_to_be_subtracted = [ # Match with ESA_meteorites_dark_averages
	# '2024-08-29_13-31-17_nir2_20_alb_target',
	'2024-08-29_13-32-44_nir2_20_alb_target_tint25000',
	'2024-08-29_14-36-57_nir2_20_nir2_25000',
	'2024-08-29_14-43-36_nir2_Kilabo_nir2_25000',
	# '2024-08-29_14-47-38_nir2_Kilabo_nir2_20000',
	'2024-08-29_14-51-53_nir2_Kilabo_nir2_15000',
	'2024-08-29_14-55-51_nir2_Kilabo_nir2_10000',
	'2024-08-29_15-03-03_nir2_Iran0006_nir2_15000',
	'2024-08-29_15-07-52_nir2_Iran0006_nir2_15000',
	'2024-08-29_15-12-43_nir2_ElHammami_nir2_15000',
	'2024-08-29_15-21-37_nir2_Allende_nir2_25000',
	'2024-08-29_15-29-28_nir2_SaU001_nir2_25000',
	'2024-08-29_15-33-06_nir2_20_nir2_15000',
	'2024-08-29_15-35-50_nir2_20_nir2_10000',
	'2024-08-29_15-41-19_nir2_ElHammamiB_nir2_15000',
	'2024-08-29_15-46-37_nir2_ElHammamiB_nir2_10000',
	# '2024-08-29_15-50-26_nir2_edge_l_nir2_10000',
	# '2024-08-29_15-55-34_nir2_dark_nir2_10000',
	# '2024-08-29_15-57-12_nir2_dark_nir2_15000',
	# '2024-08-29_15-58-38_nir2_dark_nir2_25000',
	# '2024-08-30_11-12-41_nir2_psk',
	# '2024-08-30_11-14-25_nir2_psk',
	# '2024-08-30_11-15-33_nir2_psk',
	# '2024-08-30_11-15-57_nir2_psk',
	# '2024-08-30_11-17-07_nir2_psk',
	# '2024-08-30_11-17-26_nir2_psk',
	# '2024-08-30_11-18-07_nir2_psk',
	# '2024-08-30_11-18-40_nir2_psk',
	# '2024-08-30_11-39-06_nir1_nir2_test',
	# '2024-08-30_11-43-11_nir1_nir1_test',
	# '2024-08-30_11-46-42_nir1_nir1_test',
	'2024-08-30_11-51-32_nir1_20_nir1_15000',
	# '2024-08-30_11-55-27_nir1_20_nir1_2000',
	# '2024-08-30_12-01-18_nir1_Kilaboo_nir1_2000',
	# '2024-08-30_12-04-52_nir1_Kilaboo_nir1_2000',
	# '2024-08-30_12-07-33_nir1_Kilaboo_nir1_2000',
	# '2024-08-30_12-10-39_nir1_Kilaboo_nir1_4000',
	'2024-08-30_12-13-49_nir1_Kilaboo_nir1_5000',
	'2024-08-30_12-18-38_nir1_Iran0006_nir1_5000',
	'2024-08-30_12-21-27_nir1_Iran0006_nir1_5000',
	'2024-08-30_12-24-32_nir1_Iran0006_nir1_5000',
	'2024-08-30_12-27-25_nir1_ElHammami_nir1_5000',
	'2024-08-30_12-31-12_nir1_Allende_nir1_15000',
	'2024-08-30_12-34-28_nir1_SaU001_nir1_15000',
	'2024-08-30_12-36-59_nir1_SaU001_nir1_15000',
	'2024-08-30_12-40-38_nir1_ElHammamiB_nir1_5000',
	'2024-08-30_12-43-27_nir1_20_nir1_5000',
	# '2024-08-30_12-46-58_nir1_nir1_edge_l_20000',
	# '2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	# '2024-08-30_12-57-11_nir1_nir1_dark_l_5000',
	# '2024-08-30_12-58-14_nir1_nir1_dark_l_5000',
	# '2024-08-30_14-22-11_vis_20_vis_2500',
	# '2024-08-30_14-29-18_vis_20_vis_4000',
	# '2024-08-30_14-35-40_vis_20_vis_3000',
	'2024-08-30_14-41-10_vis_20_vis_2800',
	'2024-08-30_14-47-39_vis_20_vis_1400',
	'2024-08-30_14-54-01_vis_Kilaboo_vis_1400',
	'2024-08-30_15-00-42_vis_Iran0006_vis_1400',
	'2024-08-30_15-06-16_vis_ElHammami_vis_1400',
	'2024-08-30_15-11-36_vis_ElHammami_vis_1400',
	'2024-08-30_15-17-13_vis_Allende_vis_2800',
	'2024-08-30_15-22-37_vis_SaU001_vis_2800',
	'2024-08-30_15-27-59_vis_SaU001_vis_1400',
	'2024-08-30_15-33-30_vis_ElHammamiB_vis_1400',
	# '2024-08-30_15-41-29_vis_vis_edge_h_2800',
	# '2024-08-30_15-51-17_vis_vis_dark_2800',
	# '2024-08-30_15-54-10_vis_vis_dark_1400'
]

ESA_meteorites_dark_averages = [ # Matched with ESA_meteorites_directories_to_process
	'2024-08-29_15-58-38_nir2_dark_nir2_25000',
	'2024-08-29_15-58-38_nir2_dark_nir2_25000',
	'2024-08-29_15-58-38_nir2_dark_nir2_25000',
	'2024-08-29_15-57-12_nir2_dark_nir2_15000',
	'2024-08-29_15-55-34_nir2_dark_nir2_10000',
	'2024-08-29_15-57-12_nir2_dark_nir2_15000',
	'2024-08-29_15-57-12_nir2_dark_nir2_15000',
	'2024-08-29_15-57-12_nir2_dark_nir2_15000',
	'2024-08-29_15-58-38_nir2_dark_nir2_25000',
	'2024-08-29_15-58-38_nir2_dark_nir2_25000',
	'2024-08-29_15-57-12_nir2_dark_nir2_15000',
	'2024-08-29_15-55-34_nir2_dark_nir2_10000',
	'2024-08-29_15-57-12_nir2_dark_nir2_15000',
	'2024-08-29_15-55-34_nir2_dark_nir2_10000',
	'2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	'2024-08-30_12-57-11_nir1_nir1_dark_l_5000',
	'2024-08-30_12-57-11_nir1_nir1_dark_l_5000',
	'2024-08-30_12-57-11_nir1_nir1_dark_l_5000',
	'2024-08-30_12-57-11_nir1_nir1_dark_l_5000',
	'2024-08-30_12-57-11_nir1_nir1_dark_l_5000',
	'2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	'2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	'2024-08-30_12-52-23_nir1_nir1_dark_l_15000',
	'2024-08-30_12-57-11_nir1_nir1_dark_l_5000',
	'2024-08-30_12-57-11_nir1_nir1_dark_l_5000',
	'2024-08-30_15-51-17_vis_vis_dark_2800',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-51-17_vis_vis_dark_2800',
	'2024-08-30_15-51-17_vis_vis_dark_2800',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
	'2024-08-30_15-54-10_vis_vis_dark_1400',
]

subtract_average_bins_ESA_meteorites = False
if subtract_average_bins_ESA_meteorites:
	for to_be_subtracted, average_dark in zip(
		ESA_meteorites_folders_to_be_subtracted,
		ESA_meteorites_dark_averages
	):
		subtract_average_bin(
			ESA_meteorites_folder,
			to_be_subtracted,
			f"/home/sysa/HERA/test_data/test_outputs/ESA_meteorites_dark_averages/{average_dark}.bin",
			"/home/sysa/HERA/test_data/test_outputs/ESA_meteorites_dark_subtracted"
		)

stats_ESA_meteorites = False
if stats_ESA_meteorites:
	timestamp = time.time()
	local_time = time.localtime(timestamp)
	formatted_time = time.strftime("%Y-%m-%dT%H:%M:%S", local_time)
	ESA_meteorites_stats_output_folder = f"/home/sysa/HERA/test_data/test_outputs/{formatted_time}_ESA_meteorites_stats_test"
	df = dataframe_mean_median_sd(
		ESA_meteorites_folder,
		ESA_meteorites_directories_to_process,
		ESA_meteorites_stats_output_folder,
		make_histograms=True
	)
	save_excel(df, ESA_meteorites_stats_output_folder)
