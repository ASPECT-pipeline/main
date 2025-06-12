import os
import numpy as np
from astropy.io import fits
import math
import matplotlib.pyplot as plt

coefficient_folder = os.path.join(os.getcwd(), 'test_data/matlab_coefficients')
vis_coef_path = os.path.join(coefficient_folder, 'vis-conversion.dat')
nir_coef_path = os.path.join(coefficient_folder, 'nir-conversion.dat')
simulated_cube = os.path.join(os.getcwd(), 'test_data/test_outputs/simulated_full_datacube.fits')


# Funtion to get coefficient to convert matlab pixel values into reflectances for specific wavelengths
def get_coef(path:str):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2: 
                try: 
                    wl = float(parts[0])
                    coef = float(parts[1])
                    data[wl] = coef
                except ValueError:
                    print(f'error on line: {line}') 
    return data


# Function to convert to reflectanses

def get_reflectances(spectra, wavelengths):
    vis_coef = get_coef(vis_coef_path)
    nir_coef = get_coef(nir_coef_path)
    all_coef = {**vis_coef, **nir_coef}

    reflectances = np.zeros(len(spectra), dtype=float)
    for i, s in enumerate(spectra):

        v = s / all_coef[wavelengths[i]]
        if i < 10:
            v = v / 0.02
            v += 0.008 # fixed value to reduce some error in the coefficients
        else:
            v = v / 0.04
        reflectances[i] = v
    return reflectances

def readFits(fitsPath, visualise:bool = False):
    name = os.path.splitext(os.path.basename(fitsPath))[0]
    print(f'Reading file: {name}')
    # Open FITS file using astropy
    with fits.open(fitsPath) as hdul:
        print(f'info:\n {hdul.info()}') # Print the info of the hdul
        total_size = hdul._file.size # Total size of the file
        print(f"Total File Size: {total_size} bytes")
        num_extensions = len(hdul) - 1 # Number of extensions
        print(f"Number of Extensions: {num_extensions}")


        # Iterate through each extension
        for i, hdu in enumerate(hdul):
            print('')
            print(f'\nHDU number: {i}') # Extension number
            header = hdu.header
            print(repr(header))
            data = hdu.data
            

            if isinstance(hdu, fits.ImageHDU):
                naxis1 = header.get('NAXIS1')  # Width
                naxis2 = header.get('NAXIS2')  # Height
                naxis3 = header.get('NAXIS3')  # Number of images
                print(f'Data cube shape: {data.shape}')

                print('Image Data:')
                print()

                # for i, image in enumerate(data):
                #     print(f'frame {i}')
                #     plt.imshow(image, cmap='gray')
                #     plt.show()


                # Determine grid layout
                max_images_per_row = 5
                rows = math.ceil(naxis3 / max_images_per_row)
                cols = min(max_images_per_row, naxis3)

                if visualise:
                # Display all 2D images
                    plt.figure(figsize=(cols * 4, rows * 4))
                    plt.suptitle('2D Slices from Data Cube', fontsize=16)
                    for i in range(naxis3):
                        plt.subplot(rows, cols, i + 1)
                        plt.imshow(data[i, :, :], cmap='gray')
                        plt.title(f'Slice {i + 1}')
                        plt.axis('off')

                    plt.tight_layout()
                    plt.show()

# readFits(simulated_cube)