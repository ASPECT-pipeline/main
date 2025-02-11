import spiceypy as spice
import pyperclip
import io
import sys

# Work in progress

"""
This is a file for using HERA SPICE kernels.
To use this file, you need to have HERA SPICE kernel dataset installed.
You may need to write the correct path to the kernel folder specified in metakernel.
The SPICE dataset has too large file sizes to be uploaded into github.
Refer to the readme files in SPICE dataset for more information about SPICE kernels.
"""

"""
HERA SPICE kernel dataset: https://s2e2.cosmos.esa.int/bitbucket/projects/SPICE_KERNELS/repos/hera/browse
SpiceyPy docs: https://spiceypy.readthedocs.io/en/stable/documentation.html#
WebGeocalc: http://spice.esac.esa.int/webgeocalc/#NewCalculation
"""

# Add metakernel path
metakernel_path = "" # for example ...SPICE/HERA/kernels/mk/hera_ops.tm
	  
def compute_distance(position):
	return (position[0]**2 + position[1]**2 + position[2]**2) ** 0.5

def spice_example_query(metakernel_path):
	# Load SPICE kernels
	spice.furnsh(metakernel_path)

	# Specify target, observer, and time
	target = "DIDYMOS_BARYCENTER"
	observer = "HERA"
	frame = "J2000"
	utc_time = "2025-12-27T00:00:00"

	# Convert time to ET
	et = spice.str2et(utc_time)

	# Get state vector (position and velocity) of the target relative to the observer
	state, light_time = spice.spkezr(target, et, frame, "NONE", observer)

	# Extract position and velocity
	position = state[:3]  # x, y, z coordinates (km)
	velocity = state[3:]  # vx, vy, vz components (km/s)

	distance = compute_distance(position)

	# Print results
	print(f"UTC Time: {utc_time}")
	print(f"Position of {target} relative to {observer}:")
	print(f"  X: {position[0]:.3f} km")
	print(f"  Y: {position[1]:.3f} km")
	print(f"  Z: {position[2]:.3f} km")
	print(f"Velocity of {target} relative to {observer}:")
	print(f"  VX: {velocity[0]:.3f} km/s")
	print(f"  VY: {velocity[1]:.3f} km/s")
	print(f"  VZ: {velocity[2]:.3f} km/s")
	print(f"Distance between {observer} and {target}: {distance:.3f} km")

	# Unload SPICE kernels
	spice.kclear()

# spice_example_query()

def explore_kernel_info():
	"""
	Comprehensive function to explore different types of information available in SPICE kernels
	"""
	def print_section(title):
		print(f"\n{'='*20} {title} {'='*20}")
	
	def print_kernel_coverage(kernel_id):
		"""Print time coverage for specified kernel"""
		try:
			# Get first and last times covered by the kernel
			window = spice.getwni(kernel_id)
			if window:
				start_et = window[0]
				end_et = window[-1]
				print(f"Coverage start: {spice.etcal(start_et)}")
				print(f"Coverage end: {spice.etcal(end_et)}")
		except:
			print("No time windows available for this kernel")

	# Print basic kernel information
	print_section("Basic Kernel Information")
	kernel_count = spice.ktotal("ALL")
	print(f"Total Kernels Loaded: {kernel_count}")
	
	for i in range(kernel_count):
		filetype, source, handle, found = spice.kdata(i, "ALL")
		print(f"\nKernel {i+1}:")
		print(f"  File Type: {filetype}")
		print(f"  Source: {source}")
		print(f"  Handle: {handle}")
		
		# Get kernel type-specific information
		if filetype == 'SPK':
			print_section(f"SPK (Ephemeris) Kernel Details - {source}")
			# List bodies in the kernel
			ids = spice.spkobj(source)
			print("Bodies in kernel:")
			for body_id in ids:
				try:
					body_name = spice.bodc2n(body_id)
					print(f"  Body ID {body_id}: {body_name}")
				except:
					print(f"  Body ID {body_id}: [Name not found]")
			print_kernel_coverage(handle)
			
		elif filetype == 'PCK':
			print_section(f"PCK (Planetary Constants) Kernel Details - {source}")
			# List available constants
			try:
				variables = spice.gnpool("BODY*_", 0, 100)
				print("Available constants:")
				for var in variables:
					print(f"  {var}")
			except:
				print("No planetary constants found")
				
		elif filetype == 'IK':
			print_section(f"IK (Instrument) Kernel Details - {source}")
			# List instrument IDs
			try:
				ids = spice.kinfo(handle, "IK", 100)
				print("Instruments in kernel:")
				for inst_id in ids:
					try:
						inst_name = spice.bodc2n(inst_id)
						print(f"  Instrument ID {inst_id}: {inst_name}")
					except:
						print(f"  Instrument ID {inst_id}")
			except:
				print("No instrument information found")
				
		elif filetype == 'CK':
			print_section(f"CK (C-matrix/Pointing) Kernel Details - {source}")
			# List frames
			try:
				ids = spice.ckobj(source)
				print("Frames in kernel:")
				for frame_id in ids:
					try:
						frame_name = spice.frmnam(frame_id)
						print(f"  Frame ID {frame_id}: {frame_name}")
					except:
						print(f"  Frame ID {frame_id}")
				print_kernel_coverage(handle)
			except:
				print("No frame information found")

	# Print available reference frames
	print_section("Available Reference Frames")
	try:
		frame_count = spice.countfr()
		print(f"Total number of frames: {frame_count}")
		print("\nSample of available frames:")
		for i in range(min(10, frame_count)):
			frame_id = spice.frinfo(i)[0]
			try:
				frame_name = spice.frmnam(frame_id)
				print(f"  Frame ID {frame_id}: {frame_name}")
			except:
				continue
	except:
		print("Unable to retrieve frame information")

def list_available_spice_info(metakernel_path: str):
	# Load your kernels first
	spice.furnsh(metakernel_path)
	
	try:
		explore_kernel_info()
	finally:
		# Always clean up
		spice.kclear()
		
# list_available_spice_info(metakernel_path) # Lists file paths

def print_pool_variables(metakernel_path: str):
	"""
	Prints all kernel pool variables and their values.
	
	It uses:
	  - gnpool() to get the list of variable names in the pool,
	  - gdpool() to attempt to get numeric (double precision) values, and
	  - gcpool() to get character values if numeric ones are not found.
	  
	Note: Make sure that one or more kernels have been loaded (e.g. via furnsh)
	before calling this function.
	"""
	copy_to_clipboard = False
	spice.furnsh(metakernel_path)
	# Retrieve all kernel pool variable names (allowing up to 1000 values per variable)
	varnames = spice.gnpool("*ver*", 0, 1000)
	print(len(varnames))
	if not varnames:
		print("No kernel pool variables found.")
		return
	
	if copy_to_clipboard:
		output = io.StringIO()  # Create a buffer to capture output
		sys.stdout = output  # Redirect print output

	print("Kernel Pool Variables:")
	for name in varnames:
		printed = False
		# Try to retrieve numeric values for the variable.
		try:
			# gdpool returns a list of double precision values
			dvals = spice.gdpool(name, 0, 1000)
			if dvals:
				print(f"{name} (numeric): {dvals}")
				printed = True
				try:
					cvals = spice.gcpool(name, 0, 1000)
					if cvals:
						print(f"{name} (character): {cvals}")
				except Exception:
					pass
		except Exception:
			# If the variable does not have numeric values, ignore the exception.
			pass

		# If no numeric values were found, try to get character values.
		if not printed:
			try:
				cvals = spice.gcpool(name, 0, 1000)
				if cvals:
					print(f"{name} (character): {cvals}")
				else:
					print(f"{name}: (no values)")
			except Exception as e:
				print(f"{name}: error retrieving values ({e})")
	spice.kclear()

	if copy_to_clipboard:
		sys.stdout = sys.__stdout__  # Restore normal print behavior
		output_text = output.getvalue()  # Get captured text
		pyperclip.copy(output_text)  # Copy text to clipboard
		print("Output copied to clipboard!")
				
# print_pool_variables(metakernel_path)

def query_spacecraft_position_vectors(
		metakernel_path: str,
		target: str = 'HERA',
		utc_time: str = '2025-12-27T00:00:00',
		frame: str = 'J2000',
		observer: str = 'EARTH',
		test: bool = False
	):
	"""
	Query the position vectors of spacecrafts in a specified time frame.
	"""
	spice.furnsh(metakernel_path)
	
	et = spice.str2et(utc_time)
	
	# Get state vectors for the spacecraft
	state, _ = spice.spkezr(target, et, frame, "NONE", observer)
	position = state[:3] # X, Y, Z
	velocity = state[3:]
	distance = compute_distance(position)
	spice.kclear()
	
	if test:
		print(f"Target: {target}")
		print(f"UTC Time: {utc_time}")
		print(f"Frame: {frame}")
		print(f"Observer: {observer}")
		print(f"Position: {position}")
		print(f"Velocity: {velocity}")
		print(f"Distance: {distance:.3f} km\n")
	else:
		return position

# query_spacecraft_position_vectors(metakernel_path, test=True)

def query_spacecraft_solar_distance(
        spice_metakernel_path: str,
        target: str = 'HERA',
        utc_time: str = '2025-12-27T00:00:00',
        frame: str = 'J2000',
        observer: str = 'SUN'
    ):
    """
    Query the distance between a spacecraft and the Sun.
    
    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    target (str): The spacecraft of interest.
    utc_time (str): The UTC time for which the distance is required.
    frame (str): The reference frame for calculations.
    observer (str): The observing body (default is the Sun).
    
    Returns:
    float: Distance between the spacecraft and the Sun in kilometers.
    """
    try:
        # Ensure the metakernel path is provided
        if not spice_metakernel_path:
            raise ValueError("SPICE metakernel path is required.")
        
        # Query spacecraft position relative to the Sun
        position = query_spacecraft_position_vectors(
            spice_metakernel_path, target, utc_time, frame, observer
        )
        
        # Compute the distance from the Sun
        distance = compute_distance(position)
        
        return distance
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft solar distance: {e}")
    
    return None

# print(query_spacecraft_solar_distance(metakernel_path))

def query_spacecraft_quaternions(
		metakernel_path: str,
		utc_time: str = '2025-12-27T00:01:00',
		inertial_frame: str = 'J2000',
		spacecraft_frame: str = 'HERA_DIDYMOS_NPO'
	):
    """
    Query the quaternion representing the orientation of the spacecraft.
    
    This function loads the specified SPICE kernels, computes the rotation matrix
    from the given inertial frame to the spacecraft's body-fixed frame at the 
    specified UTC time, converts it to a quaternion, and returns the quaternion 
    in the order (W, X, Y, Z).

    Parameters:
      metakernel_path (str): Path to the SPICE metakernel file.
      utc_time (str): The UTC time for which the quaternion is queried.
      inertial_frame (str): The inertial reference frame (default is 'J2000').
      spacecraft_frame (str): The spacecraft's body-fixed frame (default is 'HERA_FIXED').

    Returns:
      tuple: A tuple (W, X, Y, Z) representing the spacecraft quaternion.
      
    Note:
      Ensure that the necessary CK kernels (providing the spacecraft's orientation)
      are available and correctly referenced in the metakernel.
    """
    # Load the SPICE kernels
    spice.furnsh(metakernel_path)
    
    # Convert UTC time to ephemeris time (ET)
    et = spice.str2et(utc_time)
    
    # Get the rotation matrix from the inertial frame to the spacecraft frame.
    # This matrix represents the spacecraft's orientation at time et.
    rot_matrix = spice.pxform(inertial_frame, spacecraft_frame, et)
    
    # Convert the rotation matrix to a quaternion (W, X, Y, Z).
    quat = spice.m2q(rot_matrix)
    
    # Clear kernels to free resources.
    spice.kclear()
    
    return quat

# print(query_spacecraft_quaternions(metakernel_path))

def list_available_frames(metakernel_path: str):
    # Load SPICE kernels
    spice.furnsh(metakernel_path)
    
    # Get a list of all available frames
    frame_count = spice.ktotal("ALL")
    print(f"Total number of frames: {frame_count}")
    
    for i in range(frame_count):
        frame_name = spice.kdata(i, "ALL")
        print(f"Frame {i+1}: {frame_name}")
    
    # Clear SPICE kernels
    spice.kclear()

# list_available_frames(metakernel_path)
