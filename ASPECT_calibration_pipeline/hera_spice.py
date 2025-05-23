import spiceypy as spice
import numpy as np

# Work in progress

"""
This is a Python file for using HERA SPICE kernels.
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

"""
Memo:
- hera_plan.tm metakernel provides Milani long term predicted trajectory
- hera_ops.tm is later updated with asteroid phase cubesat trajectories
"""

# Add metakernel path
metakernel_path = "/home/sysa/HERA/SPICE/hera_v180/hera/kernels/mk/hera_plan.tm" # for example /home/sysa/HERA/SPICE/hera/kernels/mk/hera_plan.tm

test_time = '2027-01-02T05:40:46.6666'

def compute_distance(position):
	return (position[0]**2 + position[1]**2 + position[2]**2) ** 0.5

def km_to_au(km):
    AU_PER_KM = 149597870.7
    return km / AU_PER_KM

def explore_kernel_info():
	
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
		
# list_available_spice_info(metakernel_path)

def print_pool_variables(metakernel_path: str):
	"""
	Prints all specified kernel pool variables and their values.
	Edit the search pattern as needed.
	
	It uses:
	  - gnpool() to get the list of variable names in the pool,
	  - gdpool() to attempt to get numeric (double precision) values, and
	  - gcpool() to get character values if numeric ones are not found.
	  
	Note: Make sure that one or more kernels have been loaded (e.g. via furnsh)
	before calling this function.
	"""
	spice.furnsh(metakernel_path)
	# Retrieve all kernel pool variable names (allowing up to 1000 values per variable)
	varnames = spice.gnpool("*FRAME*", 0, 1000) # Edit the search pattern as needed
	print(len(varnames))
	if not varnames:
		print("No kernel pool variables found.")
		return
	
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
				
# print_pool_variables(metakernel_path)

def query_spacecraft_position_vectors(
		metakernel_path: str,
		target: str = 'Milani',
		utc_time: str = test_time,
		frame: str = "DIMORPHOS_FIXED",  # DIDYMOS_FIXED or DIMORPHOS_FIXED, or J2000 for inertial
		observer: str = "Dimorphos", # Didymos or Dimorphos, or Didymos_barycenter
		test: bool = False
	):
	"""
	Query the position vectors of spacecraft in a specified time frame.

	Parameters:
	metakernel_path (str): Path to the SPICE metakernel file.
	target (str): The spacecraft of interest.
	utc_time (str): The UTC time for which the position vectors are required.
	frame (str): The reference frame.
	observer (str): The observing body.
	test (bool): If True, print the results to the console.
	"""
	spice.furnsh(metakernel_path)
	
	et = spice.str2et(utc_time)
	
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

def test_meta():
	spice.furnsh("/home/sysa/HERA/SPICE/HERA/kernels/mk/hera_plan.tm")
	spice.kclear()

# test_meta()

def query_spacecraft_solar_distance(
        spice_metakernel_path: str,
        target: str = 'SUN',
        utc_time: str = test_time,
        frame: str = 'J2000',
        observer: str = 'Milani'
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
        
        return km_to_au(distance)
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft solar distance: {e}")

# print(query_spacecraft_solar_distance(metakernel_path))

def query_spacecraft_quaternions(
		metakernel_path: str,
		utc_time: str = test_time,
		inertial_frame: str = "DIMORPHOS_FIXED",  # DIDYMOS_FIXED or DIMORPHOS_FIXED, or J2000 for inertial
		spacecraft_frame: str = 'MILANI_SPACECRAFT'
	):
    """
    Query the quaternion representing the orientation of the spacecraft.
    
    This function computes the rotation matrix
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

# print(query_spacecraft_quaternions(metakernel_path, utc_time='2027-02-03T05:40:46.6666', spacecraft_frame='DIMORPHOS_FIXED'))

def query_solar_elongation(
		metakernel_path: str,
		utc_time: str = test_time,
		target: str = "Dimorphos", # Didymos or Dimorphos, or Didymos_barycenter
		observer: str = 'Milani'
	):
	"""
	Calculates the solar elongation angle for a given target as observed from the spacecraft.

	Solar elongation is the angle between the vector from the observer to the target 
	and the vector from the observer to the Sun.

	Args:
		metakernel_path (str): Path to the SPICE meta-kernel file containing necessary kernels.
		utc_time (str): UTC time of the observation, in the format 'YYYY-MM-DDTHH:MM:SS'. 
						Default is '2025-12-27T00:00:00'.
		target (str): Name of the target body for which the solar elongation is to be computed.
		observer (str): Name of the observing spacecraft or body.

	Returns:
		float: Solar elongation angle in degrees.
	"""
	try:
		# --- Load SPICE kernels (update the path to your Hera meta-kernel) ---
		spice.furnsh(metakernel_path)  # This meta-kernel should load HERA, SUN, and target ephemeris

		# --- Set the epoch of interest ---
		et = spice.str2et(utc_time)

		# --- Retrieve state vectors from HERA ---
		# Get the state vector of the target relative to HERA
		state_target, lt_target = spice.spkezr(target, et, "J2000", "NONE", observer)
		# Get the state vector of the Sun relative to HERA
		state_sun, lt_sun = spice.spkezr("SUN", et, "J2000", "NONE", observer)

		# Extract the position (first 3 components) from the state vectors
		vec_target = state_target[:3]
		vec_sun    = state_sun[:3]

		# --- Compute solar elongation ---
		# The solar elongation is the angular separation between the two vectors:
		angle_rad = spice.vsep(vec_target, vec_sun)
		angle_deg = np.degrees(angle_rad)

		# --- Clean up ---
		spice.kclear()

		return angle_deg

	except Exception as e:
		print(f"An error occurred while computing solar elongation: {e}")

# print(query_solar_elongation(metakernel_path))

def query_spacecraft_clock_start(
		metakernel_path: str,
		pool_variable: str = "SCLK_PARTITION_START_9102"
	):
    """
	Disclaimer: SCLK contains fictional data (last checked 2025-02-10)

    Retrieves the spacecraft clock start (i.e. the SCLK partition start)
    from the HERA SPICE kernel dataset.

    This function loads the meta-kernel (which should furnish the necessary
    SCLK kernel for HERA), retrieves the SCLK partition start value from the
    kernel pool.

    Args:
        metakernel_path (str): Path to the SPICE meta-kernel file containing the
                               necessary kernels for HERA.
        pool_variable (str): The name of the kernel pool variable that holds
                             the spacecraft clock start value.
                             Default is "SCLK_PARTITION_START_HERA".

    Returns:
        float: sclk_start
    """
    try:
        # Load the SPICE kernels
        spice.furnsh(metakernel_path)
        
        # Retrieve the SCLK partition start value from the kernel pool.
        # gdpool returns an array of doubles.
        sclk_start_array = spice.gdpool(pool_variable, 0, 1)
        sclk_start = sclk_start_array[0]
        
        # Decode the SCLK start value to a SCLK string.
        # The first argument ('HERA') identifies the spacecraft (it should match the
        # spacecraft clock kernel's internal label).
        # sclk_start_str = spice.scdecd('HERA', sclk_start)
        
        # Clear the loaded kernels
        spice.kclear()
        
        return sclk_start#, sclk_start_str

    except Exception as e:
        print(f"An error occurred while retrieving the spacecraft clock start: {e}")
        spice.kclear()
        return None#, None
	
# Work in progress
# print_pool_variables(metakernel_path)
# print(query_spacecraft_clock_start(metakernel_path))

def spice_dataset_version(metakernel_path: str):
	"""
	Get the version of the HERA SPICE dataset from the metakernel.
	"""
	spice.furnsh(metakernel_path)
	version = spice.gcpool("SKD_VERSION", 0, 1)
	spice.kclear()
	return version[0]

# print(spice_dataset_version(metakernel_path))

def query_sun_position_vectors(
		metakernel_path: str,
		utc_time: str = test_time,
		frame: str = 'J2000', # Is this the correct frame?
		observer: str = 'Milani'
	):
	"""
	Query the position vectors of the Sun in a specified time frame.

	Parameters:
	metakernel_path (str): Path to the SPICE metakernel file.
	utc_time (str): The UTC time for which the position vectors are required.
	frame (str): The reference frame.
	observer (str): The observing body.

	Returns:
	tuple: A tuple (X, Y, Z) representing the Sun's position vector.
	"""
	spice.furnsh(metakernel_path)
	
	et = spice.str2et(utc_time)
	
	state, _ = spice.spkezr("SUN", et, frame, "NONE", observer)
	position = state[:3] # X, Y, Z
	spice.kclear()
	
	return position

# print(query_sun_position_vectors(metakernel_path))

def query_earth_position_vectors(
		metakernel_path: str,
		utc_time: str = test_time,
		frame: str = 'J2000', # Is this the correct frame?
		observer: str = 'Milani'
	):
	"""
	Query the position vectors of the Earth in a specified time frame.

	Parameters:
	metakernel_path (str): Path to the SPICE metakernel file.
	utc_time (str): The UTC time for which the position vectors are required.
	frame (str): The reference frame.
	observer (str): The observing body.

	Returns:
	tuple: A tuple (X, Y, Z) representing the Earth's position vector.
	"""
	spice.furnsh(metakernel_path)
	
	et = spice.str2et(utc_time)
	
	state, _ = spice.spkezr("EARTH", et, frame, "NONE", observer)
	position = state[:3] # X, Y, Z
	spice.kclear()
	
	return position

def query_spacecraft_earth_distance(
        spice_metakernel_path: str,
        target: str = 'EARTH',
        utc_time: str = test_time,
        frame: str = 'J2000',
        observer: str = 'Milani'
    ):
    """
    Query the distance between a spacecraft and the Earth.
    
    Parameters:
    spice_metakernel_path (str): Path to the SPICE metakernel file.
    target (str): The spacecraft of interest.
    utc_time (str): The UTC time for which the distance is required.
    frame (str): The reference frame for calculations.
    observer (str): The observing body (default is the Earth).
    
    Returns:
    float: Distance between the spacecraft and the Earth in kilometers.
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
        
        return km_to_au(distance)
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while retrieving spacecraft solar distance: {e}")

# print(query_spacecraft_earth_distance(metakernel_path))

def query_target_distance(
		metakernel_path: str,
		target: str = "Dimorphos", # Didymos or Dimorphos, or Didymos_barycenter
		utc_time: str = test_time,
		frame: str = 'DIMORPHOS_FIXED',
		observer: str = 'Milani'
	):
	"""
	Query the distance between the observer and the target body.

	Parameters:
	metakernel_path (str): Path to the SPICE metakernel file.
	target (str): The target body for which the distance is required.
	utc_time (str): The UTC time for which the distance is required.
	frame (str): The reference frame.
	observer (str): The observing body.

	Returns:
	float: The distance between the observer and the target body in kilometers.
	"""
	spice.furnsh(metakernel_path)
	
	et = spice.str2et(utc_time)
	
	state, _ = spice.spkezr(target, et, frame, "NONE", observer)
	position = state[:3] # X, Y, Z
	distance = compute_distance(position)
	
	spice.kclear()
	
	return km_to_au(distance)

# print(query_target_distance(metakernel_path))

def vector_to_unit_vector(vector):
	"""
	Converts a 3D vector to a unit vector.
	
	Parameters:
	vector (list): The input vector.
	
	Returns:
	list: The unit vector.
	"""
	magnitude = compute_distance(vector)
	return [component / magnitude for component in vector]

def info_for_asteroid_image_simulator(metakernel_path: str, utc_time: str = test_time, target: str = "Dimorphos"):
	"""
	Provides information for the asteroid image simulator.
	
	Parameters:
	metakernel_path (str): Path to the SPICE metakernel file.
	utc_time (str): The UTC time for which the information is required.
	target (str): The target body for which the information is required.
	
	Returns:
	dict: A dictionary containing the sibling asteroid's location,
	sun direction, camera direction, camera boresight, Didymos attitude, Dimorphos attitude.
	"""
	if target == "Dimorphos":
		sibling_asteroid_location = query_spacecraft_position_vectors(
			metakernel_path,
			target='Didymos',
			utc_time=utc_time,
			frame='DIMORPHOS_FIXED',
			observer='Dimorphos'
		)
		sun_location = query_sun_position_vectors(
			metakernel_path,
			utc_time=utc_time,
			frame='DIMORPHOS_FIXED',
			observer='Dimorphos'
		)
		sun_unit_vector = vector_to_unit_vector(sun_location)
		camera_location = query_spacecraft_position_vectors(
			metakernel_path,
			target='Milani',
			utc_time=utc_time,
			frame='DIMORPHOS_FIXED',
			observer='Dimorphos'
		)
		didymos_attitude = query_spacecraft_quaternions(
			metakernel_path,
			utc_time=utc_time,
			inertial_frame='DIMORPHOS_FIXED',
			spacecraft_frame='DIDYMOS_FIXED'
		)
		dimorphos_attitude = query_spacecraft_quaternions(
			metakernel_path,
			utc_time=utc_time,
			inertial_frame='DIMORPHOS_FIXED',
			spacecraft_frame='DIMORPHOS_FIXED'
		)
		result = {
			'utc time': utc_time,
			'target': target,
			'didymos location': tuple(sibling_asteroid_location),
			'sun location': tuple(sun_location),
			'sun unit vector': tuple(sun_unit_vector),
			'camera location': tuple(camera_location),
			'camera boresight': target,
			'didymos attitude (quaternion: W, X, Y, Z)': tuple(didymos_attitude),
			'dimorphos attitude (quaternion: W, X, Y, Z)': tuple(dimorphos_attitude)
		}
	elif target == "Didymos":
		sibling_asteroid_location = query_spacecraft_position_vectors(
			metakernel_path,
			target='Dimorphos',
			utc_time=utc_time,
			frame='DIDYMOS_FIXED',
			observer='Didymos'
		)
		sun_location = query_sun_position_vectors(
			metakernel_path,
			utc_time=utc_time,
			frame='DIDYMOS_FIXED',
			observer='Didymos'
		)
		sun_unit_vector = vector_to_unit_vector(sun_location)
		camera_location = query_spacecraft_position_vectors(
			metakernel_path,
			target='Milani',
			utc_time=utc_time,
			frame='DIDYMOS_FIXED',
			observer='Didymos'
		)
		didymos_attitude = query_spacecraft_quaternions(
			metakernel_path,
			utc_time=utc_time,
			inertial_frame='DIDYMOS_FIXED',
			spacecraft_frame='DIDYMOS_FIXED'
		)
		dimorphos_attitude = query_spacecraft_quaternions(
			metakernel_path,
			utc_time=utc_time,
			inertial_frame='DIDYMOS_FIXED',
			spacecraft_frame='DIMORPHOS_FIXED'
		)
		result = {
			'utc time': utc_time,
			'target': target,
			'dimorphos location': tuple(sibling_asteroid_location),
			'sun location': tuple(sun_location),
			'sun unit vector': tuple(sun_unit_vector),
			'camera location': tuple(camera_location),
			'camera boresight': target,
			'didymos attitude (quaternion: W, X, Y, Z)': tuple(didymos_attitude),
			'dimorphos attitude (quaternion: W, X, Y, Z)': tuple(dimorphos_attitude)
		}
	return result

# timestamp = '2027-02-27T00:00:00.0000'

# info = info_for_asteroid_image_simulator(metakernel_path, utc_time=timestamp, target='Didymos')
# for key, value in info.items():
# 	print(f"{key}: {value}")
