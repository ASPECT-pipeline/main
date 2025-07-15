
# Meta data
software: str = 'ASPECTCAL v1.0'
missphase: str = ''
observph: str = '505'
target: str = 'DIDYMOS'
object: str = 'Didymos'

# Adjust these to point to the locations of spice metakernels
spice_mk_plan = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_plan.tm"
spice_mk_ops = "/Users/valtterimj/Downloads/Työ/Aalto/Hera/hera_spice/kernels/mk/hera_ops.tm"

# Adjust this to point to the metakernel to be used for FITS header data
spice_mk = spice_mk_plan

kelvin: float = 273.15

channel_map = {
    0 : 'VIS',
    1 : 'NIR1',
    2 : 'NIR2',
    3 : 'SWIR'
}