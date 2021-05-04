"""
"""
import matplotlib.pyplot as plt

qual_colorlist = [plt.cm.tab20(i) for i in range(20)]

crs_pneuma = 4326  # WGS84
crs_pneuma_proj = 32634  # projects crs, region specific (UTM zone)
type_dict = {'Car': 'C', 'Taxi': 'T', 'Motorcycle': 'M', 'Bus': 'B', 'Medium Vehicle': 'MV', 'Heavy Vehicle': 'HV'}
vehicle_dim = {'Car': [2, 5], 'Motorcycle': [1, 2.5], 'Bus': [4, 12.5], 'Taxi': [2, 5],
               'Medium Vehicle': [2.67, 5.83], 'Heavy Vehicle': [3.3, 12.5],
               'Bicycle': [0, 0], 'Pedestrian': [0, 0]}
bb_athens = 37.9936, 37.9738, 23.7424, 23.7201


#  Detectors default settings
len_det = 15
dfi = 10
n_det = 2

# Name of dataset dictionary
datasets_name = "datasets_paths"

# data name after reading it for first time --# pickle file in data folder
init_name = 'initial_data'

# HDF5 settings
hdf_name = 'pneuma_hdf'

# Custom filter to include pedestrian streets
adj_filter = ('["highway"]["area"!~"yes"]["access"!~"private"]'
              '["highway"!~"cycleway|footway|path|steps'
              '|track|corridor|elevator|escalator|proposed|construction|bridleway'
              '|abandoned|platform|raceway"]["motor_vehicle"!~"no"]["motorcar"!~"no"]'
              '["service"!~"parking|parking_aisle|private|emergency_access"]')

# logging setup
log_to_file = True
log_folder = 'logs'
log_level = 20  # Info level
log_filename = 'pneuma'

"""
logging info about levels:
CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0
"""

