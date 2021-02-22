"""
"""

crs_pneuma = 'epsg:4326'  # WGS84
crs_pneuma_proj = "epsg:32634"  # projects crs, region specific (UTM zone)
type_dict = {'Car': 'C', 'Taxi': 'T', 'Motorcycle': 'M', 'Bus': 'B', 'Medium Vehicle': 'MV', 'Heavy Vehicle': 'HV'}
vehicle_dim = {'Car': [2, 5], 'Motorcycle': [1, 2.5], 'Bus': [4, 12.5], 'Taxi': [2, 5],
               'Medium Vehicle': [2.67, 5.83], 'Heavy Vehicle': [3.3, 12.5],
               'Bicycle': [0, 0], 'Pedestrian': [0, 0]}
bb_athens = 37.9936, 37.9738, 23.7424, 23.7201
bb_alexandras = 37.992392, 37.990870, 23.733559, 23.730533

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

### Temporarily ###
# CSV data sets
zone_7 = '20181030_d7_0900_0930.csv'
zone_10 = '20181030_d10_0900_0930.csv'
zone_all = '20181030_dX_0800_0830.csv'

pan_id = [41, 42, 43, 46, 47, 48, 49, 1032, 1034, 1129, 1308, 1309, 2204, 2544]

# Test case Panepistimiou xt-plots
pan_l1 = [1129, 47, 1308]
pan_l2 = [1032, 1034, 42, 46, 48, 1309]
pan_l3 = [2204, 2254, 49]
