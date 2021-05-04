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
bb_alexandras = 37.992392, 37.990870, 23.733559, 23.730533


#  Detectors default settings
len_det = 15
dfi = 10
n_det = 2

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
datasets_name = "datasets"
datasets = {
    'zone_7': '20181030_d7_0900_0930.csv',
    'zone_10': '20181030_d10_0900_0930.csv',
    'zone_all': '20181030_dX_0800_0830.csv',
    'zone_5': '/Volumes/Samsung_T5/pNEUMA_tests/datasets/20181030_d5_0930_1000.csv',
    'z2410_d5_10': '/Volumes/Samsung_T5/pNEUMA_tests/datasets/20181024_d5_1000_1030.csv'
}

pan_id = [41, 42, 43, 46, 47, 48, 49, 1032, 1034, 1129, 1308, 1309, 2204, 2544]
alexandras_ids = [1941, 2251, 2250, 1952, 1943, 1951, 1946, 1948, 2254, 2253, 2249, 2247, 2245, 1953, 371]
alexandras_ids_extra = [1312, 1997, 1996, 2257, 2255, 2512, 2509, 2000, 2256, 2248, 2664, 2663, 2332, 277, 278, 247,
                        373, 2246, 374]

# Test case Panepistimiou xt-plots
pan_l1 = [1129, 47, 1308]
pan_l2 = [1032, 1034, 42, 46, 48, 1309]
pan_l3 = [2204, 2254, 49]

# From QGis: In and Out detectors
# From detectors features
row_in = 't_ts1'
row_in_det = 't2'
row_out = 't1'

pan_in = [(46, 6630009199, row_in), (1129, 6630009200, row_in), (2544, 808033547, row_in)]
pan_in_det = [(46, row_in_det), (1129, row_in_det), (2544, row_in_det)]
# From detectors generic
pan_out = [(48, row_out), (49, row_out), (1308, row_out)]
