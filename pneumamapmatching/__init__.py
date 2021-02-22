"""
pNEUMA package to work with dataset of pNEUMA experiment

Author: Landtmeters Joachim, KU Leuven
Data source: pNEUMA – open-traffic.epfl.ch
"""
import os
from pathlib import Path
import pickle

import pneumapackage.settings

# Folders to store intermediate results and data sets
# Created in working directory of run python script
# In future versions decisions on how to handle best with intermediate data and results will be made
results_folder = 'results'
data_folder = 'data'
Path(os.getcwd()+"/"+data_folder).mkdir(parents=True, exist_ok=True)
Path(os.getcwd()+"/"+results_folder).mkdir(parents=True, exist_ok=True)

# Folders to store intermediate results and data sets
path_data = os.getcwd() + '/' + data_folder + '/'
path_results = os.getcwd() + '/' + results_folder + '/'


def write_pickle(obj, filename, path=None):
    if path is None:
        path = os.getcwd()
    filename = path + filename
    with open(filename, 'wb') as a:
        pickle.dump(obj, a)


def read_pickle(filename, path=None):
    if path is None:
        path = os.getcwd()
    filename = path + filename
    with open(filename, 'rb') as a:
        obj = pickle.load(a)
    return obj


