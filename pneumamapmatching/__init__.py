"""
pNEUMA package to work with dataset of pNEUMA experiment

Author: Landtmeters Joachim, KU Leuven
Data source: pNEUMA – open-traffic.epfl.ch
"""
import os
from pathlib import Path
import pickle

from .settings import *
from ._api import *

# Folders to store intermediate results and data sets
# Created in working directory of run python script
# In future versions decisions on how to handle best with intermediate data and results will be made
results_folder = 'results'
data_folder = 'data'
Path(os.getcwd()+"/"+data_folder).mkdir(parents=True, exist_ok=True)
Path(os.getcwd()+"/"+results_folder + "/plots").mkdir(parents=True, exist_ok=True)
Path(os.getcwd()+"/"+results_folder + "/crossings").mkdir(parents=True, exist_ok=True)
Path(os.getcwd()+"/"+data_folder + "/shapefiles").mkdir(parents=True, exist_ok=True)
Path(os.getcwd()+"/"+results_folder + "/case_studies").mkdir(parents=True, exist_ok=True)
Path(os.getcwd()+"/"+results_folder).mkdir(parents=True, exist_ok=True)

# Folders to store intermediate results and data sets
path_data = os.path.join(os.getcwd(), data_folder)
path_results = os.path.join(os.getcwd(), results_folder)
path_case_studies = os.path.join(os.getcwd(), results_folder, 'case_studies')


def write_pickle(obj, filename, path=None):
    if path is None:
        path = os.getcwd()
    filename = os.path.join(path, filename)
    filename = os.path.normpath(filename)
    with open(filename, 'wb') as a:
        pickle.dump(obj, a)


def read_pickle(filename, path=None):
    if path is None:
        path = os.getcwd()
    filename = os.path.join(path, filename)
    filename = os.path.normpath(filename)
    with open(filename, 'rb') as a:
        obj = pickle.load(a)
    return obj


