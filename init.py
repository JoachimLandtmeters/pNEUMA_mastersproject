import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import networkx as nx
import pandas as pd
import geopandas as gpd
from collections import Counter, OrderedDict
from shapely.geometry import Point, LineString
from operator import itemgetter
from statistics import mean
import numpy as np
from pylab import *
import pickle
import json
import compassbearing
import geopy.distance as dist
import time
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm, PowerNorm, Normalize, LinearSegmentedColormap
import collections
from tqdm import tqdm
from tqdm.contrib import tenumerate
import timeit
import seaborn
import scipy
import leuvenmapmatching as lm
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import *
from leuvenmapmatching import visualization as mmviz
import leuvenmapmatching.util.dist_latlon as lm_dist
import leuvenmapmatching.util.dist_euclidean as lm_dist_euclidean
import sys
import logging
import smopy
import sklearn
import sys
import math
import statistics
from pyproj import Proj, transform
import similaritymeasures
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

from create_network import *
from map_matching_trajectories import *
from make_detector import *
from calc_parameters import *
import visualization as vz

tic = time.time()

do_counts = False
all_arterials = False
pickle_out = True

frequency = 10000
n_det = 3
double = True
dfi = 15
det_width = 15
loop_distance = 5
colors = ['g', 'sandybrown', 'darkorange', 'r', 'deeppink', 'gold', 'purple', 'b']
color_by_type = {'Car': 'r', 'Taxi': 'b', 'Motorcycle': 'g', 'Bus': 'orange',
                 'Medium Vehicle': 'purple', 'Heavy Vehicle': 'purple'}

with open('athens_obj', 'rb') as a:
    athens = pickle.load(a)
used_network = pd.read_pickle('used_network_lastit')
drive_network = pd.read_pickle('drive_network_long_lastit')


def traffic_analysis_obj_to_pickle(ta_obj, name, frequency, dfi):
    with open(f'traffic_analysis_{name}_f{round(int(frequency / 1000))}_dfi{dfi}', 'wb') as a:
        pickle.dump(ta_obj, a)
    return print('File saved')


def traffic_analysis_obj_read_pickle(name, frequency, dfi):
    with open(f'traffic_analysis_{name}_f{round(int(frequency / 1000))}_dfi{dfi}', 'rb') as a:
        ta_obj = pickle.load(a)
    return ta_obj


def traffic_analysis_count(det_gdf, traj, netw, modes_excl=(), loop_distance=loop_distance,
                           n_det=n_det, frequency=frequency, dfi=dfi, double=True,
                           save=True, name=None):
    tic = time.time()
    ta = TrafficAnalysis(det_gdf, traj, netw, n_det=n_det, freq=frequency, double_loops=double,
                         loop_distance=loop_distance, dfi=dfi,
                         mode_exclusion=modes_excl)
    if save:
        if name is None:
            name = 'new'
        traffic_analysis_obj_to_pickle(ta, name, frequency, dfi)
    toc = time.time()
    print(f'{int(divmod(toc - tic, 60)[0])} min {int(divmod(toc - tic, 60)[1])} sec')
    return ta


def effect_mode_fd(parameters, mode, n_det, exclude=False):
    for i, j in enumerate(parameters):
        for det in range(1, n_det+1):
            j[f'density_h{mode}_{det}'] = j[f'density_{det}']*j[f'{mode}_{det}']/100
            j[f'flow_h{mode}_{det}'] = j[f'flow_{det}'] * j[f'{mode}_{det}'] / 100
            j[f'speed_h{mode}_{det}'] = [j[f'flow_h{mode}_{det}'][e]/j[f'density_h{mode}_{det}'][e]
                                        if j[f'density_h{mode}_{det}'][e] > 0 else 0 for e, v in j.iterrows()]
            if exclude:
                j[f'density_hno{mode}_{det}'] = j[f'density_{det}'] * (100-j[f'{mode}_{det}']) / 100
                j[f'flow_hno{mode}_{det}'] = j[f'flow_{det}'] * (100-j[f'{mode}_{det}']) / 100
                j[f'speed_hno{mode}_{det}'] = [j[f'flow_hno{mode}_{det}'][e] / j[f'density_hno{mode}_{det}'][e]
                                            if j[f'density_hno{mode}_{det}'][e] > 0 else 0 for e, v in j.iterrows()]
    return parameters


def select_modes(param_base, mode_name, mode_selection):
    p_adj_base = param_base.traffic_parameters_adj
    p1 = copy.deepcopy(param_base)
    p2 = copy.deepcopy(param_base)
    p1.traffic_parameters = p1.calculate_parameters(mode_selection[0])  # Only this mode
    p2.traffic_parameters = p2.calculate_parameters(mode_selection[1])  # Mode excluded
    p1.traffic_parameters_adj = p1.adjustment_stopped_vehicles()
    p2.traffic_parameters_adj = p2.adjustment_stopped_vehicles()
    p_mode = effect_mode_fd(p_adj_base, mode_name, n_det, exclude=True)
    return p1, p2, p_mode, param_base


def plot_mode_fd(p_mode, p1, p2, edge, detector, mode_name):
    mode_name_2 = mode_name
    if mode_name == 'Motorcycle':
        mode_name_2 = 'PTW'
    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey='row', figsize=(12, 7))
    ax[0, 0].scatter(p_mode[edge][f'density_{detector}'], p_mode[edge][f'flow_{detector}'], color='k', s=15,
                     label='All modes')
    ax[0, 0].scatter(p_mode[edge][f'density_hno{mode_name}_{detector}'],
                     p_mode[edge][f'flow_hno{mode_name}_{detector}'], color='r', alpha=0.7,
                     label='Homogeneous')
    ax[0, 0].scatter(p2.traffic_parameters_adj[edge][f'density_{detector}'],
                     p2.traffic_parameters_adj[edge][f'flow_{detector}'], color='g',
                     alpha=0.7, label='Observed')
    ax[0, 0].legend(loc='upper left')
    ax[0, 0].set_title(f'No {mode_name_2}')
    ax[0, 0].set_ylabel('Flow [veh/h]')
    ax[0, 1].scatter(p_mode[edge][f'density_{detector}'], p_mode[edge][f'flow_{detector}'], color='k', s=15,
                     label='All modes')
    ax[0, 1].scatter(p_mode[edge][f'density_h{mode_name}_{detector}'],
                     p_mode[edge][f'flow_h{mode_name}_{detector}'], color='r', alpha=0.7,
                     label='Homogeneous')
    ax[0, 1].scatter(p1.traffic_parameters_adj[edge][f'density_{detector}'],
                     p1.traffic_parameters_adj[edge][f'flow_{detector}'], color='g',
                     alpha=0.7, label= 'Observed')
    ax[0, 1].legend(loc='upper left')
    ax[0, 1].set_title(f'Only {mode_name_2}')
    ax[1, 0].scatter(p_mode[edge][f'density_{detector}'], p_mode[edge][f'speed_{detector}'], color='k', s=15,
                     label='All modes')
    ax[1, 0].scatter(p_mode[edge][f'density_hno{mode_name}_{detector}'],
                     p_mode[edge][f'speed_hno{mode_name}_{detector}'], color='r', alpha=0.7,
                     label='Homogeneous')
    ax[1, 0].scatter(p2.traffic_parameters_adj[edge][f'density_{detector}'],
                     p2.traffic_parameters_adj[edge][f'speed_{detector}'], color='g',
                     alpha=0.7, label='Observed')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].set_ylabel('Speed [km/h]')
    ax[1, 0].set_xlabel('Density [veh/km]')
    ax[1, 1].scatter(p_mode[edge][f'density_{detector}'], p_mode[edge][f'speed_{detector}'], color='k', s=15,
                     label='All modes')
    ax[1, 1].scatter(p_mode[edge][f'density_h{mode_name}_{detector}'],
                     p_mode[edge][f'speed_h{mode_name}_{detector}'], color='r', alpha=0.7,
                     label='Homogeneous')
    ax[1, 1].scatter(p1.traffic_parameters_adj[edge][f'density_{detector}'],
                     p1.traffic_parameters_adj[edge][f'speed_{detector}'], color='g',
                     alpha=0.7, label='Observed')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].set_xlabel('Density [veh/km]')
    fig.align_labels()


onlyPTW = ('Pedestrian', 'Bicycle', 'Car', 'Taxi', 'Bus', 'Medium Vehicle', 'Heavy Vehicle')
onlyMV = ('Pedestrian', 'Bicycle', 'Car', 'Taxi', 'Bus', 'Motorcycle', 'Heavy Vehicle')
onlyHV = ('Pedestrian', 'Bicycle', 'Car', 'Taxi', 'Bus', 'Medium Vehicle', 'Motorcycle')
onlyMHV = ('Pedestrian', 'Bicycle', 'Car', 'Taxi', 'Bus', 'Motorcycle')
onlyCMHV = ('Pedestrian', 'Bicycle', 'Taxi', 'Bus', 'Motorcycle')
onlyLV = ('Pedestrian', 'Bicycle', 'Car', 'Taxi', 'Motorcycle')
onlyB = ('Pedestrian', 'Bicycle', 'Car', 'Taxi', 'Motorcycle', 'Medium Vehicle', 'Heavy Vehicle')
onlyT = ('Pedestrian', 'Bicycle', 'Car', 'Motorcycle', 'Bus', 'Medium Vehicle', 'Heavy Vehicle')
onlyC = ('Pedestrian', 'Bicycle', 'Motorcycle', 'Taxi', 'Bus', 'Medium Vehicle', 'Heavy Vehicle')

onlyDBL = ('Pedestrian', 'Bicycle', 'Car', 'Taxi', 'Medium Vehicle', 'Heavy Vehicle')
noT = ('Pedestrian', 'Bicycle', 'Taxi')
noB = ('Pedestrian', 'Bicycle', 'Bus')
noPT = ('Pedestrian', 'Bicycle', 'Taxi', 'Bus')
noPTW = ('Pedestrian', 'Bicycle', 'Motorcycle')
noDBL = ('Pedestrian', 'Bicycle', 'Motorcycle', 'Bus')
noLV = ('Pedestrian', 'Bicycle', 'Bus', 'Medium Vehicle', 'Heavy Vehicle')

used_network_dblidx = list(used_network[used_network.dbl_right == 'lane']['index'].values)


det_obj = Detectors(athens.graph, used_network, det_width, dfi, loop_distance, n_det, arterial_lists=True)
all_index = []
for i, (k,v) in enumerate(det_obj.arterial_index.items()):
    all_index = all_index + v
det_all_art = det_obj.detector_selection(all_index)
det_om = det_obj.detector_arterials('omonoia')
det_pan = det_obj.detector_arterials('panepistimiou')
det_aka = det_obj.detector_arterials('akadimias')
det_alxEW = det_obj.detector_arterials('alexandrasEW')
det_alxWE = det_obj.detector_arterials('alexandrasWE')
det_tit = det_obj.detector_arterials('titris')
det_oct = det_obj.detector_arterials('october28')
det_sta = det_obj.detector_arterials('stadiou')

det_obj_drive = Detectors(athens.graph, drive_network, det_width, dfi, loop_distance, n_det)
arterial_names = ['Akadimias', 'Alexandras EW', 'Alexandras WE', 'October 28', 'Omonoia',
                  'Panepistimiou', 'Stadiou', 'Titris']
stats_names = ['AKA', 'ALE', 'ALW', 'OCT', 'OM', 'PAN', 'STA', 'TIT']
if not all_arterials:

    if pickle_out:
        ta_aka = traffic_analysis_obj_read_pickle('akadimias', frequency, dfi)
        ta_aka.loop_distance = loop_distance
        ta_alxEW = traffic_analysis_obj_read_pickle('alexandrasEW', frequency, dfi)
        ta_alxEW.loop_distance = loop_distance
        ta_alxWE = traffic_analysis_obj_read_pickle('alexandrasWE', frequency, dfi)
        ta_alxWE.loop_distance = loop_distance
        ta_oct = traffic_analysis_obj_read_pickle('october28', frequency, dfi)
        ta_oct.loop_distance = loop_distance
        ta_om = traffic_analysis_obj_read_pickle('omonoia', frequency, dfi)
        ta_om.loop_distance = loop_distance
        ta_pan = traffic_analysis_obj_read_pickle('panepistimiou', frequency, dfi)
        ta_pan.loop_distance = loop_distance
        ta_sta = traffic_analysis_obj_read_pickle('stadiou', frequency, dfi)
        ta_sta.loop_distance = loop_distance
        ta_tit = traffic_analysis_obj_read_pickle('titris', frequency, dfi)
        ta_tit.loop_distance = loop_distance
        ls_arterial = [ta_aka, ta_alxEW, ta_alxWE, ta_oct, ta_om, ta_pan, ta_sta, ta_tit]

if do_counts:
    line_trajectories = list_dfs_read_pickle('line_traj_match_lastit')

    if all_arterials:
        ta_omonoia = TrafficAnalysis(det_om, line_trajectories, used_network,
                                     n_det=n_det, freq=frequency, double_loops=True, dfi=dfi)
        traffic_analysis_obj_to_pickle(ta_omonoia, 'omonoia', frequency, dfi)

        ta_pan = TrafficAnalysis(det_pan, line_trajectories, used_network,
                                 n_det=n_det, freq=frequency, double_loops=True, dfi=dfi)
        traffic_analysis_obj_to_pickle(ta_pan, 'panepistimiou', frequency, dfi)

        ta_akadimias = TrafficAnalysis(det_aka, line_trajectories, used_network,
                                       n_det=n_det, freq=frequency, double_loops=True, dfi=dfi)
        traffic_analysis_obj_to_pickle(ta_akadimias, 'akadimias', frequency, dfi)

        ta_alxEW = TrafficAnalysis(det_alxEW, line_trajectories, used_network,
                                   n_det=n_det, freq=frequency, double_loops=True, dfi=dfi)
        traffic_analysis_obj_to_pickle(ta_alxEW, 'alexandrasEW', frequency, dfi)

        ta_alxWE = TrafficAnalysis(det_alxWE, line_trajectories, used_network,
                                   n_det=n_det, freq=frequency, double_loops=True, dfi=dfi)
        traffic_analysis_obj_to_pickle(ta_alxWE, 'alexandrasWE', frequency, dfi)

        ta_titris = TrafficAnalysis(det_tit, line_trajectories, used_network,
                                    n_det=n_det, freq=frequency, double_loops=True, dfi=dfi)
        traffic_analysis_obj_to_pickle(ta_titris, 'titris', frequency, dfi)

        ta_october28 = TrafficAnalysis(det_oct, line_trajectories, used_network,
                                       n_det=n_det, freq=frequency, double_loops=True, dfi=dfi)
        traffic_analysis_obj_to_pickle(ta_october28, 'october28', frequency, dfi)

        ta_stadiou = TrafficAnalysis(det_sta, line_trajectories, used_network,
                                     n_det=n_det, freq=frequency, double_loops=True, dfi=dfi)
        traffic_analysis_obj_to_pickle(ta_stadiou, 'stadiou', frequency, dfi)


toc = time.time()
print(f'{int(divmod(toc - tic, 60)[0])} min {int(divmod(toc - tic, 60)[1])} sec')

"""
ta_aka = traffic_analysis_obj_read_pickle('akadimias', frequency, dfi)
ta_alxEW = traffic_analysis_obj_read_pickle('alexandrasEW', frequency, dfi)
ta_alxWE = traffic_analysis_obj_read_pickle('alexandrasWE', frequency, dfi)
ta_oct = traffic_analysis_obj_read_pickle('october28', frequency, dfi)
ta_om = traffic_analysis_obj_read_pickle('omonoia', frequency, dfi)
ta_pan = traffic_analysis_obj_read_pickle('panepistimiou', frequency, dfi)
ta_sta = traffic_analysis_obj_read_pickle('stadiou', frequency, dfi)
ta_tit = traffic_analysis_obj_read_pickle('titris', frequency, dfi)
ls_arterial = [ta_aka, ta_alxEW, ta_alxWE, ta_oct, ta_om, ta_pan, ta_sta, ta_tit]
arterial_names = ['Akadimias', 'Alexandras EW', 'Alexandras WE', 'Ocotber 28', 'Omonoia',
 'Panepistimiou', 'Stadiou', 'Titris']
mfd_par = []
for i, j in enumerate(ls_arterial):
    art = j.arterial_parameters_all()

mfd_par = []
length_network = []
for i, j in enumerate(ls_arterial):
    length_network.append(sum(j.traffic_counts['detectors']['length']*j.traffic_counts['detectors']['lanes_adj']))
    mfd_par.append(j.arterial_parameters_all)
mfd = {'accumulation': mfd_par[0].accumulation_arterial, 'production': mfd_par[0].production_arterial}
for i, j in enumerate(mfd_par):
    if i > 0:
        mfd['accumulation'] += j.accumulation_arterial
        mfd['production'] += j.production_arterial
        
        
Arterial Fundamental Diagrams

afd = []
for ind, art in enumerate(ls_arterial):
    afd.append(art.arterial_parameters_all())
fig, ax = plt.subplots(nrows=2, ncols=4)
for el, val in enumerate(afd):
    if el < 4:
        ax[0, el].scatter(val.accumulation_arterial, val.average_speed_arterial, color='k', s=10)
        ax[0, el].set_title(str(el))
        ax[0, el].set_xlim(0,)
        ax[0, el].set_ylim(0,)
    else:
        ax[1, el-4].scatter(val.accumulation_arterial, val.average_speed_arterial, color='k', s=10)
        ax[1, el-4].set_title(str(el))
        ax[1, el-4].set_xlim(0,)
        ax[1, el-4].set_ylim(0,)
plt.tight_layout(rect=[0, 0.03, 1, 0.90])

fig, ax = plt.subplots(nrows=2, ncols=4, sharey=True)
for i, j in enumerate(ls_arterial):
    if i < 4:
        ax[0,i].plot(j.traffic_counts['detectors'].arterial_order+1,j.traffic_counts['detectors'].lanes_adj, marker='o', lw=0.75)
        ax[0, i].set_title(stats_names[i])
        ax[0, i].grid(True)
        ax[0, i].set_xticks([])
    else:
        ax[1,i-4].plot(j.traffic_counts['detectors'].arterial_order+1,j.traffic_counts['detectors'].lanes_adj, marker='o', lw=0.75)
        ax[1, i-4].set_title(stats_names[i])
        ax[1, i-4].grid(True)
        ax[1, i-4].set_xticks([])
plt.tight_layout(rect=[0, 0.03, 1, 0.90])

par_omT = ta_om.calculate_parameters(onlyT)
par_omNT = ta_om.calculate_parameters(noT)
par_omPTW = ta_om.calculate_parameters(onlyPTW)
par_omNPTW = ta_om.calculate_parameters(noPTW)
omT = copy.deepcopy(ta_om)
omT.traffic_parameters = par_omT
omT.traffic_parameters_adj = omT.adjustment_stopped_vehicles()
omNT = copy.deepcopy(ta_om)
omNT.traffic_parameters = par_omNT
omNT.traffic_parameters_adj = omNT.adjustment_stopped_vehicles()
omPTW = copy.deepcopy(ta_om)
omPTW.traffic_parameters = par_omPTW
omPTW.traffic_parameters_adj = omPTW.adjustment_stopped_vehicles()
omNPTW = copy.deepcopy(ta_om)
omNPTW.traffic_parameters = par_omNPTW
omNPTW.traffic_parameters_adj = omNPTW.adjustment_stopped_vehicles()
p_omT = effect_mode_fd(ta_om.traffic_parameters_adj, 'Taxi', n_det, exclude=True)
p_omPTW = effect_mode_fd(ta_om.traffic_parameters_adj, 'Motorcycle', n_det, exclude=True)

fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey='row')
ax[0,0].scatter(p_omadj[0].density_2,p_omadj[0].flow_2, color='k', s=15, label='All modes')
ax[0,0].scatter(p_omadj[0].density_hnoTaxi_2,p_omadj[0].flow_hnoTaxi_2, color='r', alpha=0.7, label='No Taxi: Homogeneous')
ax[0,0].scatter(omNT.traffic_parameters_adj[0].density_2,omNT.traffic_parameters_adj[0].flow_2, color='g', alpha=0.7, label='No Taxi: Observed')
ax[0,0].legend()
ax[0,1].scatter(p_omadj[0].density_2,p_omadj[0].flow_2, color='k', s=15, label='All modes')
ax[0,1].scatter(p_omadj[0].density_hTaxi_2,p_omadj[0].flow_hTaxi_2, color='r', alpha=0.7, label='Only Taxi: Homogeneous')
ax[0,1].scatter(omT.traffic_parameters_adj[0].density_2,omT.traffic_parameters_adj[0].flow_2, color='g', alpha=0.7, label='Only Taxi: Observed')
ax[0,1].legend()
ax[1,0].scatter(p_omadj[0].density_2,p_omadj[0].speed_2, color='k', s=15, label='All modes')
ax[1,0].scatter(p_omadj[0].density_hnoTaxi_2,p_omadj[0].flow_hnoTaxi_2/p_omadj[0].density_hnoTaxi_2, color='r', alpha=0.7, label='No Taxi: Homogeneous')
ax[1,0].scatter(omNT.traffic_parameters_adj[0].density_2,omNT.traffic_parameters_adj[0].speed_2, color='g', alpha=0.7, label='No Taxi: Observed')
ax[1,0].legend()
ax[1,1].scatter(p_omadj[0].density_2,p_omadj[0].speed_2, color='k', s=15, label='All modes')
ax[1,1].scatter(p_omadj[0].density_hTaxi_2,p_omadj[0].flow_hTaxi_2/p_omadj[0].density_hTaxi_2, color='r', alpha=0.7, label='Only Taxi: Homogeneous')
ax[1,1].scatter(omT.traffic_parameters_adj[0].density_2,omT.traffic_parameters_adj[0].speed_2, color='g', alpha=0.7, label='Only Taxi: Observed')
ax[1,1].legend()


dict50 = {j: [] for i, j in enumerate(ls_idx_50)}
for i, j in enumerate(ls_idx_50):
    for e, v in enumerate(ta_all.traffic_parameters):
        dict50[j].append(v.density_lane_1[j])
        if ta_all.traffic_counts['detectors']['length'].values[e] > (2*dfi+15):
            dict50[j].append(v.density_lane_2[j])
            dict50[j].append(v.density_lane_3[j])
avg_df50 = pd.DataFrame(dict50)
avg_df50.hist(bins=np.arange(0,500,step=10))
df_corr = pd.DataFrame() # Correlation matrix
df_p = pd.DataFrame()  # Matrix of p-values
for x in avg_df50.columns:
    for y in avg_df50.columns:
        corr = scipy.stats.mannwhitneyu(avg_df50[x], avg_df50[y], alternative='two-sided')
        df_corr.loc[x,y] = corr[0]
        df_p.loc[x,y] = corr[1]
"""
