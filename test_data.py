"""
Test file for writing and reading trajectory data
"""
from pneumapackage.settings import *
import pneumapackage.mapmatching as mm
from pneumapackage.__init__ import read_pickle, write_pickle, path_data, path_results
import pneumapackage.iodata as rd

import test_network as tn

import h5py
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd

import time
import os
from pathlib import Path


# Initialize dataset

# Specify path to folder with data files, for windows using raw string input is advised
specify_data_path = None
if specify_data_path is None:
    specify_data_path = input('Specify path to folder with all data files: ')
    # If on windows replace single backslash with double backslash
data_dict = rd.initialize_data_paths(specify_data_path)


def get_hdf_names(group_id):
    try:
        group_path = rd.get_group(group_id)
    except KeyError:
        _, group_path = rd.io_data(group_id)
    path_id = group_path + '/all_id'
    path_org = group_path + '/original_trajectories'
    path_adj = group_path + '/adjusted_trajectories'
    return path_id, path_org, path_adj


def all_data_to_hdf(dict_data, day='', zone='', time_of_day=''):

    for k in dict_data.keys():
        if zone in k:
            _ = get_hdf_names(k)
    if len(zone) > 0:
        path_dict = rd.get_path_dict()
        path_dict['extra'] = zone
        write_pickle(path_dict, 'path_dict', path_data)
    print('All data stored in HDF5')


"""
Map-match trajectory data to network
"""


def test_adjusted_traj(group_id, redo=False, bearing=True, resample=True, step=1000):
    tic = time.time()
    print('Start: …load trajectories (resampled)')
    hdf_path = rd.get_hdf_path()
    path_id, path_org, path_adj = get_hdf_names(group_id)
    if redo:
        print('Create adjusted trajectory table and save the file in HDF5')
        ldf = rd.get_from_hdf(hdf_path, path_id, path_org, result='list')
        rd.new_dfs(ldf, group_id, bearing=bearing, resample=resample, step=step)
        ldf = rd.get_from_hdf(hdf_path, path_id, path_adj, result='list')
    else:
        try:
            ldf = rd.get_from_hdf(hdf_path, path_id, path_adj, result='list')
        except KeyError:
            print('Create adjusted trajectory table and save the file in HDF5')
            ldf = rd.get_from_hdf(hdf_path, path_id, path_org, result='list')
            rd.new_dfs(ldf, group_id, bearing=bearing, resample=resample, step=step)
            ldf = rd.get_from_hdf(hdf_path, path_id, path_adj, result='list')
    toc = time.time()
    print(f'Adjusted trajectories loaded, took {toc - tic}')
    return ldf


def adjust_data_in_hdf():
    tic = time.time()
    groups = rd.get_path_dict()['groups'].keys()
    for i in groups:
        print(f'Dataset: {i}')
        _ = test_adjusted_traj(i)
    toc = time.time()
    print(f'All datasets adjusted, took {toc-tic} sec')


# The initial values for the map-matching algorithm are chosen


def test_map_matching(group_id, traj_obj='/adjusted_trajectories', max_distance=10, rematch=False,
                      match_latlon=True, save_shp=False, path=path_data):
    tic = time.time()
    print('Start: …load map matching')
    # load network
    network_obj = tn.test_network()
    hdf_path = rd.get_hdf_path()
    group_path = rd.get_path_dict()['groups'][group_id]
    step = rd.get_path_dict()['current_resample_step'][group_id]
    if traj_obj not in ['/original_trajectories', '/adjusted_trajectories']:
        raise ValueError(f'traj_obj should be in ["/original_trajectories", "/adjusted_trajectories"]')
    try:
        with h5py.File(hdf_path, 'r') as s:
            tag_match = s[group_path].attrs[f'tag_mapmatching_{step}']
        if tag_match != network_obj.mm_id[group_id]:
            rematch = True
    except KeyError:
        rematch = True

    if not rematch:
        try:
            dfmatch_all = rd.get_from_hdf(hdf_path, key_id=group_path + '/all_id', key_tr=group_path + f'/mm_{step}ms',
                                          result='df_all')
        except (KeyError, TypeError):
            max_init_dist = int(max(network_obj.network_edges['length'])) + 1
            print(f'Initial distance: {max_init_dist} m, maximum distance (start): {max_distance} m')
            traj_unm = rd.get_from_hdf(hdf_path, key_id=group_path + '/all_id', key_tr=group_path + traj_obj,
                                       result='list')
            tmm = mm.MapMatching(traj_unm, network_obj, max_init=max_init_dist, max_d=max_distance,
                                 match_latlon=match_latlon)
            match_all = tmm.match_variable_distance(progress=False)
            dfmatch_all = pd.concat(match_all)
            step = rd.get_path_dict()['current_resample_step'][group_id]
            dfmatch_all.to_hdf(hdf_path, key=group_path + f'/mm_{step}ms', format='table', mode='a', append=False,
                               data_columns=['track_id', 'time', 'n1', 'n2', '_id'])
            dt = datetime.datetime.now()
            tag = int(dt.strftime('%Y%m%d%H%M'))
            with h5py.File(hdf_path, 'a') as s:
                s[group_path].attrs[f'tag_mapmatching_{step}'] = tag
            network_obj.add_mapmatch_tag(group_id, tag)
            write_pickle(network_obj, 'network', path)
    else:
        max_init_dist = int(max(network_obj.network_edges['length'])) + 1
        print(f'Initial distance: {max_init_dist} m, maximum distance (start): {max_distance} m')
        step = rd.get_path_dict()['current_resample_step'][group_id]
        traj_unm = rd.get_from_hdf(hdf_path, key_id=group_path + '/all_id', key_tr=group_path + traj_obj,
                                   result='list')
        tmm = mm.MapMatching(traj_unm, network_obj, max_init=max_init_dist, max_d=max_distance,
                             match_latlon=match_latlon)
        match_all = tmm.match_variable_distance(progress=False)
        dfmatch_all = pd.concat(match_all)
        step = rd.get_path_dict()['current_resample_step'][group_id]
        dfmatch_all.to_hdf(hdf_path, key=group_path + f'/mm_{step}ms', format='table', mode='a', append=False,
                           data_columns=['track_id', 'time', 'n1', 'n2', '_id'])
        dt = datetime.datetime.now()
        tag = int(dt.strftime('%Y%m%d%H%M'))
        with h5py.File(hdf_path, 'a') as s:
            s[group_path].attrs[f'tag_mapmatching_{step}'] = tag
        network_obj.add_mapmatch_tag(group_id, tag)
        write_pickle(network_obj, 'network', path)

    if save_shp:
        fn = path + f'/shapefiles/used_network_{group_id}_mm{step}'
        used_network = network_obj.network_edges[network_obj.network_edges['_id'].isin(dfmatch_all['_id'])]
        used_network.loc[:, ~used_network.columns.isin(['edge'])].to_file(filename=fn)
        Path(path + "/used_network").mkdir(parents=True, exist_ok=True)
        write_pickle(used_network, f'used_network_{group_id}_mm{step}', path=path + '/used_network')
        print('Shapefiles stored')

    toc = time.time()
    print(f'Map-matched trajectories loaded, took {toc - tic} sec')
    return {'tracks': dfmatch_all, 'network': network_obj}


def test_matched_LineTrajectories(group_id, selection=None, reload=False, **kwargs):
    tic = time.time()
    print('Start: …load matched trajectories')
    hdf_path = rd.get_hdf_path()
    group_path = rd.get_path_dict()['groups'][group_id]
    if reload:
        traj_obj = test_map_matching(group_id=group_id, **kwargs)
        traj_matched = traj_obj['tracks']
        network_obj = traj_obj['network']
        traj_match = mm.TransformTrajectories(traj_matched, network_obj)
        step = rd.get_path_dict()['current_resample_step'][group_id]
        traj_match.tracks_line.to_hdf(hdf_path, key=group_path + f'/mm_line_{step}ms', format='table', mode='a',
                                      append=False,
                                      data_columns=['time', 'u_match', 'v_match'])
        line_traj = traj_match.tracks_line
        if selection is not None:
            line_traj = traj_match.tracks_line.query(selection)
    else:
        try:
            step = rd.get_path_dict()['current_resample_step'][group_id]
            line_traj = rd.get_from_hdf(hdf_path, key_id=group_path + '/all_id', key_tr=group_path +
                                                                                        f'/mm_line_{step}ms',
                                        result='df_all', select_tr=selection)
        except (KeyError, TypeError):
            traj_obj = test_map_matching(group_id=group_id, **kwargs)
            traj_matched = traj_obj['tracks']
            network_obj = traj_obj['network']
            traj_match = mm.TransformTrajectories(traj_matched, network_obj)
            step = rd.get_path_dict()['current_resample_step'][group_id]
            traj_match.tracks_line.to_hdf(hdf_path, key=group_path + f'/mm_line_{step}ms', format='table', mode='a',
                                          append=False, data_columns=['time', 'u_match', 'v_match'])
            line_traj = traj_match.tracks_line
            if selection is not None:
                line_traj = traj_match.tracks_line.query(selection)

    toc = time.time()
    print(f'Matched trajectories loaded, took {toc - tic} sec')
    return line_traj


def get_lt(group_id, edges, gdf=False, lonlat=False):
    tic = time.time()
    hdf_path = rd.get_hdf_path()
    lt = test_matched_LineTrajectories(group_id=group_id, hdf_path=hdf_path, selection=f'u_match in {edges}')
    lt2 = test_matched_LineTrajectories(group_id=group_id, hdf_path=hdf_path, selection=f'v_match in {edges}')
    lt3 = pd.merge(lt.reset_index(), lt2.reset_index(), how='outer')
    lt3.set_index(['track_id', 'rid'], inplace=True)
    lt3.sort_index(inplace=True)
    if gdf:
        lt3 = mm.make_gdf(lt3, line=True, latlon=lonlat)
    toc = time.time()
    print(f'Selection line trajectories loaded, took {toc - tic} sec')
    return lt3


def get_lt_from_id(track_id, group_id, key_id='/all_id', gdf=False, lonlat=False):
    tic = time.time()
    hdf_path = rd.get_hdf_path()
    group_path = rd.get_path_dict()['groups'][group_id]
    key_tr = f'/mm_line_{rd.get_path_dict()["current_resample_step"][group_id]}ms'
    lt = rd.get_from_hdf(hdf_path, key_id=f'{group_path}{key_id}', key_tr=f'{group_path}{key_tr}',
                         result='df_all', select_tr=f'track_id in {track_id}')
    if gdf:
        lt = mm.make_gdf(lt, line=True, latlon=lonlat)
    toc = time.time()
    print(f'Selection line trajectories loaded, took {toc - tic} sec')
    return lt


def match_all_data_in_hdf():
    tic = time.time()
    groups = rd.get_path_dict()['groups'].keys()
    for i in groups:
        print(f'Dataset: {i}')
        _ = test_matched_LineTrajectories(i)
    toc = time.time()
    print(f'All datasets matched, took {toc-tic} sec')
