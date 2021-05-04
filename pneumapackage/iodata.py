"""
Data sets downloaded from pNEUMA – open-traffic.epfl.ch are read and converted to pandas dataframes to use in
subsequent coding.

Input: path to csv-file
Output: - List of dataframes, pickled to internal package folder 'data' (created on initialization of package)
        - HDF5 database, hierarchical structure with all recorded trajectories

This module serves as an initialization trying to store the trajectories in a structured way using pandas

Note:
Compressing is needed when append mode claims more and more disk space
HDF5 does not reclaim space even when file is overwritten with smaller file

in CLI : ptrepack --complevel=1 --complib=blosc {hdf_filename_original} {hdf_filename_new}
remove original file and rename new file to old filename

This is a bit of a hack but no simple feature of pandas or pytables is known to me that attains the same goal
"""

from pneumapackage.settings import init_name, hdf_name, crs_pneuma, datasets_name
from pneumapackage.__init__ import path_data, write_pickle, read_pickle
from pneumapackage.logger import log
from pneumapackage.network import project_improved
import pneumapackage.compassbearing as cpb

import os
import sys
from pathlib import Path
import time
import datetime

import pandas as pd
import h5py
import csv
from tqdm import tqdm
from tqdm.contrib import tenumerate
import pickle
import numpy as np

maxInt = min(sys.maxsize, 2147483646)
start = time.time()


# HDF5 will be used to structurally store the trajectory data
# Using this structure enable a lot of possibilities to only read in specific numeric data into memory so
# programs will not be slowed down by a large data sets taking up active memory
# The workflow currently proposed does operations very sequentially therefore this data format is a good choice
#
# Step 1: Create general hdf5 structure at the 'path_data' location
# Step 2: When adding specific data a group is added with the corresponding metadata and numeric values


def get_data_dict(path=path_data, name=datasets_name, reset=False):
    if not reset:
        try:
            data_dict = read_pickle(name, path=path)
            return data_dict
        except FileNotFoundError:
            data_dict = {}
            write_pickle(data_dict, name, path=path)
            return data_dict
    else:
        data_dict = {}
        write_pickle(data_dict, name, path=path)
        return data_dict


def create_group_id(csv_filename):
    csv_filename = csv_filename.rsplit(".")[0].rsplit("_")
    key_name = f"{csv_filename[0].replace('2018', '')}_{csv_filename[1]}_{csv_filename[2]}"
    return key_name


def group_id_from_path(data_path):
    if sys.platform == "win32":
        key_name = data_path.rsplit("\\")[-1]
    else:
        key_name = data_path.rsplit("/")[-1]
    key_name = create_group_id(key_name)
    return key_name


def add_data_path(data_path=None, name=datasets_name, path=path_data):
    if data_path is None:
        data_path = input('Give data path: ').strip('"').strip("'")
    data_dict = get_data_dict(name=name)
    key_name = group_id_from_path(data_path)
    data_dict[key_name] = data_path
    write_pickle(data_dict, datasets_name, path=path)
    return key_name


def initialize_data_paths(data_directory_path=None):
    if data_directory_path is None:
        data_directory_path = input('Give data path: ').strip('"').strip("'")
    filenames = []
    for r, d, f in os.walk(data_directory_path):
        for file in f:
            if not file.startswith('._'):
                filenames.append(file)
    data_paths = []
    for fn in filenames:
        data_paths.append(os.path.join(data_directory_path, fn))
    for dp in data_paths:
        _ = add_data_path(dp)
    return get_data_dict()


def init_hdf(path=path_data, name=hdf_name, reset=False):
    hdf_fn = os.path.join(path, name)
    if not os.path.exists(hdf_fn) or reset:
        with h5py.File(hdf_fn, 'w') as s:
            s.create_group('info')
            s['info'].attrs['pNEUMA'] = 'store pNEUMA data in HDF5'
            s['info'].attrs['crs'] = crs_pneuma
            dt = datetime.datetime.now()
            tag = dt.strftime('%Y/%m/%d %H:%M')
            s['info'].attrs['created'] = tag
        log(f'[HDF5] database created ({tag})', 20)
    return hdf_fn


def get_from_hdf(hdf, key_id=None, key_tr=None, select_id=None, select_tr=None, result='all'):
    """
    Get data from HDF5 database

    Parameters
    ----------
    hdf: path to hdf5 file
    key_id: group with table of all track ids
    key_tr: group with table of all individual trajectories
    select_id:
    select_tr:
    result:

    Returns
    ----------
    return: df with all ids, concatenated dataframe of all trajectories,  list of dataframes with trajectories
    """
    result_options = ['all', 'ids', 'df_all', 'list']
    if result not in result_options:
        raise KeyError(f'result parameter should be in {result_options}, current value = {result}')
    df_id, df_all, ldf = None, None, None
    if key_id is not None:
        df_id = pd.read_hdf(hdf, key=key_id, mode='r', where=select_id)
    if key_tr is not None:
        df_all = pd.read_hdf(hdf, key=key_tr, mode='r', where=select_tr)
        ldf = [gr for _, gr in df_all.groupby('track_id')]
        _ = [gr.reset_index(inplace=True, drop=True) for gr in ldf]
    if result == 'all':
        return df_id, df_all, ldf
    elif result == 'ids':
        return df_id
    elif result == 'df_all':
        return df_all
    elif result == 'list':
        return ldf


def get_path_dict(path=path_data, reset=False):
    name = 'path_dict'
    if not reset:
        try:
            path_dict = read_pickle(name, path=path)
            return path_dict
        except FileNotFoundError:
            path_dict = {'hdf_path': init_hdf(), 'groups': {}, 'current_resample_step': {}, 'extra': 'all_data'}
            write_pickle(path_dict, name, path=path_data)
            return path_dict
    else:
        path_dict = {'hdf_path': init_hdf(), 'groups': {}, 'current_resample_step': {}, 'extra': 'all_data'}
        write_pickle(path_dict, name, path=path_data)
        return path_dict


def get_hdf_path(**kwargs):
    path_dict = get_path_dict(**kwargs)
    hdf_path = path_dict['hdf_path']
    return hdf_path


def get_group(group_id=None, **kwargs):
    path_dict = get_path_dict(**kwargs)
    groups = path_dict['groups']
    if group_id is not None:
        return groups[group_id]
    else:
        return groups


def get_tree_structure(**kwargs):
    with pd.HDFStore(get_hdf_path(**kwargs), 'r') as f:
        for (path, subgroups, subkeys) in f.walk():
            for subgroup in subgroups:
                print("GROUP: {}/{}".format(path, subgroup))
            for subkey in subkeys:
                key = "/".join([path, subkey])
                print("KEY: {}".format(key))


def memory_usage(df):
    return round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)


def data_csv_to_list_dfs(path_trajectory_data):
    """
    Read raw csv data
    Structure every individual trajectory in a pandas dataframe and append to a list object comprising all trajectories

    :param path_trajectory_data: path to file with csv file of pNEUMA trajectory data
    Note that the conversion is dependent on the chosen format of LUTS lab (EPFL) to share pNEUMA trajectory data
    :return: List of DataFrames, with every index an individual trajectory
    """
    csv.field_size_limit(maxInt)
    data_file = open(path_trajectory_data, 'r')
    data_reader = csv.reader(data_file)
    data = []
    col_line = True
    for row in tqdm(data_reader):
        cl = [elem for elem in row[0].split("; ")]
        if col_line:
            col_types = {c: 'float32' for c in cl}
            col_types['type'] = 'category'
            col_types['lat'] = 'float64'
            col_types['lon'] = 'float64'
            col_line = False
        else:
            cl.pop()
            len_ls = int((len(cl) - 4) / 6)
            df = pd.DataFrame({j[0]: pd.Series([cl[i]] * len_ls, dtype=j[1]) if i < 4
            else pd.Series(cl[i::6], dtype=j[1]) for i, j in enumerate(col_types.items())})
            # time in ms
            df.time = df.time * 1000
            data.append(df)
    log('[CSV] converted to list of dataframes', 20)
    return data


def data_csv_to_hdf5(path_trajectory_data, process_time=True, **kwargs):
    """
    Read raw csv data
    Structure every individual trajectory in a pandas dataframe and append to a list object comprising all trajectories

    Parameters
    ----------

    path_trajectory_data: path to file with csv file of pNEUMA trajectory data
    Note that the conversion is dependent on the chosen format of LUTS lab (EPFL) to share pNEUMA trajectory data
    in_memory:
    process_time: print elapsed time
    **kwargs: additional parameters for init_hdf function

    Returns
    ----------
    return: None, data is written to HDF5 instance on disk. Log message added to corresponding folder.
    """
    tic = time.time()
    if sys.platform == "win32":
        split_name = path_trajectory_data.rsplit("\\")[-1].rsplit('.')[0]
    else:
        split_name = path_trajectory_data.rsplit('/')[-1].rsplit('.')[0]
    data_info = split_name.rsplit('_')
    # Nested info: date, drone, time
    data_name = f'day{data_info[0]}/{data_info[1]}/time{data_info[2]}'
    hdf_fn = init_hdf(**kwargs)
    tr_group = f'/{data_name}'
    csv.field_size_limit(maxInt)
    print("Start reading csv…")
    #  Create new table in HDF5 with 'track_id', 'type', 'traveled_d' and 'speed'
    hdf_id = tr_group + '/all_id'
    val = csv.reader(open(path_trajectory_data, 'r'), delimiter=';')
    header = next(val)
    header = [i.strip() for i in header]
    data = pd.DataFrame(columns=header[:4])
    df_all = []
    fps = 0
    for line in tqdm(val):
        data.loc[len(data)] = line[:4]
        if round(float(line[9]) * 1000) % 40 > 0.0001:  # row 9 is time column
            fps = 1
        _ = line.pop()
        tmp_df = pd.DataFrame({j: pd.Series(line[(i + 4)::6], dtype='float64') for i, j in enumerate(header[4:])})
        tmp_df.insert(0, 'track_id', [int(line[0])] * len(tmp_df))
        df_all.append(tmp_df)
    data = data.astype({'track_id': 'int64', 'type': 'category', 'traveled_d': 'float64', 'avg_speed': 'float64'})
    #print(data.head())
    #print(data.dtypes)
    data.to_hdf(hdf_fn, key=hdf_id, format='table', data_columns=['track_id', 'type'], append=False, mode='a')
    toc1 = time.time()
    if process_time:
        print(f'/all_id table written to HDF5, took {toc1 - tic} sec')
    val = pd.concat(df_all, axis=0)
    if fps == 0:
        # time in ms
        val.loc[:, 'time'] = val.loc[:, 'time'].values * 1000
        val = val.astype({'time': 'int64'})
    elif fps == 1:
        print("Check time rate, not equal to stated rate of 40 ms")
        val.loc[:, 'time'] = val.loc[:, 'time'].values * 1000
        val = val.astype({'time': 'int64'})
    #print(val.dtypes)
    #print('concat done', time.time() - toc1)
    val.loc[:, 'x'], val.loc[:, 'y'] = project_improved(val.lon.values, val.lat.values)
    #print('projection done', time.time() - toc1)
    hdf_traj = tr_group + '/original_trajectories'
    #print(val.head())
    val.to_hdf(hdf_fn, key=hdf_traj, format='table', data_columns=['track_id', 'time'], mode='a')
    path_dict = get_path_dict()
    group_id = group_id_from_path(path_trajectory_data)
    path_dict['hdf_path'] = hdf_fn
    path_dict['groups'][group_id] = data_name
    path_dict['current_resample_step'][group_id] = 40
    write_pickle(path_dict, 'path_dict', path=path_data)
    toc2 = time.time()
    if process_time:
        print(f'/original_trajectories table written to HDF5, took {toc2 - toc1} sec')
        print(f'Total time: {toc2 - tic} sec')
    log(f'[CSV] GROUP: {data_name} converted to hdf5', 20)
    return hdf_fn, data_name


def new_dfs(ldf, group_id, bearing=True, resample=True, step=1000):
    tic = time.time()
    new_ldf = []
    path_dict = get_path_dict()
    key = path_dict['groups'][group_id]
    for df in tqdm(ldf):
        if bearing:
            df = add_bearing(df)
        if resample:
            path_dict['current_resample_step'][group_id] = int(step)
            write_pickle(path_dict, 'path_dict', path=path_data)
            df = resample_time_rate(df, time_step=step)
        new_ldf.append(df)
    df_all = pd.concat(new_ldf, axis=0)
    df_all.reset_index(drop=True, inplace=True)
    path_dict = get_path_dict()
    hdf_fn = path_dict['hdf_path']
    hdf_traj = key + '/adjusted_trajectories'
    df_all.to_hdf(hdf_fn, key=hdf_traj, format='table', data_columns=['track_id', 'time'], mode='a')
    toc = time.time()
    print(f'{toc - tic} sec')


def resample_time_rate(df, time_step=1000):  # 0.04 s base sample rate
    """
    Re-sampling is very simple and basic for this data set. Since the location of every data point is not using GPS but
    referenced to accurately defined locations in the study area of the video footage it can be assumed that every
    data point is validly recorded. Only when a rate is chosen which is not a multiple of the base rate (0.04 s) the
    procedure becomes more complex, possibly using a form of interpolation.

    :param df: dataframe with the base sample rate
    :param time_step: rows of dataframe selected according to chose time step (in ms) [multiple of 40 ms]
    :return: DataFrame with chosen sample rate (subset of original dataframe)
    """
    start_time = int(round(df.time[0]))
    max_time = int(round(max(df.time)))
    select_time = [i for i in range(start_time, max_time, int(time_step))]
    dfn = df[df.time.isin(select_time)]
    dfn.reset_index(inplace=True)
    dfn.rename(columns={'index': '_oid'}, inplace=True)
    return dfn


def add_bearing(df):
    """
    Determine bearing for every data point t(i) using consecutive point at t(i+1). The last point takes the value
    of its predecessor. (No added value in predicting a value for this last point) If vehicle is static for some time
    the value of the predecessor is reused until a new bearing can be calculated, when no predecessor exists the first
    point that results in a valid bearing calculation is used and all the points in between are assigned the same value

    North, east, south, west => 0, 90, 180, 270, 360 degrees (ClockWise)

    Note: Determining bearing is done for data set with the base sample rate, gives best approximation of direction of
    travel. A smoothing can afterwards be performed to filter out possible unrealistic direction shifts.
    (Care is needed)

    :param df: DataFrame with latitude-longitude coordinates
    :return: DataFrame with new bearing column specifying the direction of each data point

    TODO: accuracy vs precision, smooth bearing calculation on difference between successive lat-lon data points
    TODO: five decimals are accurate enough, 0.00001 difference is around 1.1 m
    """
    bearing = []
    tr = [tuple(i) for i in zip(df.lat, df.lon)]
    r = 0
    for i, j in enumerate(tr):
        if i < r:
            continue
        if len(tr) == 1:
            bearing.append(999)  # Static vehicle
        elif i < len(tr) - 1:
            A = j
            B = tr[i + 1]
            if A == B:
                if i == 0:
                    r = 2
                    while A == tr[r] and r + 1 < len(tr):
                        r = r + 1
                    if A == tr[r]:
                        comp = 999
                        ls_comp = [round(comp)] * r
                        bearing.extend(ls_comp)  # Static vehicle
                    else:
                        C = tr[r]
                        comp_1 = cpb.calculate_initial_compass_bearing(A, C)
                        ls_comp = [round(comp_1, 1)] * r
                        bearing.extend(ls_comp)
                else:
                    comp_2 = bearing[i - 1]
                    bearing.append(round(comp_2, 1))
            else:
                comp_1 = cpb.calculate_initial_compass_bearing(A, B)
                bearing.append(round(comp_1, 1))
        elif len(tr) > 1:  # Last point same as predecessor
            try:
                comp = bearing[i - 1]
                bearing.append(round(comp, 1))
            except IndexError:
                print(i, df)
                print(bearing)
        else:
            bearing.append(999)  # Static vehicle
    df = df.assign(bearing=bearing)
    return df


def main(path_to_csv=None, filename=init_name):
    if path_to_csv is None:
        try:
            list_df = read_pickle(filename, path=path_data)
            return list_df
        except FileNotFoundError:
            ms = f'No file with the name {filename} exists at location {path_data}'
            print(ms)
            log(ms, 20)
    else:
        try:
            list_df = data_csv_to_list_dfs(path_to_csv)
            write_pickle(list_df, filename, path=path_data)
            return list_df
        except FileNotFoundError:
            ms = f'No csv file found, check if path {path_to_csv} is correct'
            print(ms)
            log(ms, 20)


def io_data(group_id):
    data_path = get_data_dict()
    try:
        data = data_path[group_id]
    except KeyError:
        group_id = add_data_path()
        data = data_path[group_id]
    hdf_name, group_path = data_csv_to_hdf5(data)
    return hdf_name, group_path
