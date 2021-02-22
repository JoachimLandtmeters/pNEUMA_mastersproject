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

from pneumapackage.settings import init_name, hdf_name, crs_pneuma
from pneumapackage.__init__ import path_data, write_pickle, read_pickle
from pneumapackage.logger import log
from pneumapackage.network import project_point
import pneumapackage.compassbearing as cpb

import os
import sys
from pathlib import Path
import time

import pandas as pd
import h5py
import csv
from tqdm import tqdm
import pickle

maxInt = sys.maxsize  # to read problematic csv files with various numbers of columns
start = time.time()


# HDF5 will be used to structurally store the trajectory data
# Using this structure enable a lot of possibilities to only read in specific numeric data into memory so
# programs will not be slowed down by a large data sets taking up active memory
# The workflow currently proposed does operations very sequentially therefore this data format is a good choice
#
# Step 1: Create general hdf5 structure at the 'path_data' location
# Step 2: When adding specific data a group is added with the corresponding metadata and numeric values


def init_hdf(path=path_data, name=hdf_name, reset=False):
    hdf_fn = path + name
    if not os.path.exists(hdf_fn) or reset:
        with h5py.File(hdf_fn, 'w') as s:
            s.create_group('info')
            s['info'].attrs['pNEUMA'] = 'store pNEUMA data in HDF5'
            s['info'].attrs['crs'] = crs_pneuma
    return hdf_fn


def get_from_hdf(hdf, key_id=None, key_tr=None, result='all', select_id=None, select_all=None):
    """
    Get data from hdf5

    :param hdf: path to hdf5 file
    :param key_id: group with dataframe of all track ids
    :param key_tr: group with dataframe of all trajectory points
    :return: df with all ids, concatenated dataframe of all trajectories,  list of dataframes with trajectories
    """
    assert result in ['all', 'ids', 'df_all', 'list']
    df_id, df_all, ldf = None, None, None
    if key_id is not None:
        df_id = pd.read_hdf(hdf, key=key_id, mode='r', where=select_id)
    if key_tr is not None:
        df_all = pd.read_hdf(hdf, key=key_tr, mode='r', where=select_all)
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


def get_path_dict(path=path_data, name='path_dict', reset=False):
    if not reset:
        try:
            path_dict = read_pickle(name, path=path)
            return path_dict
        except FileNotFoundError:
            path_dict = {'hdf_path': 0, 'groups': {}}
            write_pickle(path_dict, 'path_dict', path=path_data)
            return path_dict
    else:
        path_dict = {'hdf_path': 0, 'groups': {}}
        write_pickle(path_dict, 'path_dict', path=path_data)
        return path_dict


def get_hdf_path(**kwargs):
    path_dict = get_path_dict(**kwargs)
    hdf_path = path_dict['hdf_path']
    return hdf_path


def get_group(group_number=None, **kwargs):
    path_dict = get_path_dict(**kwargs)
    groups = path_dict['groups']
    if group_number is not None:
        return groups[group_number]
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
    csv.field_size_limit(sys.maxsize)
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
    log('csv-file with trajectory data converted to list of dataframes', 20)
    return data


def data_csv_to_hdf5(path_trajectory_data, group_number, data_name=None, process_time=True, **kwargs):
    """
    Read raw csv data
    Structure every individual trajectory in a pandas dataframe and append to a list object comprising all trajectories

    :param path_trajectory_data: path to file with csv file of pNEUMA trajectory data
    Note that the conversion is dependent on the chosen format of LUTS lab (EPFL) to share pNEUMA trajectory data
    :param data_name: name of data set to assign to group key in HDF5,
    if none the last piece of the path string is taken
    :return: List of DataFrames, with every index an individual trajectory
    """
    tic = time.time()
    if data_name is None:
        split_name = path_trajectory_data.rsplit('/')[-1].rsplit('.')[0]
        data_info = split_name.rsplit('_')
        # Nested info: date, drone, time
        data_name = f'day{data_info[0]}/{data_info[1]}/time{data_info[2]}'
    hdf_fn = init_hdf(**kwargs)
    tr_group = f'/{data_name}'
    path_dict = get_path_dict()
    path_dict['hdf_path'] = hdf_fn
    path_dict['groups'][group_number] = data_name
    write_pickle(path_dict, 'path_dict', path=path_data)
    csv.field_size_limit(sys.maxsize)
    data_file = open(path_trajectory_data, 'r')
    data_reader = csv.reader(data_file)
    hdf_id = tr_group + '/all_id'
    tr_id = pd.read_csv(path_trajectory_data, sep=';', header=None, skiprows=[0], usecols=[0, 1, 2, 3],
                        dtype={0: 'int64', 1: 'category', 2: 'float16', 3: 'float16'})
    tr_id.columns = ['track_id', 'type', 'traveled_d', 'avg_speed']
    tr_id.to_hdf(hdf_fn, key=hdf_id, format='table', data_columns=['track_id', 'type'], append=False, mode='a')
    toc1 = time.time()
    if process_time:
        print(f'{toc1 - tic} sec')
    col_line = 0
    data = []
    for row in tqdm(data_reader):
        cl = [elem for elem in row[0].split("; ")]
        if col_line == 0:
            cl_type = [e for e in cl if e not in ['track_id', 'type', 'traveled_d', 'avg_speed']]
        else:
            cl.pop()
            df = pd.DataFrame({j: pd.Series(cl[(i + 4)::6], dtype='float64') for i, j in enumerate(cl_type)})
            # time in ms
            df.time = [int(round(i * 1000)) for i in df.time.to_list()]
            df.insert(0, 'track_id', pd.Series([cl[0]] * len(df), dtype='int64'))
            data.append(df)
        col_line += 1
    toc2 = time.time()
    if process_time:
        print(f'{toc2 - toc1} sec')
    df_all = pd.concat(data, axis=0)
    x, y = zip(*project_point(list(zip(df_all.lon, df_all.lat))))
    df_all = df_all.assign(x=x, y=y)
    hdf_traj = tr_group + '/original_trajectories'
    df_all.to_hdf(hdf_fn, key=hdf_traj, format='table', data_columns=['track_id', 'time'], mode='a')
    toc3 = time.time()
    if process_time:
        print(f'{toc3 - toc2} sec')
        print(f'Total time: {toc3 - tic} sec')
    log('csv-file with trajectory data converted to hdf5', 20)


def new_dfs(ldf, key, bearing=True, resample=True, step=1000):
    tic = time.time()
    new_ldf = []
    for df in tqdm(ldf):
        if bearing:
            df = add_bearing(df)
        if resample:
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

# df_street = pd.read_csv('street_information.csv', header=0)
