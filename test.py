"""
test file for code snippets and intermediate results
"""
from pneumapackage.settings import *
import pneumapackage.network as cn
import pneumapackage.mapmatching as mm
import pneumapackage.virtualdetector as md
import pneumapackage.compute as cp
import pneumapackage.compassbearing as cpb
from pneumapackage.__init__ import read_pickle, write_pickle, path_data, path_results
import pneumapackage.iodata as rd

import h5py
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import geopy.distance as geodist
import leuvenmapmatching.util.dist_euclidean as distxy
import leuvenmapmatching.util.dist_latlon as distlatlon
import osmnx as ox
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import collections
from tqdm.contrib import tenumerate
from tqdm import tqdm
import time

"""
Create network from OSM given a pre-specified bounding box for the area
"""


def test_network(bbox=bb_athens, path=path_data, osm_filter=adj_filter, save_shp=True, save_pickle=True, reload=False):
    tic = time.time()
    print('Start: …load network ')
    if reload:
        network = cn.CreateNetwork(bounding_box=bbox, custom_filter=osm_filter)
        _ = network.network_dfs()
        if save_pickle:
            write_pickle(network, 'network', path)
    else:
        try:
            network = read_pickle('network', path)
        except FileNotFoundError:
            network = cn.CreateNetwork(bounding_box=bbox, custom_filter=osm_filter)
            _ = network.network_dfs()
            if save_pickle:
                write_pickle(network, 'network', path)

    if save_shp:
        network.save_graph_to_shp()
        print('Shapefiles stored')
    toc = time.time()
    print(f'Network loaded, took {toc - tic} sec')
    return network


data_p = path_data + zone_all
gr_number = 1


def test_createHDFdata(data=data_p, group_number=gr_number, reset=False):
    tic = time.time()
    print('Start: …create HDF dataset')
    if reset:
        print('Reset HDF and create new instance…')
    rd.data_csv_to_hdf5(data, group_number, reset=reset)
    hdfp = rd.get_hdf_path()
    group_p = rd.get_group(gr_number)

    toc = time.time()
    print(f'HDF dataset loaded, took {toc - tic} sec')
    return hdfp, group_p


def get_hdf_names(number, reset=False):
    try:
        hdfp = rd.get_hdf_path()
        group_p = rd.get_group(number)
    except:
        hdfp, group_p = test_createHDFdata(group_number=number, reset=reset)
    pid = group_p + '/all_id'
    porg = group_p + '/original_trajectories'
    padj = group_p + '/adjusted_trajectories'
    return hdfp, group_p, pid, porg, padj


hdfp, group_p, pid, porg, padj = get_hdf_names(gr_number)
"""
Map-match trajectory data to network
"""


def test_load_traj(hdf_path=hdfp, group_path=group_p, path_id=pid, path_original=porg,
                   path_adjusted=padj, redo=False):
    tic = time.time()
    print('Start: …load trajectories (resampled)')
    if redo:
        print('Create trajectory dataset and save the file in HDF5')
        ldf = rd.get_from_hdf(hdf_path, path_id, path_original, result='list')
        rd.new_dfs(ldf, group_path)
        ldf = rd.get_from_hdf(hdf_path, path_id, path_adjusted, result='list')
    else:
        try:
            ldf = rd.get_from_hdf(hdf_path, path_id, path_adjusted, result='list')
        except KeyError:
            print('Create trajectory dataset and save the file in HDF5')
            ldf = rd.get_from_hdf(hdf_path, path_id, path_original, result='list')
            rd.new_dfs(ldf, group_path)
            ldf = rd.get_from_hdf(hdf_path, path_id, path_adjusted, result='list')

    toc = time.time()
    print(f'Trajectories loaded, took {toc - tic}')
    return ldf


# The initial values for the map-matching algorithm are chosen


def test_map_matching(rematch=False, max_distance=10, hdf_path=hdfp, group_path=group_p, save_shp=True,
                      network_obj=None, traj_obj=None, match_latlon=True, path=path_data):
    tic = time.time()
    print('Start: …load map matching')
    # load network
    network = network_obj
    if network_obj is None:
        network = test_network()
    try:
        with h5py.File(hdf_path, 'r') as s:
            tag_match = s[group_path].attrs['tag_mapmatching']
        if tag_match != network.mm_id:
            rematch = True
    except KeyError:
        rematch = True

    def load_input(max_d=max_distance):
        max_init_dist = int(max(network.network_edges['length'])) + 1
        print(f'Initial distance: {max_init_dist} m, maximum distance (start): {max_d} m')

        # load unmatched trajectories
        traj_unm = traj_obj
        if traj_obj is None:
            traj_unm = test_load_traj()
        return max_init_dist, max_d, traj_unm

    if not rematch:
        try:
            dfmatch_all = rd.get_from_hdf(hdf_path, key_id=group_path + '/all_id', key_tr=group_path + '/mm_1s',
                                          result='df_all')
        except (KeyError, TypeError):
            max_init, max_d, traj_unmatched = load_input(max_distance)
            tmm = mm.MapMatching(traj_unmatched, network, max_init=max_init, max_d=max_d,
                                 match_latlon=match_latlon)
            match_all = tmm.match_variable_distance(progress=False)
            dfmatch_all = pd.concat(match_all)
            dfmatch_all.to_hdf(hdf_path, key=group_path + '/mm_1s', format='table', mode='a', append=False,
                               data_columns=['track_id', 'time', 'n1', 'n2', '_id'])
            dt = datetime.datetime.now()
            tag = int(dt.strftime('%Y%m%d%H%M'))
            with h5py.File(hdf_path, 'a') as s:
                s[group_path].attrs['tag_mapmatching'] = tag
            network.add_mapmatch_tag(tag)
            write_pickle(network, 'network', path)
    else:
        max_init, max_d, traj_unmatched = load_input(max_distance)
        tmm = mm.MapMatching(traj_unmatched, network, max_init=max_init, max_d=max_d,
                             match_latlon=match_latlon)
        match_all = tmm.match_variable_distance(progress=False)
        dfmatch_all = pd.concat(match_all)
        dfmatch_all.to_hdf(hdfp, key=group_path + '/mm_1s', format='table', mode='a', append=False,
                           data_columns=['track_id', 'time', 'n1', 'n2', '_id'])
        dt = datetime.datetime.now()
        tag = int(dt.strftime('%Y%m%d%H%M'))
        with h5py.File(hdf_path, 'a') as s:
            s[group_path].attrs['tag_mapmatching'] = tag
        network.add_mapmatch_tag(tag)
        write_pickle(network, 'network', path)

    if len(network.used_network) == 0:
        used_network = network.network_edges[network.network_edges['_id'].isin(dfmatch_all['_id'])]
        network.add_used_network(used_network)
        write_pickle(network, 'network', path)
    else:
        network = read_pickle('network', path)

    if save_shp:
        fn = path + '/shapefiles/used_network'
        network.used_network.loc[:, ~network.used_network.columns.isin(['edge'])].to_file(filename=fn)
        print('Shapefiles stored')

    toc = time.time()
    print(f'Map-matched trajectories loaded, took {toc - tic} sec')
    return {'tracks': dfmatch_all, 'network': network}


def test_matched_LineTrajectories(group_path=group_p, hdf_path=hdfp, traj_obj=None, selection=None, reload=False,
                                  **kwargs):
    tic = time.time()
    print('Start: …load matched trajectories')
    if reload:
        traj_obj = test_map_matching(group_path=group_path, **kwargs)
        traj_matched = traj_obj['tracks']
        network = traj_obj['network']

        traj_match = mm.TransformTrajectories(traj_matched, network.used_network)
        traj_match.tracks_line.to_hdf(hdf_path, key=group_path + '/mm_line_1s', format='table', mode='a', append=False,
                                      data_columns=['time', 'u_match', 'v_match'])
        line_traj = traj_match.tracks_line
        if selection is not None:
            line_traj = traj_match.tracks_line.query(selection)
    else:
        try:
            line_traj = rd.get_from_hdf(hdf_path, key_id=group_path + '/all_id', key_tr=group_path + '/mm_line_1s',
                                        result='df_all', select_all=selection)
        except (KeyError, TypeError):
            if traj_obj is not None:
                assert isinstance(traj_obj, dict)
                assert 'tracks' and 'network' in traj_obj.keys()
                traj_matched = traj_obj['tracks']
                network = traj_obj['network']
            else:
                traj_obj = test_map_matching(group_path=group_path, **kwargs)
                traj_matched = traj_obj['tracks']
                network = traj_obj['network']

            traj_match = mm.TransformTrajectories(traj_matched, network.used_network)
            traj_match.tracks_line.to_hdf(hdf_path, key=group_path + '/mm_line_1s', format='table', mode='a', append=False,
                                          data_columns=['time', 'u_match', 'v_match'])
            line_traj = traj_match.tracks_line
            if selection is not None:
                line_traj = traj_match.tracks_line.query(selection)

    toc = time.time()
    print(f'Matched trajectories loaded, took {toc - tic} sec')
    return line_traj


lonlat = False


def get_lt(edges=pan_id, gdf=True, latlon=lonlat):
    tic = time.time()
    lt = test_matched_LineTrajectories(selection=f'u_match in {edges}')
    lt2 = test_matched_LineTrajectories(selection=f'v_match in {edges}')
    lt3 = pd.merge(lt.reset_index(), lt2.reset_index(), how='outer')
    lt3.set_index(['track_id', 'rid'], inplace=True)
    lt3.sort_index(inplace=True)
    if gdf:
        lt3 = mm.make_gdf(lt3, line=True, latlon=latlon)
    toc = time.time()
    print(f'Selection line trajectories loaded, took {toc-tic} sec')
    return lt3


"""
# Up until now we have list of dataframes, trajectories, with a column with the matched edge in the extracted OSM network
# The line trajectories can be used to count vehicles crossing specific locations in the network, keeping the individual
# information for more specific aggregations afterwards (augmented loop detector data)

# Place detectors on the edges in the used network and select specific edges
# --> using Qgis to select manually the needded edge ids

# Input parameters:
# - 20 m = width of detector edges, make sure they span the whole road
# - 10 m = distance from intersection
# - True = place double virtual loops
# - 1 m = loop distance
# - 2 = number of detectors on every link
"""
# tic = time.time()
len_det = 15
dfi = 10
ld = 1
n_det = 2
double_loops = False

network = test_network(save_shp=False)
try:
    det = read_pickle('detectors', path_data)
except FileNotFoundError:
    det = md.Detectors(network.used_network, len_det=len_det, dfi=dfi, ld=ld, n_det=n_det, double_loops=double_loops,
                       lonlat=lonlat, gdf_special=network.node_tags)
    write_pickle(det, 'detectors', path_data)
    det.detector_to_shapefile(double_loops=double_loops, folder='data')
    print('Shapefiles stored')

ds = det.detector_selection(pan_id)


def vehicle_crossings(d_gdf, gdf_traj, n_det=n_det, bearing_difference=90, lonlat=lonlat):
    tic = time.time()
    print('Start: …searching crossings')
    assert isinstance(gdf_traj, gpd.GeoDataFrame)
    c1, c2 = 'x', 'y'
    if lonlat:
        c1, c2 = 'lon', 'lat'
    # column multi-index
    col_names = [[f'cross_{c1}{i}', f'cross_{c2}{i}', f'rid{i}',
                  f'd{i}', f't{i}', f'v{i}'] for i in range(1, n_det + 1)]
    col_names = [item for sublist in col_names for item in sublist]
    col_index = pd.MultiIndex.from_product([d_gdf['_id'], col_names], names=['edge', 'detector'])
    if isinstance(gdf_traj.index, pd.MultiIndex):
        assert gdf_traj.index.names.index('track_id') == 0
        row_index = set(gdf_traj.index.get_level_values(0))
    else:
        row_index = set(gdf_traj['track_id'])
        gdf_traj.set_index(['track_id', 'rid'], inplace=True)
    p1 = list(zip(*(gdf_traj[f'{c1}_1'], gdf_traj[f'{c2}_1'])))
    p2 = list(zip(*(gdf_traj[f'{c1}_2'], gdf_traj[f'{c2}_2'])))
    if lonlat:
        line_dist = [round(geodist.distance(*xy).m, 3) for xy in zip(p1, p2)]
    else:
        line_dist = [round(distxy.distance(*xy), 3) for xy in zip(p1, p2)]
    gdf_traj['p1'] = p1
    gdf_traj['dist'] = line_dist
    df_result = pd.DataFrame(index=row_index, columns=col_index)
    df = gdf_traj
    for i, det_link in tqdm(d_gdf.iterrows(), total=d_gdf.shape[0]):  # Counting vehicles for every used edge
        df_bool_wm = df[['wm1', 'wm2']].values < bearing_difference
        df_wm = df[np.logical_and(df_bool_wm[:, 0], df_bool_wm[:, 1])]
        df_bool_edge = df_wm[['u_match', 'v_match']].values == det_link['_id']
        df_u = df_wm[df_bool_edge[:, 0]].index.to_list()
        df_v = df_wm[df_bool_edge[:, 1]].index.to_list()
        set_index = set(df_u + df_v)
        if len(set_index) == 0:
            continue
        df2 = df_wm.loc[set_index]
        for n in range(1, n_det + 1):
            df_search = df2['geometry'].values.intersects(det_link[f'det_edge_{n}'])
            df_intersect = df2[df_search].index.to_list()
            if not df_intersect:
                continue
            tid, rid = zip(*df_intersect)
            df_search_cross = df2[df_search]
            df_cross = df_search_cross['geometry'].values.intersection(det_link[f'det_edge_{n}'])
            df_cross = [(c.x, c.y) for c in df_cross]
            if lonlat:
                df_dist = np.array([round(geodist.distance(*xy).m, 3) for xy in zip(df_search_cross.p1, df_cross)])
            else:
                df_dist = np.array([round(distxy.distance(*xy), 3) for xy in zip(df_search_cross.p1, df_cross)])
            t, v = interpolate_crossing(df_search_cross, df_dist)
            df_c1, df_c2 = zip(*df_cross)
            df_result.loc[list(tid), ({det_link["_id"]}, f'rid{n}')] = list(rid)
            df_result.loc[list(tid), ({det_link["_id"]}, f'cross_{c1}{n}')] = df_c1
            df_result.loc[list(tid), ({det_link["_id"]}, f'cross_{c2}{n}')] = df_c2
            df_result.loc[list(tid), ({det_link["_id"]}, f'd{n}')] = df_dist
            df_result.loc[list(tid), ({det_link["_id"]}, f't{n}')] = t
            df_result.loc[list(tid), ({det_link["_id"]}, f'v{n}')] = v
    df_result.sort_index(inplace=True)
    df_result = df_result.transpose()
    toc = time.time()
    print(f'Finding crossings done, took {toc - tic} sec')
    return df_result


def cross_matrix_sparse(df_crossings):
    df_crossings = df_crossings.fillna(0)


def interpolate_crossing(df, p):
    assert 'time' in df.columns
    assert 'speed_1' and 'speed_2' in df.columns
    assert 'dist' in df.columns
    t = np.round(df.time.values - 1000 + p / df.dist.values * 1000)
    v = np.round((df.speed_2.values - df.speed_1.values) * (t - df.time.values + 1000)/1000 + df.speed_1.values, 3)
    return t, v


def get_traj_crossings(track, df_crossings):
    df_crossings = df_crossings[~df_crossings[track].isna()][track]
    return df_crossings


    # traj_t[f't_{det}'] = round(traj_match_values[m][idx][3] - 1000 + d1c / d12 * 1000)


# toc = time.time()
# print(toc - tic)
# print('Detectors loaded')

"""
# From Qgis the edges and detectors of the intersection are selected manually for usage in further analysis
# Detector index list:
# det_index = [1718,1560,766,1265,781,759,782,1510,1269,1267,1509]
"""
# det_index = [1718, 766, 1265, 781, 782, 1510, 1267, 1509]
# det_sel = det.detector_selection(index_list=det_index)

# Split up matching result, trajectories with matched edge and used network
# Add line trajectories


"""
# All needed steps are now done to determine the crossing of every individual vehicle
# Using the count function the crossings of every individual vehicle are stored together
# with additional useful information

# Input parameters:
# - detector specifications, see earlier
# - freq = 10000 ms, frequency for aggregation
# -
"""
"""
frequency = 1000

try:
    ta_traj = read_pickle(f'traffic_analysis_{int(frequency / 1000)}s', path_results)
except FileNotFoundError:
    ta_traj = cp.TrafficAnalysis(det_sel, traj_match.line_trajectories, traj_match.used_network, n_det=n_det,
                                 freq=frequency, double_loops=double_loops, dfi=dfi, loop_distance=ld)
    write_pickle(ta_traj, f'traffic_analysis_{int(frequency / 1000)}s', path_results)

print('Counts object loaded')

"""
# Visualize signal timings by plotting flows of incoming links, clearly show the case
"""

i1 = [781, 782]
i2 = [1718]
i3 = [1509, 1510]

j1 = [766]
j2 = [1267]
j3 = [1265]


def show_signal_timings(link, ta, **kwargs):
    ls = list(ta.traffic_counts['detectors']['index'])
    veh1 = ta.traffic_parameters[ls.index(link)]['vehicles_1']
    veh2 = ta.traffic_parameters[ls.index(link)]['vehicles_2']
    plt.figure(figsize=(12, 8))
    plt.plot(veh1, '-o', **kwargs)
    plt.grid(True)
    plt.title(f'Flow [veh/h] ({float(frequency / 1000)} sec time intervals)')
    plt.savefig(path_results + f'signal_timings_{link}_1.png', )
    plt.figure(figsize=(12, 8))
    plt.plot(veh2, '-o', **kwargs)
    plt.grid(True)
    plt.title(f'Flow [veh/h] ({float(frequency / 1000)} sec time intervals)')
    plt.savefig(path_results + f'signal_timings_{link}_2.png', )


"""
# The signal timings are a start but having the turning fractions is the goal. Depending on the frequency used
# the array shape will be different, more zeros with high frequency (same as trajectory data is maximum)
# The crossings of every individual vehicle are logged with their corresponding ID, therefore it is easy to obtain
# where every individual vehicle is going

# Making numpy arrays from the crossing times dataframes makes fast enumeration possible
# use: to_array --> np.stack(array, axis=0)
# """
