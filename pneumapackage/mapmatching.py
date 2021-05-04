# Link between theoretic network graph and trajectories
# Map-matching the trajectories to the underlying theoretical network
# Using Leuven Map-matching algorithm
# Start and End node of matched edge in dataframe of trajectories --> link between theoretical network and measured data
from pneumapackage.settings import bb_athens
from pneumapackage.__init__ import path_data, path_results, write_pickle, read_pickle
from pneumapackage.settings import *
import pneumapackage.compassbearing as cpb

import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import time
import numpy as np

import leuvenmapmatching as lm
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import *
from leuvenmapmatching import visualization as mmviz
import leuvenmapmatching.util.dist_latlon as distlatlon
import leuvenmapmatching.util.dist_euclidean as distxy
import geopy.distance as geodist

from tqdm import tqdm
from tqdm.contrib import tenumerate
from shapely.geometry import Point, LineString
from pyproj import Proj, transform
# import similaritymeasures
from collections import Counter
import rtree
import sys


logger = False

if logger:
    logger = lm.logger
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))


class MapMatching:

    # max initial distance is best chosen as the maximum length of an edge in the network graph
    # max(network_edges['length'])

    def __init__(self, list_traj, network_obj, max_init, max_d, match_latlon=True):
        self.list_traj = list_traj
        self.max_init = max_init
        self.max_d = max_d
        self.network_edges = network_obj.network_edges
        self.network_nodes = network_obj.network_nodes
        self.graph = network_obj.graph_latlon
        self.match_latlon = match_latlon
        self.map = self.make_map()

    # Create in memory graph for leuvenmapmatching from osmnx graph

    # Compatible path type for map matching

    def make_map(self, index_edges=True, rtree=True):
        map_con = InMemMap("athensmap", use_latlon=self.match_latlon, index_edges=index_edges,
                           use_rtree=rtree)  # Deprecated crs style in algorithm
        if self.match_latlon:
            for nid, rown in self.network_nodes.iterrows():
                map_con.add_node(int(rown.n1), (rown.lat1, rown.lon1))
            for eid, rowe in self.network_edges.iterrows():
                map_con.add_edge(rowe.n1, rowe.n2)
        else:
            for nid, rown in self.network_nodes.iterrows():
                map_con.add_node(int(rown.n1), (rown.y1, rown.x1))
            for eid, rowe in self.network_edges.iterrows():
                map_con.add_edge(rowe.n1, rowe.n2)
        return map_con

    def reset_map(self):
        self.map = self.make_map()

    def match_fixed_distance(self, list_index=None, logger=False, **kwargs):
        if logger:
            logger = lm.logger
            logger.setLevel(logging.DEBUG)
            logger.addHandler(logging.StreamHandler(sys.stdout))
        tic = time.time()
        traj_mov_match = []
        special_cases = []
        point_traj = self.list_traj
        if list_index is not None:
            point_traj = [j for i, j in enumerate(point_traj) if i in list_index]
        for i, j in tenumerate(point_traj):
            while True:
                try:
                    traj = map_matching(j, self.network_edges, self.map, self.max_init, self.max_d,
                                        latlon=self.match_latlon, **kwargs)
                    traj = traj.merge(self.network_edges[['_id', 'n1', 'n2']], how='left', on=['n1', 'n2'])
                    traj_mov_match.append(traj)
                    break
                except Exception:
                    special_cases.append(j)
                    break
        toc = time.time()
        print(f'{int(divmod(toc - tic, 60)[0])} min {int(divmod(toc - tic, 60)[1])} sec')
        return traj_mov_match, special_cases

    def match_variable_distance(self, list_index=None, logger=False, **kwargs):
        if logger:
            logger = lm.logger
            logger.setLevel(logging.DEBUG)
            logger.addHandler(logging.StreamHandler(sys.stdout))
        tic = time.time()
        traj_mov_match = []
        fails = []
        point_traj = self.list_traj
        if list_index is not None:
            point_traj = [j for i, j in enumerate(point_traj) if i in list_index]
        for j in tqdm(point_traj):
            dist_init = self.max_init
            dist = self.max_d
            fail = 0
            while True:
                try:
                    traj = map_matching(j, self.map, dist_init, dist, latlon=self.match_latlon, **kwargs)
                    traj = traj.merge(self.network_edges[['_id', 'n1', 'n2']], how='left', on=['n1', 'n2'])
                    traj_mov_match.append(traj)
                    break
                except Exception:
                    if fail < 3:
                        # print('Set distance higher:')
                        dist += 5
                        fail = fail + 1
                        # print(dist)
                        # print('Number of fails: ' + str(fail))
                    elif 2 < fail <= 10:
                        dist += 10
                        fail = fail + 1
                        # print('Set distance higher:')
                        # print(dist)
                        # print('Number of fails: ' + str(fail))
                    elif fail > 10:
                        dist += 10
                        dist_init += 50
                        fail += 1
                    # print('Still at list ' + str(i))
            fails.append(fail)
        toc = time.time()
        print(f'{int(divmod(toc - tic, 60)[0])} min {int(divmod(toc - tic, 60)[1])} sec')
        return traj_mov_match


class TransformTrajectories:

    def __init__(self, tr_match_all, network_obj):
        tic = time.time()
        # tr_match_all is a concatenated dataframe
        # set multiindex with track_id and row number
        tr_match_all.reset_index(drop=False, inplace=True)
        tr_match_all = tr_match_all.rename(columns={'index': 'rid'})
        tr_match_all.set_index(['track_id', 'rid'], inplace=True)
        idx_counts = tr_match_all.index.get_level_values(0).value_counts()
        sp = idx_counts[idx_counts == 1]
        mp = idx_counts[idx_counts != 1]
        self.tracks = tr_match_all
        self.tracks_point = tr_match_all[~tr_match_all._id.isna()].loc[mp.index].sort_index()
        self.tracks_single = tr_match_all.loc[sp.index].sort_index()
        self.tracks_nan = tr_match_all[tr_match_all._id.isna()]
        self.gdf_point = []
        self.gdf_line = []
        self.column_names = []
        self.network = network_obj.network_edges
        self.edges_counts = pd.DataFrame()
        self.tracks_line = self.wrong_match()
        toc = time.time()
        print(toc - tic)

    def wrong_match(self):
        tic = time.time()
        gdf_netw = self.network
        ldf_all = self.tracks_point.copy()
        # Do operations on total dataframe
        ldf_all = ldf_all.join(gdf_netw['bearing'], how='left', rsuffix='_edge', on='_id')
        diff = ldf_all[['bearing', 'bearing_edge']].values
        bearing_diff = [round(min(abs(diff[a][0] - diff[a][1]), 360 - abs(diff[a][0] - diff[a][1])), 1)
                        for a in range(0, len(ldf_all))]
        ldf_all['wrong_match'] = bearing_diff
        u_edge, v_edge, w_1, w_2 = [], [], [], []
        for j in tqdm(set(ldf_all.index.get_level_values(0))):
            df = ldf_all.loc[(j,), ['_id', 'wrong_match']].copy()
            # point dataset with nodes of matched edge, this adds column to all original dataframes (chained assignment)
            # making line dataset --> always start and end point --> last point has no successive point --> -1
            u_edge.extend(df['_id'].values[:-1])
            v_edge.extend(df['_id'].values[1:])
            w_1.extend(df['wrong_match'].values[:-1])
            w_2.extend(df['wrong_match'].values[1:])
        print('end of loop')
        ldf_start = ldf_all.drop(ldf_all.groupby(level=0).tail(1).index)
        ldf_end = ldf_all.drop(ldf_all.groupby(level=0).head(1).index)
        ldf_end.set_index(ldf_start.index, inplace=True)
        ldf_start = ldf_start.assign(u_match=u_edge, v_match=v_edge, wm1=w_1, wm2=w_2)
        ldf_start.drop(['_id', 'n1', 'n2', 'wrong_match', 'time'], axis=1, inplace=True)
        ldf_line = ldf_start.join(ldf_end[['lat', 'lon', 'speed', 'lon_acc', 'lat_acc', 'time', 'x', 'y',
                                           'bearing']], lsuffix='_1', rsuffix='_2')
        p1 = list(zip(*(ldf_line[f'lat_1'], ldf_line[f'lon_1'])))
        p2 = list(zip(*(ldf_line[f'lat_2'], ldf_line[f'lon_2'])))
        line_distlatlon = [round(distlatlon.distance(*xy), 3) for xy in tqdm(zip(p1, p2), total=len(p1))]
        p1 = list(zip(*(ldf_line[f'y_1'], ldf_line[f'x_1'])))
        p2 = list(zip(*(ldf_line[f'y_2'], ldf_line[f'x_2'])))
        line_distyx = [round(distxy.distance(*xy), 3) for xy in tqdm(zip(p1, p2), total=len(p1))]
        ldf_line['line_length_latlon'] = line_distlatlon
        ldf_line['line_length_yx'] = line_distyx
        print('Line length column added')
        toc = time.time()
        print(toc - tic)
        return ldf_line

    def make_point_trajecories(self):
        ldf = self.tracks_point.copy()
        if isinstance(ldf, pd.DataFrame):
            ldf = make_gdf(ldf)
        self.gdf_point = ldf

    def make_line_trajectories(self):
        ldf = self.tracks_line.copy()
        if isinstance(ldf, pd.DataFrame):
            ldf = make_gdf(ldf, line=True)
        self.gdf_line = ldf

    def select_rows(self, segment_index=None):
        gdf_list = self.tracks_point
        gdf_netw = self.network
        if segment_index is None:
            segment_index = gdf_netw._id.to_list()
        traj_eval = []
        for traj in tqdm(gdf_list):
            tr = traj.drop(['lon', 'lat'], axis=1)
            tr_first = tr.drop_duplicates('_id', keep='first')
            idx_first = list(tr_first.index)
            tr_first = pd.merge(tr_first, gdf_netw[['_id', 'lon1', 'lat1', 'length']].loc[segment_index]
                                , how='left', on=['_id'])
            tr_first = tr_first.rename(columns={'lon1': 'lon', 'lat1': 'lat'})
            tr_first = tr_first.assign(index=idx_first)
            tr_last = tr.drop_duplicates('_id', keep='last')
            idx_last = list(tr_last.index)
            tr_last = pd.merge(tr_last, gdf_netw[['_id', 'lon2', 'lat2', 'length']].loc[segment_index]
                               , how='left', on=['_id'])
            tr_last = tr_last.rename(columns={'lon2': 'lon', 'lat2': 'lat'})
            tr_last = tr_last.assign(index=idx_last)
            tr_sel = pd.concat([tr_first, tr_last])
            tr_sel = tr_sel.sort_values(by='index')
            df = traj.loc[idx_first + idx_last]
            df = df.sort_index()
            traj_eval.append([tr_sel, df])
        return traj_eval

    """
    def evaluation_measures(self):
        df_path = []
        df_match = []
        for i, j in tenumerate(self.select_rows()):
            a = converting_path_to_xy(j[1])
            b = converting_path_to_xy(j[0])
            df_path.append(a)
            df_match.append(b)
        dist_frech_cut = []
        dist_frech_full = []
        arc_length_diff_cut = []
        arc_length_diff_full = []
        tracked_vehicle = []
        mode = []
        for i, j in tenumerate(df_path):
            tracked_vehicle.append(j['track_id'].values[0])
            mode.append(j['type'].values[0])
            p = j.loc[:, ['x', 'y']]
            q = df_match[i].loc[:, ['x', 'y']]
            if len(j) < 3:
                dist_frech_cut.append(0)
                arc_length_diff_cut.append(0)
                d2 = similaritymeasures.frechet_dist(p.values, q.values)
                dist_frech_full.append(d2)
                l_p_f = similaritymeasures.get_arc_length(p.values)
                l_p2 = l_p_f[0]
                l_m_f = similaritymeasures.get_arc_length(q.values)
                l_m2 = l_m_f[0]
                arc_length_diff_full.append(round(abs(l_p2 - l_m2), 3))
                continue
            d1 = similaritymeasures.frechet_dist(p.values[1:-1], q.values[1:-1])
            d2 = similaritymeasures.frechet_dist(p.values, q.values)
            l_p = similaritymeasures.get_arc_length(p.values[1:-1])
            l_p1 = l_p[0]
            l_m = similaritymeasures.get_arc_length(q.values[1:-1])
            l_m1 = l_m[0]
            l_p_f = similaritymeasures.get_arc_length(p.values)
            l_p2 = l_p_f[0]
            l_m_f = similaritymeasures.get_arc_length(q.values)
            l_m2 = l_m_f[0]
            dist_frech_full.append(d2)
            dist_frech_cut.append(d1)
            arc_length_diff_cut.append(round(abs(l_p1 - l_m1), 3))
            arc_length_diff_full.append(round(abs(l_p2 - l_m2), 3))
        evaluation = {'ID': tracked_vehicle, 'type': mode,
                      'Frechet_distance': dist_frech_full, 'Frechet_distance_cut': dist_frech_cut,
                      'Length_difference': arc_length_diff_full, 'Length_difference_cut': arc_length_diff_cut}
        evaluation = pd.DataFrame(evaluation)
        return evaluation
     

    def distance_point_to_matched_edge(self):
        distances = {'ID': [], 'type': [], 'length_traj': [], 'max_distance': [], 'median_distance': [],
                     'mean_distance': [],
                     '99_percentile': [], 'length_diff': [], 'length_diff_rel': []}  # , 'frechet_distance': []}
        list_distances_traj = []
        for ind, traj in tenumerate(self.point_trajectories):
            distances['ID'].append((ind, traj['track_id'].values[0]))
            distances['type'].append(traj['type'].values[0])
            distances['length_traj'].append(len(traj))
            dist = []
            mapped_length = 0
            traj_val = traj[['lon', 'lat']].values
            xy_crds = converting_path_to_xy(traj)
            p_xy = xy_crds[['x', 'y']].values
            path_length = similaritymeasures.get_arc_length(p_xy)
            # print(path_length[0])
            traj_match = traj.rename(columns={'N1_match': 'N1', 'N2_match': 'N2'})
            match_df = pd.merge(traj_match[['edge']], self.used_network[['N1', 'lat1', 'lon1', 'N2', 'lat2', 'lon2',
                                                                         'length', 'edge']], how='left', on=['edge'])
            match_val = match_df[['lat1', 'lon1', 'lat2', 'lon2', 'edge', 'length']].values
            # q_1 = [xy for xy in zip(match_df.Lat1.values, match_df.Long1)]
            # idx = [i for i in range(len(q_1)) if q_1[i] != q_1[i-1]]
            # q_1 = list(match_df[['Lat1', 'Long1']].loc[idx].values)
            # if len(q_1) < 1:  # Interpolated points have to be appended
            #    q_1 = [0]
            # q_1 = []
            for row in range(0, len(traj)):
                p = (traj_val[row][1], traj_val[row][0])  # Lat-lon order
                s1 = (match_val[row][0], match_val[row][1])
                s2 = (match_val[row][2], match_val[row][3])
                d, pi, ti = lm_dist.distance_point_to_segment(p, s1, s2)
                dist.append(d)
                if row == 0:
                    mapped_length += match_val[row][5] * (1 - ti)
                    # q_1.append(pi)
                elif row == len(traj) - 1:
                    mapped_length += match_val[row][5] * ti
                    # q_1.append(pi)
                elif 0 < row and match_val[row][4] != match_val[row - 1][4]:
                    # q_1.append(pi)
                    mapped_length += match_val[row][5]
            if match_val[len(traj) - 2][4] == match_val[len(traj) - 1][4]:
                mapped_length -= match_val[len(traj) - 1][5]
            # print(mapped_length)
            diff_len = abs(path_length[0] - mapped_length)
            list_distances_traj.append(dist)
            # q_xy = converting_list_to_xy(q_1)
            # d_fr = similaritymeasures.frechet_dist(p_xy, q_xy.values)
            distances['max_distance'].append(max(dist))
            distances['median_distance'].append(np.median(dist))
            distances['mean_distance'].append(np.mean(dist))
            distances['99_percentile'].append(np.percentile(dist, 99))
            distances['length_diff'].append(diff_len)
            distances['length_diff_rel'].append(diff_len / path_length[0])
            # distances['frechet_distance'].append(d_fr)
        distances = pd.DataFrame(distances)
        return distances, list_distances_traj
     """

    def fraction_wrongly_matched(self, threshold_angle=45):
        wrongly_matched = {'id': [], 'type': [], 'wrong_1': [], 'average_speed_1': [], 'bool_w1': [], 'wrong_2': [],
                           'average_speed_2': [], 'bool_w2': [], 'wrong_both': [], 'bool_wb': [],
                           'length_trajectory': []}
        for ind, traj in tenumerate(self.tracks_line):
            wrongly_matched['id'].append((ind, traj['track_id'].values[0]))
            wrongly_matched['type'].append(traj['type'].values[0])
            wm = traj['wrong_match'].values
            speeds = traj[['speed_x', 'speed_y']].values
            w1 = [speeds[i][0] for i, j in enumerate(wm) if j[0] > threshold_angle]
            w2 = [speeds[i][1] for i, j in enumerate(wm) if j[1] > threshold_angle]
            wb = [i for i, j in enumerate(wm) if j[0] > threshold_angle and j[1] > threshold_angle]
            wrongly_matched['wrong_1'].append(round(len(w1) / len(traj) * 100, 1))
            wrongly_matched['wrong_2'].append(round(len(w2) / len(traj) * 100, 1))
            wrongly_matched['wrong_both'].append(round(len(wb) / len(traj) * 100, 1))
            if w1:
                wrongly_matched['bool_w1'].append(True)
            else:
                wrongly_matched['bool_w1'].append(False)
            if w2:
                wrongly_matched['bool_w2'].append(True)
            else:
                wrongly_matched['bool_w2'].append(False)
            if wb:
                wrongly_matched['bool_wb'].append(True)
            else:
                wrongly_matched['bool_wb'].append(False)
            if w1:
                wrongly_matched['average_speed_1'].append(np.mean(w1))
            else:
                wrongly_matched['average_speed_1'].append(0)
            if w2:
                wrongly_matched['average_speed_2'].append(np.mean(w2))
            else:
                wrongly_matched['average_speed_2'].append(0)
            wrongly_matched['length_trajectory'].append(len(traj))
        wrongly_matched = pd.DataFrame(wrongly_matched)
        return wrongly_matched

    def interesting_edges(self, sort=False):
        list_traj = self.tracks_point
        gdf_netw = self.network
        ed = []
        nan_ed = []
        for k, l in enumerate(tqdm(list_traj)):
            t_edges = pd.Series(list(zip(l.N1_match, l.N2_match)))
            t_edges.drop_duplicates(inplace=True)
            for i, j in enumerate(t_edges.to_list()):
                # if not j[0] > 0:
                # print(l[['Type','u_match','v_match']])
                # nan_ed.append(k)
                # traj_line_match[k].plot(ax=ax)
                if j[0] > 0:
                    ed.append(j)
        ed = Counter(ed)
        ed_pd = pd.DataFrame.from_dict(dict(ed), orient='index', columns=['counts'])
        ed_pd = ed_pd.reset_index()
        ed_pd = ed_pd.rename(columns={'index': 'edge'})
        int_ed = pd.merge(gdf_netw, ed_pd, how='inner', on=['edge'])
        if sort:
            int_ed = int_ed.sort_values(['counts', 'highway', 'length'], ascending=False)
            int_ed.reset_index(inplace=True, drop=True)
        self.edges_counts = int_ed
        return int_ed


def map_matching(traj, map, max_init, max_d, std_noise=1, progress=True, plot_match=False, latlon=True):
    matcher = DistanceMatcher(map, max_dist_init=max_init, max_dist=max_d,
                              non_emitting_states=False, obs_noise=std_noise)
    if latlon:
        path = list(zip(traj.lat, traj.lon))  # Make list of tuples of lat-lon columns of trajectory
    else:
        path = list(zip(traj.y, traj.x))
    if progress:
        states, _ = matcher.match(path, unique=False, tqdm=tqdm)
    else:
        states, idx = matcher.match(path, unique=False)
    # print(idx)
    if not states:
        raise Exception("Initial distance too small.")
    elif len(states) != len(traj):
        raise Exception("At least one datapoint not mapped or more states than datapoints")
    # Without verbose level on, exception is not raised and states list will be empty
    # --> raise own exception when list
    # is empty --> while loop will work as intended
    if plot_match:
        if len(path) > 1:
            fig, ax = plt.subplots(figsize=(12, 7))
            mmviz.plot_map(map, ax=ax, matcher=matcher,
                           use_osm=True, zoom_path=True,
                           show_labels=False, show_matching=True, show_graph=True)
    # filename="traj_test_"+str(j['Tracked Vehicle'][0])+"_init_dist.png")
    # nodes = matcher.path_pred_onlynodes
    map_matched = pd.DataFrame(states, columns=['n1', 'n2'])
    # print(map_matched_nodes.append(nodes))
    traj_moving_match = pd.concat([traj, map_matched], axis=1)
    return traj_moving_match


def make_gdf(df, line=False, latlon=False):
    if not line:
        if latlon:
            p = df.lon, df.lat
            crs = crs_pneuma
        else:
            p = df.x, df.y
            crs = crs_pneuma_proj
        geometry = [Point(xy) for xy in zip(*p)]
    else:
        if latlon:
            p1 = df.lon_1, df.lat_1
            p2 = df.lon_2, df.lat_2
            crs = crs_pneuma
        else:
            p1 = df.x_1, df.y_1
            p2 = df.x_2, df.y_2
            crs = crs_pneuma_proj
        s = [Point(yx) for yx in zip(*p1)]
        e = [Point(yx) for yx in zip(*p2)]
        geometry = [LineString(xy) for xy in zip(s, e)]
    df = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    return df


def converting_path_to_xy(gdf, zone=34, ellps='WGS84'):
    a_lon = list(gdf['lon'])
    a_lat = list(gdf['lat'])
    p = Proj(proj='utm', zone=zone, ellps=ellps, preserve_units=False)
    x, y = p(a_lon, a_lat, inverse=False)
    if 'track_id' not in list(gdf) and 'type' not in list(gdf):
        c = {'x': x, 'y': y, 'speed': gdf.speed, 'time': gdf.time}
    else:
        c = {'track_id': gdf['track_id'], 'type': gdf['type'], 'x': x, 'y': y, 'speed': gdf.speed, 'time': gdf.time}
    df = pd.DataFrame(c)
    return df


def converting_list_to_xy(ls, zone=34, ellps='WGS84'):
    a_lat = list(zip(*ls))[0]
    a_lon = list(zip(*ls))[1]
    p = Proj(proj='utm', zone=zone, ellps=ellps, preserve_units=False)
    x, y = p(a_lon, a_lat, inverse=False)
    c = {'x': x, 'y': y}
    df = pd.DataFrame(c)
    return df


def list_dfs_to_pickle(list_gdf, filename):
    filename = path_results + filename
    df_total = pd.concat(list_gdf, axis=0)
    df_total.to_pickle(filename)


def list_dfs_read_pickle(filename):
    filename = path_results + filename
    df_total = pd.read_pickle(filename)
    list_df = [j for i, j in df_total.groupby('track_id')]
    return list_df
