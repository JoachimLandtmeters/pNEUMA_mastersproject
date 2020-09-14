# Link between theoretic network graph and trajectories
# Map-matching the trajectories to the underlying theoretical network
# Using Leuven Map-matching algorithm
# Start and End node of matched edge in dataframe of trajectories --> link between theoretical network and measured data
import pickle
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import time
import numpy as np
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
from tqdm import tqdm
from tqdm.contrib import tenumerate
import sklearn
import compassbearing
from shapely.geometry import LineString
import sys
import math
import statistics
from pyproj import Proj, transform
import similaritymeasures
from collections import Counter
import seaborn
""" 
with open('athens_network_dbl.pkl', 'rb') as f:
    athens_network = pickle.load(f)

with open('network_matrix_dbl.pkl', 'rb') as g:
    network_matrix = pickle.load(g)

with open('trajectories_moving.pkl', 'rb') as h:
    trajectories_moving = pickle.load(h)
"""


class MapMatching:

    # max initial distance is best chosen as the maximum length of an edge in the network graph
    # max(network_matrix['length'])

    def __init__(self, list_traj, graph, gdf_netw, max_init, max_d):
        self.list_traj = list_traj
        self.max_init = max_init
        self.max_d = max_d
        self.network_graph = graph
        self.gdf_netw = gdf_netw

        def make_map(graph, latlon=True, index_edges=True):
            map_con = InMemMap("athensmap", use_latlon=latlon, index_edges=index_edges, use_rtree=True)  # Deprecated crs style in algorithm
            nodes_proj, edges_proj = ox.graph_to_gdfs(graph, nodes=True, edges=True)
            for nid, row in nodes_proj[['y', 'x']].iterrows():
                map_con.add_node(nid, (row['y'], row['x']))
            for nid, row in edges_proj[['u', 'v']].iterrows():
                map_con.add_edge(row['u'], row['v'])
            return map_con

        self.map = make_map(self.network_graph)

    # Create in memory graph for leuvenmapmatching from osmnx graph

    # Compatible path type for map matching

    def match_fixed_distance(self, list_index=None):
        # logger = lm.logger
        # logger.setLevel(logging.DEBUG)
        # logger.addHandler(logging.StreamHandler(sys.stdout))
        tic = time.time()
        traj_mov_match = []
        special_cases = []
        point_traj = self.list_traj
        if list_index is None:
            point_traj = [j for i, j in enumerate(point_traj) if i in list_index]
        for i, j in tenumerate(point_traj):
            while True:
                try:
                    traj = map_matching(j, self.gdf_netw, self.map, self.max_init, self.max_d)
                    traj_mov_match.append(traj)
                    break
                except Exception:
                    special_cases.append(j)
                    break
        toc = time.time()
        print(f'{int(divmod(toc - tic, 60)[0])} min {int(divmod(toc - tic, 60)[1])} sec')
        return traj_mov_match, special_cases

    def match_variable_distance(self, list_index=None):
        # logger = lm.logger
        # logger.setLevel(logging.DEBUG)
        # logger.addHandler(logging.StreamHandler(sys.stdout))
        tic = time.time()
        traj_mov_match = []
        fails = []
        point_traj = self.list_traj
        if list_index is not None:
            point_traj = [j for i, j in enumerate(point_traj) if i in list_index]
        for i, j in tenumerate(point_traj):
            start_time = time.time()
            dist_init = self.max_init
            dist = self.max_d
            fail = 0
            while True:
                try:
                    traj = map_matching(j, self.gdf_netw, self.map, dist_init, dist)
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

    def __init__(self, list_matched_trajectories):
        self.list_mm = list_matched_trajectories
        self.point_trajectories = []
        self.nan_traj = []
        self.used_network = pd.DataFrame()
        self.edges_counts = pd.DataFrame()
        self.line_trajectories = []

    def map_matching_result_split(self):
        trajectories_moving = []
        netw = []
        nan_traj = []
        for i, j in tenumerate(self.list_mm):
            j[0]['edge'] = list(zip(j[0]['N1_match'].values, j[0]['N2_match'].values))
            if np.any(np.isnan(np.sum(j[0]['edge'].values))):
                nan_traj.append(j[0])
                continue
            trajectories_moving.append(j[0])
            netw.append(j[1])
        used_network = pd.concat(netw, axis=0)
        used_network.drop_duplicates(subset=['N1', 'N2'], inplace=True)
        used_network.reset_index(inplace=True, drop=True)
        used_network['edge'] = [tuple(xy) for xy in zip(used_network['N1'], used_network['N2'])]
        self.point_trajectories = trajectories_moving
        self.nan_traj = nan_traj
        self.used_network = used_network

    def make_line_trajectories(self):
        traj_line_match = []
        gdf_netw = self.used_network
        for i, j in tenumerate(self.point_trajectories):
            tr_m = pd.merge(j, gdf_netw[['bearing', 'edge']], how='left', on=['edge'])
            tr_m = tr_m.rename(columns={'bearing_x': 'bearing', 'bearing_y': 'bearing_edge'})
            diff = tr_m[['bearing', 'bearing_edge']].values
            bearing_diff = [round(abs(diff[a][0] - diff[a][1]), 1) for a in range(0, len(tr_m))]
            for a, b in enumerate(bearing_diff):
                if b > 180:
                    bearing_diff[a] = round(360 - b, 1)
            j['wrong_match'] = bearing_diff
            # point dataset with nodes of matched edge, this adds column to all original dataframes (chained assignment)
            tr = j[:-1]
            # making line dataset --> always start and end point --> last point has no successive point --> -1
            u_edge = j['edge'].values[:-1]
            v_edge = j['edge'].values[1:]
            w_1 = j['wrong_match'].values[:-1]
            w_2 = j['wrong_match'].values[1:]
            w = tuple(zip(w_1, w_2))
            c = {'u_match': u_edge, 'v_match': v_edge, 'time': tr['time'].values + 1000, 'wrong_match': w}
            df = pd.DataFrame(c)
            p = [LineString([j['geometry'].values[k], j['geometry'].values[k + 1]]) for k in range(0, len(j) - 1)]
            tr = tr.drop(['geometry', 'time', 'N1_match', 'N2_match', 'wrong_match', 'edge'], axis=1)
            tr = pd.concat([tr, df], axis=1)
            tr = gpd.GeoDataFrame(tr, geometry=p)
            tr = pd.merge(tr, j.iloc[1:, 8:15], how='inner', on=['time'])
            traj_line_match.append(tr)
        self.line_trajectories = traj_line_match

    def select_rows(self, segment_index=None):
        gdf_list = self.point_trajectories
        gdf_netw = self.used_network
        if segment_index is None:
            segment_index = list(np.arange(0, len(gdf_netw), 1))
        traj_eval = []
        for ind, traj in tenumerate(gdf_list):
            tr = traj.drop(['Lon', 'Lat'], axis=1)
            tr_first = tr.drop_duplicates('N1_match', keep='first')
            tr_first = tr_first.rename(columns={'N1_match': 'N1', 'N2_match': 'N2'})
            idx_first = list(tr_first.index)
            tr_first = pd.merge(tr_first, gdf_netw[['N1', 'N2', 'Long1', 'Lat1', 'length']].loc[segment_index]
                                , how='left', on=['N1', 'N2'])
            tr_first = tr_first.rename(columns={'Long1': 'Lon', 'Lat1': 'Lat'})
            tr_first = tr_first.rename(columns={'Long1': 'Lon', 'Lat1': 'Lat'})
            tr_first = tr_first.assign(index=idx_first)
            tr_last = tr.drop_duplicates('N1_match', keep='last')
            tr_last = tr_last.rename(columns={'N1_match': 'N1', 'N2_match': 'N2'})
            idx_last = list(tr_last.index)
            tr_last = pd.merge(tr_last, gdf_netw[['N1', 'N2', 'Long2', 'Lat2', 'length']].loc[segment_index]
                               , how='left', on=['N1', 'N2'])
            tr_last = tr_last.rename(columns={'Long2': 'Lon', 'Lat2': 'Lat'})
            tr_last = tr_last.assign(index=idx_last)
            tr_sel = pd.concat([tr_first, tr_last])
            tr_sel = tr_sel.sort_values(by='index')
            df = traj.loc[idx_first + idx_last]
            df = df.sort_index()
            traj_eval.append([tr_sel, df])
        return traj_eval

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
            tracked_vehicle.append(j['Tracked Vehicle'].values[0])
            mode.append(j['Type'].values[0])
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
        evaluation = {'ID': tracked_vehicle, 'Type': mode,
                      'Frechet_distance': dist_frech_full, 'Frechet_distance_cut': dist_frech_cut,
                      'Length_difference': arc_length_diff_full, 'Length_difference_cut': arc_length_diff_cut}
        evaluation = pd.DataFrame(evaluation)
        return evaluation

    def distance_point_to_matched_edge(self):
        distances = {'ID': [], 'Type': [], 'length_traj': [], 'max_distance': [], 'median_distance': [],
                     'mean_distance': [],
                     '99_percentile': [], 'length_diff': [], 'length_diff_rel': []} #, 'frechet_distance': []}
        list_distances_traj = []
        for ind, traj in tenumerate(self.point_trajectories):
            distances['ID'].append((ind, traj['Tracked Vehicle'].values[0]))
            distances['Type'].append(traj['Type'].values[0])
            distances['length_traj'].append(len(traj))
            dist = []
            mapped_length = 0
            traj_val = traj[['Lon', 'Lat']].values
            xy_crds = converting_path_to_xy(traj)
            p_xy = xy_crds[['x', 'y']].values
            path_length = similaritymeasures.get_arc_length(p_xy)
            # print(path_length[0])
            traj_match = traj.rename(columns={'N1_match': 'N1', 'N2_match': 'N2'})
            match_df = pd.merge(traj_match[['edge']], self.used_network[['N1', 'Lat1', 'Long1', 'N2', 'Lat2', 'Long2',
                                                                         'length', 'edge']], how='left', on=['edge'])
            match_val = match_df[['Lat1', 'Long1', 'Lat2', 'Long2', 'edge', 'length']].values
            #q_1 = [xy for xy in zip(match_df.Lat1.values, match_df.Long1)]
            #idx = [i for i in range(len(q_1)) if q_1[i] != q_1[i-1]]
            #q_1 = list(match_df[['Lat1', 'Long1']].loc[idx].values)
            #if len(q_1) < 1:  # Interpolated points have to be appended
            #    q_1 = [0]
            #q_1 = []
            for row in range(0, len(traj)):
                p = (traj_val[row][1], traj_val[row][0])  # Lat-lon order
                s1 = (match_val[row][0], match_val[row][1])
                s2 = (match_val[row][2], match_val[row][3])
                d, pi, ti = lm_dist.distance_point_to_segment(p, s1, s2)
                dist.append(d)
                if row == 0:
                    mapped_length += match_val[row][5] * (1 - ti)
                    #q_1.append(pi)
                elif row == len(traj) - 1:
                    mapped_length += match_val[row][5] * ti
                    #q_1.append(pi)
                elif 0 < row and match_val[row][4] != match_val[row - 1][4]:
                    #q_1.append(pi)
                    mapped_length += match_val[row][5]
            if match_val[len(traj) - 2][4] == match_val[len(traj) - 1][4]:
                mapped_length -= match_val[len(traj) - 1][5]
            # print(mapped_length)
            diff_len = abs(path_length[0] - mapped_length)
            list_distances_traj.append(dist)
            #q_xy = converting_list_to_xy(q_1)
            #d_fr = similaritymeasures.frechet_dist(p_xy, q_xy.values)
            distances['max_distance'].append(max(dist))
            distances['median_distance'].append(np.median(dist))
            distances['mean_distance'].append(np.mean(dist))
            distances['99_percentile'].append(np.percentile(dist, 99))
            distances['length_diff'].append(diff_len)
            distances['length_diff_rel'].append(diff_len / path_length[0])
            #distances['frechet_distance'].append(d_fr)
        distances = pd.DataFrame(distances)
        return distances, list_distances_traj

    def fraction_wrongly_matched(self, threshold_angle=45):
        wrongly_matched = {'ID': [], 'Type': [], 'wrong_1': [], 'average_speed_1': [],'bool_w1': [], 'wrong_2': [],
                           'average_speed_2': [], 'bool_w2':[], 'wrong_both': [],'bool_wb':[], 'length_trajectory': []}
        for ind, traj in tenumerate(self.line_trajectories):
            wrongly_matched['ID'].append((ind, traj['Tracked Vehicle'].values[0]))
            wrongly_matched['Type'].append(traj['Type'].values[0])
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
        list_traj = self.point_trajectories
        gdf_netw = self.used_network
        ed = []
        nan_ed = []
        for k, l in enumerate(tqdm(list_traj)):
            t_edges = l['edge'].copy()
            t_edges.drop_duplicates(inplace=True)
            for i, j in enumerate(list(t_edges)):
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


def map_matching(gdf, gdf_netw, map, max_init, max_d, std_noise=1, progress=True, plot_match=False):
    map_matched_nodes = []
    matcher = DistanceMatcher(map, max_dist_init=max_init, max_dist=max_d,
                              non_emitting_states=False, obs_noise=std_noise)
    # Test with min prob norm
    # matcher = DistanceMatcher(map_con, min_prob_norm=prob)
    path = list(zip(gdf.loc[:, 'Lat'], gdf.loc[:, 'Lon']))  # Make list of tuples of lat-lon columns of trajectory
    if progress:
        states, _ = matcher.match(path, unique=False, tqdm=tqdm)
    else:
        states, idx = matcher.match(path, unique=False)
    # print(idx)
    if not states:
        raise Exception("Initial distance too small.")
    elif len(states) != len(gdf):
        raise Exception("At least one datapoint not mapped or more states than datapoints")
    # Without verbose level on, exception is not raised and states list will be empty
    # --> raise own exception when list
    # is empty --> while loop will work as intended
    if plot_match:
        fig, ax = plt.subplots(figsize=(12, 7))
        mmviz.plot_map(map, ax=ax,  matcher=matcher,
                       use_osm=True, zoom_path=True,
                       show_labels=False, show_matching=True, show_graph=True)
    # filename="traj_test_"+str(j['Tracked Vehicle'][0])+"_init_dist.png")
    # nodes = matcher.path_pred_onlynodes
    map_matched = pd.DataFrame(states, columns=['N1', 'N2'])
    df_match = map_matched.rename(columns={'N1': 'N1_match', 'N2': 'N2_match'})
    # print(map_matched_nodes.append(nodes))
    traj_moving_match = pd.concat([gdf, df_match], axis=1)
    used_network = map_matched
    used_network.drop_duplicates(subset=['N1', 'N2'], inplace=True)
    used_network = pd.merge(gdf_netw, used_network, how='inner', on=['N1', 'N2'])
    tr_df = [traj_moving_match, used_network]
    return tr_df


def converting_path_to_xy(gdf, zone=34, ellps='WGS84'):
    a_lon = list(gdf['Lon'])
    a_lat = list(gdf['Lat'])
    p = Proj(proj='utm', zone=zone, ellps=ellps, preserve_units=False)
    x, y = p(a_lon, a_lat, inverse=False)
    if 'Tracked Vehicle' not in list(gdf) and 'Type' not in list(gdf):
        c = {'x': x, 'y': y}
    else:
        c = {'Tracked Vehicle': gdf['Tracked Vehicle'], 'Type': gdf['Type'], 'x': x, 'y': y}
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
    df_total = pd.concat(list_gdf, axis=0)
    df_total.to_pickle(filename)


def list_dfs_read_pickle(filename):
    df_total = pd.read_pickle(filename)
    list_df = [j for i, j in df_total.groupby('Tracked Vehicle')]
    return list_df


# traj = map_matching(trajectories_moving[409], network_matrix, max(network_matrix['length']), 2)


"""
for v, w in enumerate(fails):
    off = 5
    if w <= 3:
        fails[v] = off + w * off
    if w > 3:
        fails[v] = off * 4 + (w - 2) * 10

plt.figure()
plt.hist(fails, bins=list(np.unique(fails)))
print(statistics.mean(fails))
print(statistics.median(fails))
print(np.percentile(fails, 99))

# with open('traj_tot_match_dbl.pkl', 'wb') as a:
# pickle.dump(traj_mov_match, a)

# Only keep the used edges of the network --> total link length
# Put detectors (counters) on the used edges only
"""
