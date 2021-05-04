# Calculate parameters from counts
# Draw FD by using the special points of fundamental diagrams
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib import tenumerate
import collections
import time
import copy
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
# from statsmodels.graphics.gofplots import qqplot
import operator
import scipy
import geopy.distance as geodist
import leuvenmapmatching.util.dist_latlon as distlatlon
import leuvenmapmatching.util.dist_euclidean as distxy


def vehicle_crossings(gdf_traj, d_gdf, bearing_difference=90, strict_match=True):
    tic = time.time()
    print('Start: …searching crossings')
    assert isinstance(gdf_traj, gpd.GeoDataFrame)
    assert {'n_det', 'lonlat'}.issubset(set(d_gdf.attrs.keys()))
    assert {'line_length_latlon', 'line_length_yx'}.issubset(set(gdf_traj.columns))
    n_det = d_gdf.attrs['n_det']
    lonlat = d_gdf.attrs['lonlat']
    c1, c2 = 'x', 'y'
    if lonlat:
        c1, c2 = 'lon', 'lat'
    # column multi-index
    col_names = [[f'cross_{c1}{i}', f'cross_{c2}{i}', f'rid{i}',
                  f'd{i}', f't{i}', f'v{i}'] for i in range(1, n_det + 1)]
    col_names = [item for sublist in col_names for item in sublist]
    feature = False
    if 'det_signal1' in d_gdf.columns:
        col_names = [[f'cross_{c1}_ts{i}', f'cross_{c2}_ts{i}', f'rid_ts{i}',
                      f'd_ts{i}', f't_ts{i}', f'v_ts{i}'] for i in range(1, n_det + 1)]
        col_names = [item for sublist in col_names for item in sublist]
        tuples = [(i[0], i[1], col_names[v]) for i, j in d_gdf.iterrows() for v in range(0, 6)]
        col_index = pd.MultiIndex.from_tuples(tuples, names=['edge', 'node', 'detector'])
        d_gdf = d_gdf.reset_index()
        detector_column = 'det_signal'
        feature = True
    else:
        col_index = pd.MultiIndex.from_product([d_gdf['_id'], col_names], names=['edge', 'detector'])
        detector_column = 'det_edge_'
    if isinstance(gdf_traj.index, pd.MultiIndex):
        assert gdf_traj.index.names.index('track_id') == 0
        row_index = set(gdf_traj.index.get_level_values(0))
    else:
        row_index = set(gdf_traj['track_id'])
        gdf_traj.set_index(['track_id', 'rid'], inplace=True)
    p1 = list(zip(*(gdf_traj[f'{c2}_1'], gdf_traj[f'{c1}_1'])))
    gdf_traj['p1'] = p1
    df_result = pd.DataFrame(index=row_index, columns=col_index)
    df = gdf_traj
    for i, det_link in tqdm(d_gdf.iterrows(), total=d_gdf.shape[0]):  # Counting vehicles for every used edge
        df_bool_wm = df[['wm1', 'wm2']].values < bearing_difference
        df_wm = df[np.logical_and(df_bool_wm[:, 0], df_bool_wm[:, 1])]
        if strict_match:
            df_bool_edge = df_wm[['u_match', 'v_match']].values == det_link['_id']
        else:
            df_bool_edge = df_wm[['u_match', 'v_match']].values >= 0
        df_u = df_wm[df_bool_edge[:, 0]].index.to_list()
        df_v = df_wm[df_bool_edge[:, 1]].index.to_list()
        set_index = set(df_u + df_v)
        if len(set_index) == 0:
            continue
        df2 = df_wm.loc[set_index]
        for n in range(1, n_det + 1):
            df_search = df2['geometry'].values.intersects(det_link[f'{detector_column}{n}'])
            df_intersect = df2[df_search].index.to_list()
            if not df_intersect:
                continue
            tid, rid = zip(*df_intersect)
            df_search_cross = df2[df_search]
            df_cross = df_search_cross['geometry'].values.intersection(det_link[f'{detector_column}{n}'])
            df_cross = [(c.y, c.x) for c in df_cross]
            if lonlat:
                df_dist = np.array([round(distlatlon.distance(*yx), 3) for yx in zip(df_search_cross.p1, df_cross)])
            else:
                df_dist = np.array([round(distxy.distance(*yx), 3) for yx in zip(df_search_cross.p1, df_cross)])
            t, v = interpolate_crossing(df_search_cross, df_dist, lonlat=lonlat)
            df_c2, df_c1 = zip(*df_cross)
            if not feature:
                df_result.loc[list(tid), (det_link["_id"], f'rid{n}')] = list(rid)
                df_result.loc[list(tid), (det_link["_id"], f'cross_{c1}{n}')] = df_c1
                df_result.loc[list(tid), (det_link["_id"], f'cross_{c2}{n}')] = df_c2
                df_result.loc[list(tid), (det_link["_id"], f'd{n}')] = df_dist
                df_result.loc[list(tid), (det_link["_id"], f't{n}')] = t
                df_result.loc[list(tid), (det_link["_id"], f'v{n}')] = v
            else:
                df_result.loc[list(tid), (det_link["_id"], det_link["index"], f'rid_ts{n}')] = list(rid)
                df_result.loc[list(tid), (det_link["_id"], det_link["index"], f'cross_{c1}_ts{n}')] = df_c1
                df_result.loc[list(tid), (det_link["_id"], det_link["index"], f'cross_{c2}_ts{n}')] = df_c2
                df_result.loc[list(tid), (det_link["_id"], det_link["index"], f'd_ts{n}')] = df_dist
                df_result.loc[list(tid), (det_link["_id"], det_link["index"], f't_ts{n}')] = t
                df_result.loc[list(tid), (det_link["_id"], det_link["index"], f'v_ts{n}')] = v
    df_result.sort_index(inplace=True)
    df_result = df_result.transpose()
    toc = time.time()
    print(f'Finding crossings done, took {toc - tic} sec')
    return df_result


def interpolate_crossing(df, p, lonlat=False):
    assert 'time' in df.columns
    assert 'speed_1' and 'speed_2' in df.columns
    assert {'line_length_latlon', 'line_length_yx'}.issubset(set(df.columns))
    if lonlat:
        dist = df.line_length_latlon.values
    else:
        dist = df.line_length_yx.values
    t = np.round(df.time.values - 1000 + p / dist * 1000)
    v = np.round((df.speed_2.values - df.speed_1.values) * (t - df.time.values + 1000) / 1000 + df.speed_1.values, 3)
    return t, v


def cleaning_counting(det_c, n_det, double_loops=False):
    det_cross = []
    indexes_list = []
    crossed = {}
    for det in range(1, n_det + 1):
        for a, b in det_c.iterrows():  # For every time step
            if double_loops:
                t_cross = [[b[f'times_{det}'][e], b[f'times_lp_{det}'][e]] for e in
                           range(0, len(b['times_1']))]  # Do for every trajectory
                det_cross.append(t_cross)
        crossed[f'no_cross_{det}'] = []
        crossed[f'partly_cross_{det}'] = []
        for ind in range(0, len(det_cross[0])):  # For every trajectory
            cr = False
            cr_lp = False
            for i in range(len(det_c) * (det - 1), len(det_c) * det):  # For every time step for specific detector pair
                if bool(det_cross[i][ind][0]):
                    cr = True
                if bool(det_cross[i][ind][1]):
                    cr_lp = True
            if cr and cr_lp:
                continue
            elif cr != cr_lp:
                crossed[f'partly_cross_{det}'].append(ind)
            else:
                crossed[f'no_cross_{det}'].append(ind)
    return crossed


def count_vehicles(d_gdf, gdf_traj, n_det, freq, double_loops, mode_exclusion=(),
                   vehicle_dim=None):
    if vehicle_dim is None:
        vehicle_dim = {'Car': [2, 5], 'Motorcycle': [1, 2.5], 'Bus': [4, 12.5], 'Taxi': [2, 5],
                       'Medium Vehicle': [2.67, 5.83], 'Heavy Vehicle': [3.3, 12.5],
                       'Bicycle': [0, 0], 'Pedestrian': [0, 0]}
    detector_counts = []
    vehicle_type = {}
    long_distances_dfs = []
    # Pre-loading time steps
    max_d = 0
    max_d = int(max([j['time'].values[len(j) - 1] for j in gdf_traj if j['time'].values[len(j) - 1] > max_d])) + 1
    # print(max_d)
    # +1 --> To make sure the last trajectory point is included
    time_st = []
    print('Pre-loading time steps for every trajectory …')
    for interval in tqdm(range(freq, max_d + freq, freq)):
        pre_interval = interval - freq
        steps = [np.logical_and(j['time'].values >= pre_interval, interval > j['time'].values) for j in gdf_traj]
        time_st.append(steps)
    for i, det_link in d_gdf.iterrows():  # Counting vehicles for every used edge
        print(f"Counting on edge: {det_link['index']}")
        if double_loops:
            loop_dist = round(det_link['loop_distance'], 3)
        edge_counts = collections.OrderedDict()
        edge_times = collections.OrderedDict()
        long_distances = []
        index = []
        d1 = []
        d2 = []
        traj_match_values = []  # save values of needed columns for every trajectory
        vehicle_type[f"vehicle_type_{det_link['index']}"] = []
        vehicle_type[f"vehicle_index_{det_link['index']}"] = []
        vehicle_type[f"vehicle_crossing_{det_link['index']}"] = {'index': []}
        print('Matched trajectories on edge …')
        for k, l in tenumerate(gdf_traj):  # Check for every trajectory if it is on the edge
            d1.append([tuple(xy) for xy in zip(l['lat_x'], l['lon_x'])])  # Lat-Lon of first datapoint(u) of linestring
            d2.append([tuple(xy) for xy in zip(l['lat_y'], l['lon_y'])])  # Lat-Lon of second datapoint(v) of linestring
            if np.logical_or(d_gdf.loc[i, 'edge'] in list(l['u_match'].values),
                             d_gdf.loc[i, 'edge'] in list(l['v_match'].values)):
                if l['type'].values[0] in mode_exclusion:
                    continue
                traj_match_values.append(l[['u_match', 'v_match', 'wrong_match', 'time', 'geometry', 'speed_x',
                                            'speed_y', 'bearing_x']].values)
                index.append(k)
                vehicle_type[f"vehicle_type_{det_link['index']}"].append(l['type'].values[0])
                vehicle_type[f"vehicle_index_{det_link['index']}"].append((k, f"ID:{l['track_id'].values[0]}"))
                vehicle_type[f"vehicle_crossing_{det_link['index']}"]['index'].append(k)
        for det in range(1, n_det + 1):
            edge_counts[f'counts_{det}'] = []
            edge_times[f'times_{det}'] = []
            if double_loops:
                edge_counts[f'counts_lp_{det}'] = []
                edge_times[f'times_lp_{det}'] = []
        # print(index)
        print('VKT and VHT for every time step …')
        tag = [[0] * n_det for i in range(0, len(index))]
        tag_lp = [[0] * n_det for i in range(0, len(index))]
        for g, h in tenumerate(time_st):  # for every time step
            # print('Step: ' + str(g + 1))
            cnt = {}
            det_time_st = {}
            for det in range(1, n_det + 1):
                cnt[f'cnt_{det}'] = []
                det_time_st[f'time_step_{det}'] = []
                if double_loops:
                    cnt[f'cnt_lp_{det}'] = []
                    det_time_st[f'time_step_lp_{det}'] = []
            for m, n in enumerate(index):  # Every trajectory mapped to used edges
                cnt_x = {}
                traj_t = {}
                """
                if vehicle_dim:
                    veh_l = vehicle_dim[vehicle_type[f"vehicle_type_{det_link['index']}"][m]][1]
                else:
                    veh_l = loop_dist
                """
                for det in range(1, n_det + 1):
                    cnt_x[f'x_{det}'] = 0
                    traj_t[f't_{det}'] = 0
                    if double_loops:
                        cnt_x[f'x_lp_{det}'] = 0
                        traj_t[f't_lp_{det}'] = 0
                for idx, in_timestep in enumerate(time_st[g][n]):  # time_st[g][n].iteritems():
                    if in_timestep:
                        for det in range(1, n_det + 1):
                            f = 0  #
                            if np.logical_or(traj_match_values[m][idx][0] == det_link['edge'],
                                             traj_match_values[m][idx][1] == det_link['edge']):
                                if np.logical_and(traj_match_values[m][idx][2][0] > 90,
                                                  traj_match_values[m][idx][2][1] > 90):
                                    continue
                                if traj_match_values[m][idx][4].intersects(det_link[f'det_edge_{det}']):
                                    f = 1  # tag to mark index that intersects with detector, prevents errors in traveled
                                    # distance calculation
                                    tag[m][det - 1] = 1
                                    d12 = geodist.distance(d1[n][idx], d2[n][idx]).m
                                    c = traj_match_values[m][idx][4].intersection(det_link[f'det_edge_{det}'])
                                    c = (c.y, c.x)
                                    d1c = round(geodist.distance(d1[n][idx], c).m, 3)
                                    dc2 = round(geodist.distance(c, d2[n][idx]).m, 3)
                                    cnt_x[f'x_{det}'] = 1
                                    if double_loops:
                                        cnt_x[f'x_{det}'] = dc2
                                    traj_t[f't_{det}'] = round(traj_match_values[m][idx][3] - 1000 + d1c / d12 * 1000)
                                    if traj_t[f't_{det}'] < (freq * g):
                                        edge_times[f'times_{det}'][g - 1][m] = traj_t[f't_{det}']
                                        cnt_x[f'x_{det}'] = \
                                            round((traj_match_values[m][idx][3] - freq * g) /
                                                  (traj_match_values[m][idx][3] - traj_t[f't_{det}'])
                                                  * geodist.distance(c, d2[n][idx]).m, 3)
                                        edge_counts[f'counts_{det}'][g - 1][m] = round(
                                            geodist.distance(c, d2[n][idx]).m -
                                            cnt_x[f'x_{det}'], 3)
                                        if not double_loops:
                                            cnt_x[f'x_{det}'] = 0
                                            edge_counts[f'counts_{det}'][g - 1][m] = 1
                                        traj_t[f't_{det}'] = 0
                                if double_loops:
                                    if traj_match_values[m][idx][4].intersects(det_link[f'det_edge_{det}bis']):
                                        tag_lp[m][det - 1] = 1
                                        d12 = geodist.distance(d1[n][idx], d2[n][idx]).m
                                        c_lp = traj_match_values[m][idx][4].intersection(det_link[f'det_edge_{det}bis'])
                                        c_lp = (c_lp.y, c_lp.x)
                                        d1c = round(geodist.distance(d1[n][idx], c_lp).m, 3)
                                        cnt_x[f'x_lp_{det}'] = d1c
                                        traj_t[f't_lp_{det}'] = round(
                                            traj_match_values[m][idx][3] - 1000 + d1c / d12 * 1000)
                                        if traj_t[f't_lp_{det}'] < (freq * g):  # crossing in previous time step
                                            edge_times[f'times_lp_{det}'][g - 1][m] = traj_t[f't_lp_{det}']
                                            edge_counts[f'counts_lp_{det}'][g - 1][m] = round(
                                                geodist.distance(d1[n][idx],
                                                                 c_lp).m, 3)
                                            if f > 0:
                                                edge_counts[f'counts_lp_{det}'][g - 1][m] = \
                                                    round(geodist.distance(c, c_lp).m, 3)
                                                edge_counts[f'counts_{det}'][g - 1][m] = 0
                                                # round(dist.distance(c, c_lp).m, 3)
                                                cnt_x[f'x_{det}'] = 0
                                            traj_t[f't_lp_{det}'] = 0
                                            cnt_x[f'x_lp_{det}'] = 0
                                        elif f > 0:  # same line crosses both detectors
                                            cnt_x[f'x_lp_{det}'] = round(geodist.distance(c, c_lp).m, 3)
                                            # print('Direct crossing ' + str(cnt_x['x_lp_' + str(det)]))
                                            cnt_x[f'x_{det}'] = 0
                                            if g > 0:
                                                if edge_times[f'times_{det}'][g - 1][m]:
                                                    cnt_x[f'x_lp_{det}'] = round(geodist.distance(c, c_lp).m -
                                                                                 edge_counts[f'counts_{det}'][
                                                                                     g - 1][m], 3)
                                        elif traj_match_values[m][idx][3] - 1000 < (freq * g):
                                            # first point in previous time step
                                            cnt_x[f'x_lp_{det}'] = round((traj_t[f't_lp_{det}'] - freq * g) /
                                                                         (traj_t[f't_lp_{det}'] + 1000 -
                                                                          traj_match_values[m][idx][3]) * d1c, 3)
                                            edge_counts[f'counts_{det}'][g - 1][m] = \
                                                edge_counts[f'counts_{det}'][g - 1][m] + (d1c - cnt_x[f'x_lp_{det}'])
                                            # Add extra distance to existing value
                                    elif tag[m][det - 1] > 0 and tag_lp[m][det - 1] < 1 and f < 1:
                                        d_int = round(geodist.distance(d1[n][idx], d2[n][idx]).m, 3)
                                        if traj_match_values[m][idx][3] - 1000 < (freq * g):
                                            cnt_x[f'x_{det}'] = round((traj_match_values[m][idx][3] - freq * g) /
                                                                      1000 * d_int, 3)
                                            edge_counts[f'counts_{det}'][g - 1][m] = \
                                                edge_counts[f'counts_{det}'][g - 1][m] + (d_int -
                                                                                          cnt_x[f'x_{det}'])
                                        else:
                                            cnt_x[f'x_{det}'] = round(cnt_x[f'x_{det}'] + d_int, 3)
                                        if g > 0:
                                            if loop_dist < (cnt_x[f'x_{det}'] +
                                                            edge_counts[f'counts_{det}'][g - 1][m]):
                                                distance = round(cnt_x[f"x_{det}"] +
                                                                 edge_counts[f"counts_{det}"][g - 1][m], 3)
                                                long_distances.append([distance, n, idx, traj_match_values[m][idx][2],
                                                                       g, det])
                                    # If vehicle does not cross both detectors, discarding them can skew the data
                                    # --> the amount of vehicles is not negligible
                                    # after last data point, append traveled distance and time spent
                                    # loop distance is maximum so wrongly matched trajectories have no effect
                                    # elif idx == len(gdf_traj[n])-1 and tag_lp[m][det-1] < 1:
                                    # traj_t['t_lp_' + str(det)] = gdf_traj[n]['time'][idx]
                            if tag_lp[m][det - 1]:  # reset tags --> loops of vehicles are possible
                                tag[m][det - 1] = 0
                                tag_lp[m][det - 1] = 0
                for det in range(1, n_det + 1):
                    det_time_st[f'time_step_{det}'].append(traj_t[f't_{det}'])
                    cnt[f'cnt_{det}'].append(cnt_x[f'x_{det}'])
                    if double_loops:
                        det_time_st[f'time_step_lp_{det}'].append(traj_t[f't_lp_{det}'])
                        cnt[f'cnt_lp_{det}'].append(cnt_x[f'x_lp_{det}'])
            for det in range(1, n_det + 1):
                edge_counts[f'counts_{det}'].append(cnt[f'cnt_{det}'])
                edge_times[f'times_{det}'].append(det_time_st[f'time_step_{det}'])
                if double_loops:
                    edge_counts[f'counts_lp_{det}'].append(cnt[f'cnt_lp_{det}'])
                    edge_times[f'times_lp_{det}'].append(det_time_st[f'time_step_lp_{det}'])
        # print(vehicle_length_traveled)
        edge_counts.update(edge_times)
        counts = pd.DataFrame(edge_counts)
        detector_counts.append(counts)
        long_distances = pd.DataFrame(long_distances, columns=['distance', 'trajectory', 'line_index', 'wrong_match',
                                                               'time_step', 'detector'])
        long_distances_dfs.append(long_distances)
        df_veh = pd.DataFrame(vehicle_type[f"vehicle_crossing_{det_link['index']}"])
        vehicle_type[f"vehicle_crossing_{det_link['index']}"] = df_veh
    return {'counts': detector_counts, 'detectors': d_gdf, 'vehicle_type': vehicle_type,
            'long distances': long_distances_dfs}


def save_plot_fd(det_c, parameters_list, parameters_list_mode_sel=None, parameters_list_ao=None, labels=None,
                 colors=None, veh_area_colorbar=False, name_file=None):
    figures = []
    if not labels:
        labels = ['Original', 'All Modes', 'Mode Selection', 'Adjusted']
    if not colors:
        colors = ['b', 'r']
    for i, j in enumerate(parameters_list):
        edge_id = det_c['detectors']['index'].values[i]
        loop_distance = det_c['detectors']['loop_distance'].values[i]
        if not name_file:
            name_file = edge_id
        for det in range(1, det_c['info']['number_of_det'][0] + 1):
            fig, ax = plt.subplots()
            ax.scatter(parameters_list[i][f'density_{det}'], parameters_list[i][f'flow_{det}'],
                       color=colors[0], label=labels[0])
            if veh_area_colorbar:
                f = ax.scatter(parameters_list[i]['density_' + str(det)], parameters_list[i]['flow_' + str(det)]
                               , label=labels[0], c=parameters_list[i][f'vehicle_area_{det}'])
                fig.colorbar(f)
            ax.legend(loc='upper left')
            ax.grid(True)
            plt.title(f"FD edge {name_file} (Detector loop: {det} with loop distance {loop_distance} m)")
            plt.xlabel('Density k [veh/km]')
            plt.ylabel('Flow q [veh/h]')
            plt.axis([- 20, int(max(parameters_list[i][f'density_{det}']) + 50),
                      - 100, 10000])
            figures.append((fig, ax))
            plt.savefig(f"FD_{det}_{name_file}_{int(det_c['info']['frequency'][0] / 1000)}_lp"
                        f"{loop_distance}.png")
            if parameters_list_mode_sel:
                fig, ax_1 = plt.subplots()
                ax_1.scatter(parameters_list[i][f'density_{det}'], parameters_list[i][f'flow_{det}'],
                             color=colors[0], label=labels[1], alpha=0.5)
                ax_1.scatter(parameters_list_mode_sel[i][f'density_{det}'],
                             parameters_list_mode_sel[i][f'flow_{det}'], color=colors[1], label=labels[2])
                ax_1.legend(loc='upper left')
                ax_1.grid(True)
                plt.title(f"FD edge {j['index'].values[0]} (Detector loop: {det})")
                plt.xlabel('Density k [veh/km]')
                plt.ylabel('Flow q [veh/h]')
                figures.append((fig, ax_1))
                plt.savefig(f"FD_comparison_modes_loop_{det}_"
                            f"{j['index'].values[0]}_{int(det_c['info']['frequency'][0] / 1000)}{labels[2]}.png")
            if parameters_list_ao:
                fig, ax_2 = plt.subplots()
                ax_2.scatter(parameters_list[i][f'density_{det}'], parameters_list[i][f'flow_{det}'],
                             color=colors[0], label=labels[0], alpha=0.5)
                ax_2.scatter(parameters_list_ao[i][f'density_{det}'], parameters_list_ao[i][f'flow_{det}'],
                             color=colors[1], marker='^', label=labels[3])
                ax_2.legend(loc='upper left')
                ax_2.grid(True)
                plt.title(f"FD edge {j['index'].values[0]} (Detector loop: {det})")
                plt.xlabel('Density k [veh/km]')
                plt.ylabel('Flow q [veh/h]')
                figures.append((fig, ax_2))
                plt.savefig(f"FD_comparison_loop_{det}_"
                            f"{j['index'].values[0]}_{int(det_c['info']['frequency'][0] / 1000)}_adjustment.png")

                # int(max(max(parameters_list[i]['flow_' + str(det)]),
                # max(parameters_list_ao[i]['flow_' + str(det)])) + 500)])
            plt.show()
    return figures


class TrafficAnalysis:
    adjustment_stopped = False, 0

    def __init__(self, d_gdf, gdf_traj, gdf_netw, n_det, freq, dfi, loop_distance,
                 double_loops, mode_exclusion=(),
                 vehicle_dim=None):
        self.IDarterial = list(d_gdf['index'].values)
        self.network = gdf_netw
        self.numberofdetectors = n_det
        self.frequency = freq
        self.dfi = dfi
        self.double = double_loops
        self.loop_distance = None
        self.modes_excluded = mode_exclusion
        self.vehicle_dimensions = vehicle_dim
        if vehicle_dim is None:
            vehicle_dim = {'Car': [2, 5], 'Motorcycle': [1, 2.5], 'Bus': [4, 12.5], 'Taxi': [2, 5],
                           'Medium Vehicle': [2.67, 5.83], 'Heavy Vehicle': [3.3, 12.5],
                           'Bicycle': [0, 0], 'Pedestrian': [0, 0]}
            self.vehicle_dimensions = vehicle_dim
        self.traffic_counts = count_vehicles(d_gdf, gdf_traj, n_det, freq, double_loops, mode_exclusion,
                                             self.vehicle_dimensions)
        if double_loops:
            self.loop_distance = loop_distance
            self.traffic_parameters = self.calculate_parameters()
            ta_filter = self.filter_stopped_vehicles()
            self.traffic_parameters_adj = self.adjustment_stopped_vehicles()
            self.traffic_parameters_noLS = ta_filter[0]  # No long stops
            self.filter_stopped = ta_filter[1]
            self.traffic_parameters_arterial = []
            self.traffic_parameters_arterial_network = []
            self.traffic_parameters_agg = []

    def crossings_detectors(self, detector='times_1'):
        ind_cross = self.traffic_counts['counts']

    def multiple_crossings(self):
        counts_dict = self.traffic_counts
        n_det = self.numberofdetectors
        edge_id = counts_dict['detectors']['index'].values
        multi_list = {f'{j}': {f'det_{det}': {'ID': [], 'crosses': []} for det in range(1, n_det + 1)}
                      for i, j in enumerate(edge_id)}
        for ind, counts in enumerate(counts_dict['counts']):
            len_idx = len(counts['times_1'][0])
            cnt = [[0] * len_idx for det in range(1, n_det + 1)]
            for det in range(1, n_det + 1):
                for id, row in counts[f'times_lp_{det}'].iteritems():
                    for i, val in enumerate(row):
                        if val:
                            cnt[det - 1][i] += 1
                            if cnt[det - 1][i] > 1:
                                multi_list[f'{edge_id[ind]}'][f'det_{det}']['ID']. \
                                    append(counts_dict['vehicle type'][f'vehicle_index_{edge_id[ind]}'][i])
                                multi_list[f'{edge_id[ind]}'][f'det_{det}']['crosses']. \
                                    append(cnt[det - 1][i])
        return multi_list

    def calculate_parameters(self, mode_exclusion=()):
        edges_parameters = []
        hour = 3600000  # Hour in ms to normalize units of flow
        # det_c info
        n_det = self.numberofdetectors
        freq = self.frequency
        double = self.double
        vehicle_width = self.vehicle_dimensions
        # Lanes to float
        lanes = []
        for x, y in self.traffic_counts['detectors']['lanes'].iteritems():
            if type(y) is not list:
                y = float(y)
                if np.isnan(y):  # replace nan values with default number of lanes
                    y = 1
                lanes.append(y)
            else:
                y = [float(s) for s in y]
                y = min(y)
                lanes.append(y)
        self.traffic_counts['detectors'] = self.traffic_counts['detectors'].assign(lanes_adj=lanes)
        # Clean counts
        det_error = []
        summary_clean_counts = []
        # print(det_error)
        for a, b in enumerate(tqdm(self.traffic_counts['counts'])):  # List of detector counts
            error = cleaning_counting(b, double_loops=double, n_det=n_det)
            summary = {'edge_index': self.traffic_counts['detectors']['index'].values[a], 'list_clean_counts': []}
            for key, value in error.items():
                t_count = (key, len(value))
                summary['list_clean_counts'].append(t_count)
            # print(summary)
            summary_clean_counts.append(summary)
            det_error.append(error)
            n_edge = self.traffic_counts['detectors']['index'].values[a]
            loop_distance = self.traffic_counts['detectors']['loop_distance'].values[a]
            free_flow = 100
            vehicle_type = self.traffic_counts['vehicle_type'][f'vehicle_type_{n_edge}']
            get_veh_type = collections.Counter(vehicle_type)
            total_matched = len(vehicle_type)
            # Density at zero flow and maximum occupancy (100%)
            parameters = {}
            parameters['index'] = [0] * len(b)
            parameters['index'][0] = n_edge
            parameters['loop_distance'] = [0] * len(b)
            parameters['loop_distance'][0] = loop_distance
            parameters['number_of_det'] = [0] * len(b)
            parameters['number_of_det'][0] = n_det
            parameters['frequency'] = [0] * len(b)
            parameters['frequency'][0] = freq
            for det in range(1, n_det + 1):  # For each detector pair
                parameters[f'density_{det}'] = []
                parameters[f'density_lane_{det}'] = []
                parameters[f'flow_{det}'] = []
                parameters[f'flow_lane_{det}'] = []
                parameters[f'speed_{det}'] = []
                # parameters[f'occupancy_{det}'] = []
                # parameters[f'flow_occ_{det}'] = []
                # parameters[f'density_occ_{det}'] = []
                # parameters[f'speed_occ_{det}'] = []
                # parameters[f'length_{det}'] = []
                parameters[f'vehicle_area_{det}'] = []
                parameters[f'modal_shares_{det}'] = []
                parameters[f'vehicles_{det}'] = []
                parameters[f'stopped_vehicles_{det}'] = []
                parameters[f'stops_{det}'] = []
                for veh_type in list(get_veh_type):
                    parameters[f'{veh_type}_{det}'] = []
                df = b.copy(deep=True)
                t_cross = df.loc[:, [f'times_{det}', f'times_lp_{det}']].values
                #  , f'occ_times_in_{det}', f'occ_times_out_{det}']].values
                q_cross = df.loc[:, [f'counts_{det}', f'counts_lp_{det}']].values
                for c, d in df.iterrows():  # Go over each time step of edge counts
                    time_spent_loops = []
                    traveled_distance = []
                    occupancy = []
                    flow_occupancy = []
                    vehicle_length = []
                    vehicle_area = []
                    modal_shares = {k: 0 for k in list(get_veh_type)}
                    stopped_vehicle = [0] * total_matched
                    for e in range(0, total_matched):  # For each trajectory
                        if e in det_error[a][f'no_cross_{det}']:
                            continue
                        elif e in det_error[a][f'partly_cross_{det}']:
                            """
                            if t_cross[c][2][e] > 0:
                                if t_cross[c][3][e] > 0:
                                    to_1 = t_cross[c][3][e] - t_cross[c][2][e]
                                    # ao_1 = t_1 * vehicle_width[vehicle_type[e]][0] #*vehicle_width[vehicle_type[e]][1])
                                    qo_1 = 1  # * vehicle_width[vehicle_type[e]][0]
                                    flow_occupancy.append(qo_1)
                                    occupancy.append(to_1)
                                else:
                                    to_2 = (freq * (c + 1) - t_cross[c][2][e])
                                    qo_2 = 0
                                    if c < len(b) - 1:  # Change entry time of vehicle in loop
                                        # b.loc[c + 1, f'occ_times_in_{det}'][e] = freq * (c + 1)
                                        t_cross[c + 1][2][e] = freq * (c + 1)
                                    flow_occupancy.append(qo_2)
                                    occupancy.append(to_2)
                                vehicle_length.append(vehicle_width[vehicle_type[e]][1])
                            """
                            continue
                        else:
                            if vehicle_type[e] not in mode_exclusion:
                                if t_cross[c][0][e] > 0:
                                    if t_cross[c][1][e] > 0:
                                        t_1 = (t_cross[c][1][e] - t_cross[c][0][e])  # / (frequency * loop_distance
                                        # * lanes[n_edge]) * 1000
                                        time_spent_loops.append(t_1)
                                        # print((k_1, c))
                                        q_1 = q_cross[c][0][e] + q_cross[c][1][e]
                                        traveled_distance.append(q_1)
                                    else:
                                        t_2 = (freq * (c + 1) - t_cross[c][0][e])  # / (frequency * loop_distance
                                        # * lanes[n_edge]) * 1000
                                        if c < len(df) - 1:  # Change entry time of vehicle in loop
                                            # Vehicles longer than one time step between detectors
                                            t_cross[c + 1][0][e] = freq * (c + 1)
                                        time_spent_loops.append(t_2)
                                        q_2 = q_cross[c][0][e] + q_cross[c][1][e]
                                        traveled_distance.append(q_2)
                                        if t_2 == freq:
                                            stopped_vehicle[e] = 1
                                    vehicle_area.append(np.prod(vehicle_width[vehicle_type[e]]))
                                    modal_shares[vehicle_type[e]] += 1
                                """
                                if t_cross[c][2][e] > 0:
                                    if t_cross[c][3][e] > 0:
                                        to_1 = t_cross[c][3][e] - t_cross[c][2][e]
                                        # ao_1 = t_1 * vehicle_width[vehicle_type[e]][0] #*vehicle_width[vehicle_type[e]][1])
                                        qo_1 = 1  # * vehicle_width[vehicle_type[e]][0]
                                        flow_occupancy.append(qo_1)
                                        occupancy.append(to_1)
                                    else:
                                        to_2 = (freq * (c + 1) - t_cross[c][2][e])
                                        qo_2 = 0
                                        if c < len(b) - 1:  # Change entry time of vehicle in loop
                                            t_cross[c + 1][2][e] = freq * (c + 1)
                                        flow_occupancy.append(qo_2)
                                        occupancy.append(to_2)
                                    vehicle_length.append(vehicle_width[vehicle_type[e]][1])
                                """
                    k_time_step = (sum(time_spent_loops) /
                                   (freq * loop_distance) * 1000)
                    # if sum(time_spent_loops['time_spent_loop_' + str(det)]) > freq:
                    # print(sum(time_spent_loops['time_spent_loop_' + str(det)]), c)
                    q_time_step = (sum(traveled_distance) /
                                   (freq * loop_distance) * hour)
                    # occ = (sum(occupancy)) / (freq * lanes[a])
                    # occ_flow = (sum(flow_occupancy)) / freq * hour
                    parameters[f'density_{det}'].append(k_time_step)
                    parameters[f'density_lane_{det}'].append(k_time_step / lanes[a])
                    parameters[f'flow_{det}'].append(q_time_step)
                    parameters[f'flow_lane_{det}'].append(q_time_step / lanes[a])
                    # parameters[f'occupancy_{det}'].append(occ)
                    # parameters[f'flow_occ_{det}'].append(occ_flow / lanes[a])
                    parameters[f'stopped_vehicles_{det}'].append(stopped_vehicle)
                    parameters[f'stops_{det}'].append(sum(stopped_vehicle))
                    """
                    if vehicle_length:
                        m_veh_l = np.mean(vehicle_length)
                        parameters[f'density_occ_{det}'].append(occ / m_veh_l * 1000)
                        u_occ = round(occ_flow / (occ / m_veh_l * 1000), 2)
                        parameters[f'speed_occ_{det}'].append(u_occ)
                    else:
                        m_veh_l = 0
                        parameters[f'density_occ_{det}'].append(0)
                        parameters[f'speed_occ_{det}'].append(0)
                    parameters[f'length_{det}'].append(m_veh_l)
                    """
                    if k_time_step:
                        u = round(q_time_step / k_time_step, 2)
                        parameters[f'vehicle_area_{det}'].append(np.mean(vehicle_area))
                        total = sum(modal_shares.values())
                        modal_shares = {k: round(v / total * 100, 1) for k, v in modal_shares.items()}
                        parameters[f'modal_shares_{det}'].append(modal_shares)
                        parameters[f'vehicles_{det}'].append(total)
                        for veh_type in list(get_veh_type):
                            parameters[f'{veh_type}_{det}'].append(modal_shares[veh_type])
                        if u < free_flow:
                            parameters[f'speed_{det}'].append(u)
                        else:
                            parameters[f'speed_{det}'].append(free_flow)
                    else:
                        parameters[f'speed_{det}'].append(0)
                        parameters[f'vehicle_area_{det}'].append(0)
                        parameters[f'modal_shares_{det}'].append(modal_shares)
                        parameters[f'vehicles_{det}'].append(0)
                        for veh_type in list(get_veh_type):
                            parameters[f'{veh_type}_{det}'].append(modal_shares[veh_type])
            calc_parameters = pd.DataFrame(parameters)
            calc_parameters.reset_index(inplace=True)
            edges_parameters.append(calc_parameters)
        return edges_parameters

    def adjustment_stopped_vehicles(self, stop_time=6):
        n_det = self.numberofdetectors
        loop_distance = self.traffic_counts['detectors']['loop_distance'].values
        lanes = self.traffic_counts['detectors']['lanes_adj'].values
        edge_id = list(self.traffic_counts['detectors']['index'].values)
        frequency = self.frequency / 1000
        new_param_list = []
        for ind, param in enumerate(self.traffic_parameters.copy()):
            # Value of one stopped vehicle for one time step
            filter_value = (1000 / loop_distance[ind])
            tag = [0] * n_det
            filter = {}
            for d in range(1, n_det + 1):
                # filter[f'f_{d}'] = [0] * len(param)
                filter[f'density_{d}'] = list(param[f'density_{d}'].values)
                filter[f'lanes_{d}'] = [lanes[ind]] * len(param)
                for id, vt in enumerate(list(collections.Counter(
                        self.traffic_counts['vehicle_type'][f'vehicle_type_{edge_id[ind]}']).keys())):
                    filter[f'{vt}_{d}'] = list(param[f'{vt}_{d}'].values)
                    param = param.drop([f'{vt}_{d}'], axis=1)
            for k, v in param.iterrows():
                for det in range(1, n_det + 1):
                    if v[f'stops_{det}'] == 1 and lanes[ind] > 1:
                        tag[det - 1] += 1
                        if tag[det - 1] == stop_time:
                            # filter[f'f_{det}'][k-2] = filter_value
                            # filter[f'f_{det}'][k-1] = filter_value
                            for adj in range(0, stop_time):
                                filter[f'density_{det}'][k - adj] -= filter_value
                                filter[f'lanes_{det}'][k - adj] -= 1
                        elif min(tag[det - 1], stop_time) == stop_time:
                            # filter[f'f_{det}'][k] = filter_value
                            filter[f'density_{det}'][k] -= filter_value
                            filter[f'lanes_{det}'][k] -= 1
                    else:
                        tag[det - 1] = 0
            for det in range(1, n_det + 1):
                filter[f'density_lane_{det}'] = np.array(filter[f'density_{det}']) / np.array(filter[f'lanes_{det}'])
                filter[f'speed_{det}'] = [
                    param[f'flow_{det}'][k] / filter[f'density_{det}'][k] if filter[f'density_{det}'][k]
                    else 0 for k, v in param.iterrows()]
                param = param.drop([f'density_{det}', f'density_lane_{det}', f'speed_{det}'], axis=1)
                # param[f'density_adj_{det}'] = param[f'density_{det}'] - filter[f'f_{det}']
            filter = pd.DataFrame(filter)
            param = pd.concat([param, filter], axis=1)
            for det in range(1, n_det + 1):
                param = param.drop([f'lanes_{det}'], axis=1)
            new_param_list.append(param)
            self.adjustment_stopped = True, stop_time
        return new_param_list

    def filter_stopped_vehicles(self, stop_time=0):
        if stop_time == 0:  # Default is five minutes
            stop_time = round(int(300000 / self.frequency))
        n_det = self.numberofdetectors
        loop_distance = self.traffic_counts['detectors']['loop_distance'].values
        lanes = self.traffic_counts['detectors']['lanes_adj'].values
        edge_id = list(self.traffic_counts['detectors']['index'].values)
        frequency = self.frequency / 1000
        new_param_list = []
        id_list = []
        for ind, param in tenumerate(self.traffic_parameters.copy()):
            filter = {}
            # Value of one stopped vehicle for one time step
            filter_value = (1000 / loop_distance[ind])
            tag = [[] for d in range(1, n_det + 1)]
            stp_id = [[] for d in range(1, n_det + 1)]
            f_id = [[] for d in range(1, n_det + 1)]
            f_veh_id = [[] for d in range(1, n_det + 1)]
            f_type = [[] for d in range(1, n_det + 1)]
            stp_cnt = [[0] * len(param[f'stopped_vehicles_{d}'][0]) for d in range(1, n_det + 1)]
            for row, val in param.iterrows():
                for d in range(1, n_det + 1):
                    for id, stop in enumerate(val[f'stopped_vehicles_{d}']):
                        if stp_cnt[d - 1][id] == stop_time:
                            for v in range(stop_time, 0, -1):
                                stp_id[d - 1].append((id, row - v))
                        elif stp_cnt[d - 1][id] > stop_time:
                            stp_id[d - 1].append((id, row - 1))
                        if stop > 0:
                            stp_cnt[d - 1][id] += 1
                        else:
                            stp_cnt[d - 1][id] = 0
            for d in range(1, n_det + 1):
                filter[f'density_{d}'] = list(param[f'density_{d}'].values)
                filter[f'lanes_{d}'] = [lanes[ind]] * len(param)
                filter[f'vehicles_{d}'] = list(param[f'vehicles_{d}'].values)
                filter[f'stops_{d}'] = list(param[f'stops_{d}'].values)
                for id, vt in enumerate(list(collections.Counter(
                        self.traffic_counts['vehicle_type'][f'vehicle_type_{edge_id[ind]}']).keys())):
                    filter[f'{vt}_{d}'] = list(param[f'{vt}_{d}'].values)
                    param = param.drop([f'{vt}_{d}'], axis=1)
                # filter[f'f_{d}'] = [0] * len(param)
                for r, e in enumerate(stp_id[d - 1]):
                    veh = self.traffic_counts['vehicle_type'][f'vehicle_type_{edge_id[ind]}'][e[0]]
                    if e[0] not in f_id[d - 1]:
                        f_id[d - 1].append(e[0])
                        f_veh_id[d - 1].append(self.traffic_counts['vehicle_type']
                                               [f'vehicle_index_{edge_id[ind]}'][e[0]][0])
                        f_type[d - 1].append(veh)
                    filter[f'density_{d}'][e[1]] -= filter_value
                    # filter[f'lanes_{d}'][e[1]] -= 1
                    filter[f'stops_{d}'][e[1]] -= 1
                    filter[f'vehicles_{d}'][e[1]] -= 1
                    veh_share = filter[f'{veh}_{d}'][e[1]] * filter[f'vehicles_{d}'][e[1]] - 1
                    filter[f'{veh}_{d}'][e[1]] = 0
                    if filter[f'vehicles_{d}'][e[1]] > 0:
                        filter[f'{veh}_{d}'][e[1]] = veh_share / filter[f'vehicles_{d}'][e[1]]
                filter[f'density_lane_{d}'] = np.array(filter[f'density_{d}']) / np.array(filter[f'lanes_{d}'])
                filter[f'speed_{d}'] = [
                    param[f'flow_{d}'][k] / filter[f'density_{d}'][k] if filter[f'density_{d}'][k]
                    else 0 for k, v in param.iterrows()]
                param[f'flow_lane_{d}'] = np.array(param[f'flow_{d}']) / np.array(filter[f'lanes_{d}'])
                param = param.drop([f'density_{d}', f'density_lane_{d}', f'modal_shares_{d}',
                                    f'stopped_vehicles_{d}', f'vehicle_area_{d}', f'stops_{d}',
                                    f'vehicles_{d}', f'speed_{d}'], axis=1)
            filter = pd.DataFrame(filter)
            p = pd.concat([param, filter], axis=1)
            for det in range(1, n_det + 1):
                p = p.drop([f'lanes_{det}'], axis=1)
            new_param_list.append(p)
            id_list.append([f_veh_id, f_type])
            self.adjustment_stopped = True, stop_time
        return new_param_list, id_list

    def arterial_parameters(self, aggregation_detector, mode=None, aggregated_parameters=False,
                            adjusted_parameters=False):
        # mode is tuple with (link of arterial, mode, mode share)
        n_det = self.numberofdetectors
        edge_id = list(self.traffic_counts['detectors']['index'].values)
        index_complete_counts = []
        parameters = self.traffic_parameters
        if aggregated_parameters:
            parameters = self.traffic_parameters_agg
        elif adjusted_parameters:
            parameters = self.traffic_parameters_adj
        for ind, param in enumerate(parameters):
            s = []
            for det in range(1, n_det + 1):
                s.append(sum(param[f'vehicles_{det}']))
            mu = np.mean(s)
            if mu * 1.2 < max(s):
                # 20 percent difference between mean of sum of vehicles per time step and maximum of one detector
                # raise Exception(f'Link with no counts for some detectors: check parameter list. Sum of vehicles for every'
                # f'detector is {s} on edge {edge_id[ind]}')
                print('Big differences between detector counts')
            else:
                index_complete_counts.append(ind)
        arterial_length = sum(self.traffic_counts['detectors']['length']
                              * self.traffic_counts['detectors']['lanes_adj'])
        # print(arterial_length)
        parameters_arterial = {'accumulation_arterial': 0, 'production_arterial': 0}
        denominator = 0
        det = aggregation_detector
        if mode is not None:
            par_mode = parameters[edge_id.index(mode[0])]
            ind_mode = list(par_mode[par_mode[f'{mode[1]}_{det}'] > mode[2]].index)
        for ind, param in enumerate(parameters):
            # if ind not in index_complete_counts:
            # print(f'skip: {edge_id[ind]}')
            # continue
            denom = (self.traffic_counts['detectors']['length'].values[ind])
            # * counts_dict['counts']['detectors']['lanes_adj'].values[ind])
            production = 0
            accumulation = 0
            """ 
            for det in range(1, n_det + 1):
                production = production + param[f'flow_{det}']
                accumulation = accumulation + param[f'density_{det}']
            """
            flow = param[f'flow_{det}']
            density = param[f'density_{det}']
            if mode is not None:
                flow = param[f'flow_{det}'][ind_mode]
                density = param[f'density_{det}'][ind_mode]
            production = flow * denom
            accumulation = (density * denom)  # units
            parameters_arterial['accumulation_arterial'] += accumulation
            parameters_arterial['production_arterial'] += production
            denominator = denominator + (denom * n_det)
        # print(denominator)
        parameters_arterial['accumulation_arterial'] = (parameters_arterial['accumulation_arterial']
                                                        / 1000)
        parameters_arterial['production_arterial'] = (parameters_arterial['production_arterial']
                                                      / 1000)
        parameters_arterial['average_speed_arterial'] = (parameters_arterial['production_arterial']
                                                         / parameters_arterial['accumulation_arterial'])
        art_parameters = pd.DataFrame(parameters_arterial)
        if mode is not None:
            art_parameters = [art_parameters, ind_mode]
        return art_parameters

    def arterial_parameters_all(self, aggregated_parameters=False, adjusted_parameters=False, network_wide=False):
        # mode is tuple with (link of arterial, mode, mode share)
        n_det = self.numberofdetectors
        edge_id = list(self.traffic_counts['detectors']['index'].values)
        edge_length = list(self.traffic_counts['detectors']['length'].values)
        loop_distances = list(self.traffic_counts['detectors']['loop_distance'].values)
        index_complete_counts = []
        parameters = self.traffic_parameters
        if aggregated_parameters:
            parameters = self.traffic_parameters_agg
        elif adjusted_parameters:
            parameters = self.traffic_parameters_adj
        for ind, param in enumerate(parameters):
            s = []
            for det in range(1, n_det + 1):
                s.append(sum(param[f'vehicles_{det}']))
            mu = np.mean(s)
            if mu * 1.2 < max(s):
                # 20 percent difference between mean of sum of vehicles per time step and maximum of one detector
                # raise Exception(f'Link with no counts for some detectors: check parameter list. Sum of vehicles for every'
                # f'detector is {s} on edge {edge_id[ind]}')
                print('Big differences between detector counts')
            else:
                index_complete_counts.append(ind)
        arterial_length = sum(self.traffic_counts['detectors']['length']
                              * self.traffic_counts['detectors']['lanes_adj'])
        # print(arterial_length)
        parameters_arterial = {'accumulation_arterial': 0, 'production_arterial': 0}
        denominator = 0
        if network_wide:
            parameters_arterial = {'VHT_arterial': 0, 'VKT_arterial': 0}
            tot_loop_distance = 0
            length_netw = 0
        for ind, param in enumerate(parameters):
            # if ind not in index_complete_counts:
            # print(f'skip: {edge_id[ind]}')
            # continue
            if loop_distances[ind] < self.loop_distance:
                continue
            elif edge_length[ind] >= (self.dfi * 2 + loop_distances[ind]):
                flow = param['flow_1'].copy()
                density = param['density_1'].copy()
                for det in range(2, n_det + 1):
                    flow += param[f'flow_{det}'].copy()
                    density += param[f'density_{det}'].copy()
                avg_flow = flow / n_det
                avg_density = density / n_det
                if network_wide:
                    tot_loop_distance += n_det * loop_distances[ind]
                    length_netw += edge_length[ind]
            else:
                avg_flow = param['flow_1'].copy()
                avg_density = param['density_2'].copy()
                if network_wide:
                    tot_loop_distance += loop_distances[ind]
                    length_netw += edge_length[ind]
            denom = edge_length[ind]
            # * counts_dict['counts']['detectors']['lanes_adj'].values[ind])
            production = 0
            accumulation = 0
            """ 
            for det in range(1, n_det + 1):
                production = production + param[f'flow_{det}']
                accumulation = accumulation + param[f'density_{det}']
            """
            if network_wide:
                VHT = round((avg_density * round(self.frequency / 1000) * loop_distances[ind]) / (3600 * 1000), 5)
                VKT = round((avg_flow * round(self.frequency / 1000) * loop_distances[ind]) / (3600 * 1000), 5)
                parameters_arterial['VHT_arterial'] += VHT
                parameters_arterial['VKT_arterial'] += VKT
            else:
                production = avg_flow * denom
                accumulation = (avg_density * denom)  # units
                parameters_arterial['accumulation_arterial'] += accumulation
                parameters_arterial['production_arterial'] += production
        if network_wide:
            approx = length_netw / tot_loop_distance
            parameters_arterial['accumulation_arterial'] = parameters_arterial['VHT_arterial'] \
                                                           * 3600 / (round(self.frequency / 1000)) * approx
            parameters_arterial['production_arterial'] = (parameters_arterial['VKT_arterial']
                                                          / parameters_arterial['VHT_arterial'] *
                                                          parameters_arterial['accumulation_arterial'])
            parameters_arterial['average_speed_arterial'] = (parameters_arterial['VKT_arterial'] /
                                                             parameters_arterial['VHT_arterial'])
        else:
            parameters_arterial['accumulation_arterial'] = (parameters_arterial['accumulation_arterial']
                                                            / 1000)
            parameters_arterial['production_arterial'] = (parameters_arterial['production_arterial']
                                                          / 1000)
            parameters_arterial['average_speed_arterial'] = (parameters_arterial['production_arterial']
                                                             / parameters_arterial['accumulation_arterial'])
        art_parameters = pd.DataFrame(parameters_arterial)
        return art_parameters

    def fixed_vehicle_aggregation(self, aggregation_number):  # Experiment
        n_det = self.numberofdetectors
        frequency = self.frequency
        agg_parameters = []
        for ind, param in enumerate(self.traffic_parameters.copy()):
            for det in range(1, n_det + 1):
                aggregation_parameters = {'index': [ind + 1], f'density_{det}': [], f'flow_{det}': []}
                max_N = sum(param[f'vehicles_{det}'])
                N = 0
                i = 0
                for veh in range(aggregation_number, max_N + 1, aggregation_number):
                    prev_number = veh - aggregation_number
                    agg_den = 0
                    agg_flow = 0
                    if N and i < len(param['index']):
                        agg_den = N * param[f'density_{det}'][i - 1]
                        agg_flow = N * param[f'flow_{det}'][i - 1]
                        prev_number += N * aggregation_number
                        N = 0
                    while prev_number < veh and i < len(param['index']):
                        prev_number += param[f'vehicles_{det}'][i]
                        if prev_number > veh:  # interpolation
                            x = prev_number - veh
                            y = param[f'vehicles_{det}'][i] - x
                            N = x / aggregation_number
                            agg_den += param[f'density_{det}'][i] * y / aggregation_number
                            agg_flow += param[f'flow_{det}'][i] * y / aggregation_number
                            i += 1
                        else:
                            agg_den += param[f'density_{det}'][i] * param[f'vehicles_{det}'][i] / aggregation_number
                            agg_flow += param[f'flow_{det}'][i] * param[f'vehicles_{det}'][i] / aggregation_number
                            i += 1
                    print(prev_number, agg_den, agg_flow)
                    aggregation_parameters[f'density_{det}'].append(agg_den)
                    aggregation_parameters[f'flow_{det}'].append(agg_flow)
                    if veh > aggregation_number:
                        aggregation_parameters[f'index'].append(0)
                aggregation_parameters = pd.DataFrame(aggregation_parameters)
                agg_parameters.append(aggregation_parameters)
        return agg_parameters

    def travel_time_distribution(self):
        n_det = self.numberofdetectors
        dfi = self.dfi
        edge_id = list(self.traffic_counts['detectors']['index'].values)
        if 'arterial_order' in list(self.traffic_counts['detectors']):
            order_id = list(self.traffic_counts['detectors']['arterial_order'].values)
        else:
            order_id = edge_id
        # Take into account dfi to analyze distance traveled
        tt_arterial = []
        for ind, counts in enumerate(self.traffic_counts['counts']):
            length = self.traffic_counts['detectors']['length'].values[ind] - dfi * 2
            if self.traffic_counts['detectors']['length'].values[ind] < (2 * dfi + n_det *
                                                                         self.traffic_counts['detectors'][
                                                                             'loop_distance'].values[ind]):
                length = self.traffic_counts['detectors']['loop_distance'].values[ind]
            times = counts[['times_1', f'times_lp_{n_det}']].values
            times_traj = [[0] * 3 for i in range(0, len(times[0][0]))]
            period = [0] * len(times[0][0])
            cross = [False] * len(times[0][0])
            for step in range(0, len(times)):
                for traj in range(0, len(times[0][0])):
                    if times[step][0][traj] and not cross[traj]:
                        times_traj[traj][0] = times[step][0][traj]
                        cross[traj] = True
                    if times[step][1][traj]:
                        times_traj[traj][1] = times[step][1][traj]
                        period[traj] = step
            traj_tt = [times_traj[traj][1] - times_traj[traj][0] for traj in range(0, len(times_traj))]
            travel_time = {f'travel_time_{order_id[ind]}': traj_tt,
                           f'start_time_{order_id[ind]}': [times_traj[traj][0] for traj in range(0, len(times_traj))],
                           f'time_step_{order_id[ind]}': period,
                           'vehicle_type': self.traffic_counts['vehicle_type'][f'vehicle_type_{edge_id[ind]}'],
                           'ID': self.traffic_counts['vehicle_type'][f'vehicle_index_{edge_id[ind]}']}
            travel_time = pd.DataFrame(travel_time)
            travel_time = travel_time[(travel_time[f'travel_time_{order_id[ind]}'] > 0) &
                                      (travel_time[f'start_time_{order_id[ind]}'] > 0)]
            travel_time = travel_time.assign(tt_seconds=round(travel_time[f'travel_time_{order_id[ind]}'] / 1000, 3))
            travel_time = travel_time.assign(tt_seconds_unit=round(travel_time[f'travel_time_{order_id[ind]}'] /
                                                                   (1000 * length), 3))
            travel_time = travel_time.assign(speed=round(length * 3.6 / travel_time[f'tt_seconds'], 3))
            # Averaging and smoothing
            tt_avg = {'tt_step': [], 'speed_step': []}
            for val in range(0, len(times)):
                tt = np.mean(
                    travel_time[travel_time[f'time_step_{order_id[ind]}'] == val][f'travel_time_{order_id[ind]}'])
                speed = np.mean(travel_time[travel_time[f'time_step_{order_id[ind]}'] == val]['speed'])
                if np.isnan(tt):
                    tt = 0
                    speed = 0
                tt_avg['tt_step'].append(tt)
                tt_avg['speed_step'].append(speed)
            tt_avg = pd.DataFrame(tt_avg)
            tt_arterial.append([travel_time, tt_avg])
        return tt_arterial

    def aggregation_step(self, step, adjusted=False):
        n_det = self.numberofdetectors
        edge_id = list(self.traffic_counts['detectors']['index'].values)
        parameters_agg = []
        parameters = self.traffic_parameters
        if adjusted:
            parameters = self.traffic_parameters_adj
        for ind, param in enumerate(parameters):
            index = [0] * (len(param) // step + (len(param) % step > 0))
            index[0] = edge_id[ind]
            agg_df = {'index': index}
            vehicle_type = self.traffic_counts['vehicle_type'][f'vehicle_type_{edge_id[ind]}']
            get_veh_type = collections.Counter(vehicle_type)
            for det in range(1, n_det + 1):
                agg_df[f'density_{det}'] = []
                agg_df[f'flow_{det}'] = []
                agg_df[f'speed_{det}'] = []
                agg_df[f'vehicles_{det}'] = []
                agg_df[f'stops_{det}'] = []
                for veh_type in list(get_veh_type):
                    agg_df[f'{veh_type}_{det}'] = []
            for time_step in range(step, len(param) + step, step):
                pre_interval = time_step - step
                for det in range(1, n_det + 1):
                    k = 0
                    q = 0
                    veh = 0
                    dict_veh = {k: 0 for k in list(get_veh_type)}
                    stop = []
                    for ind, val in param.iterrows():
                        if pre_interval <= ind < time_step:
                            k += val[f'density_{det}']
                            q += val[f'flow_{det}']
                            veh += val[f'vehicles_{det}']
                            for veh_type in list(get_veh_type):
                                dict_veh[f'{veh_type}'] += val[f'{veh_type}_{det}'] * val[f'vehicles_{det}']
                            stop.append(val[f'stops_{det}'])
                    agg_df[f'density_{det}'].append(k / step)
                    agg_df[f'flow_{det}'].append(q / step)
                    if k:
                        agg_df[f'speed_{det}'].append(round(q / k, 1))
                        for veh_type in list(get_veh_type):
                            agg_df[f'{veh_type}_{det}'].append(round(dict_veh[f'{veh_type}'] / veh, 1))
                    else:
                        agg_df[f'speed_{det}'].append(0)
                        for veh_type in list(get_veh_type):
                            agg_df[f'{veh_type}_{det}'].append(0)
                    agg_df[f'vehicles_{det}'].append(veh)
                    agg_df[f'stops_{det}'].append(stop)
            agg_df = pd.DataFrame(agg_df)
            parameters_agg.append(agg_df)
            self.traffic_parameters_agg = parameters_agg

    def matching_trajectories_arterial(self, agg_interval=6, selected_art_sections=None):
        tt = self.travel_time_distribution()
        edge_id = list(self.traffic_counts['detectors']['index'].values)
        if 'arterial_order' in list(self.traffic_counts['detectors']):
            edge_id = list(self.traffic_counts['detectors']['arterial_order'].values)
        if selected_art_sections is None:
            selected_art_sections = (0, len(edge_id))
        dfi = self.dfi
        list_length = list(self.traffic_counts['detectors']['length'].values)
        length_art = 0
        for id, l in enumerate(list_length):
            if id == selected_art_sections[0] and selected_art_sections[1] - selected_art_sections[0] == 1:
                # When only one edge is selected, dfi has to be subtracted two times
                if l > (dfi * 2 + self.numberofdetectors * self.loop_distance):
                    length_art += l - 2 * dfi
                else:
                    length_art += l / 2 + self.loop_distance / 2
            elif id == selected_art_sections[0]:  # First element
                if l > (dfi * 2 + self.numberofdetectors * self.loop_distance):
                    length_art += l - dfi
                else:
                    length_art += l / 2 + self.loop_distance / 2
            elif id == selected_art_sections[1] - 1:  # Last element
                if l > (dfi * 2 + self.numberofdetectors * self.loop_distance):
                    length_art += l - dfi
                else:
                    length_art += l / 2 + self.loop_distance / 2
            elif selected_art_sections[0] < id < selected_art_sections[1] - 1:
                length_art += l
            else:
                continue
        print(length_art)
        list_df = [j[0] for i, j in enumerate(tt) if selected_art_sections[0] <= i < selected_art_sections[1]]
        traj_df = list_df[0].copy()
        for ind, val in enumerate(list_df):
            sec = ind + selected_art_sections[0]
            if ind > 0:
                traj_df = pd.merge(traj_df, val[['ID', 'vehicle_type', f'travel_time_{edge_id[sec]}',
                                                 f'start_time_{edge_id[sec]}', f'time_step_{edge_id[sec]}']]
                                   , how='inner', on=['ID', 'vehicle_type'])
        traj_df = traj_df.assign(tt_art=(traj_df[f'start_time_{edge_id[selected_art_sections[1] - 1]}'] -
                                         traj_df[f'start_time_{edge_id[selected_art_sections[0]]}']
                                         + traj_df[f'travel_time_{edge_id[selected_art_sections[1] - 1]}']))
        traj_df = traj_df.assign(tt_art_seconds=traj_df.tt_art / 1000)
        traj_df = traj_df.assign(tt_art_unit=round(traj_df.tt_art / (length_art * 1000), 2))
        traj_df = traj_df.assign(speed_art=round(length_art / traj_df['tt_art'] * 3600, 1))
        traj_df_sort = traj_df.sort_values(by=f'time_step_{edge_id[selected_art_sections[1] - 1]}').copy()
        art_agg_df = {'time_step': [], 'tt_tot': [], 'tt_tot_seconds': [], 'speed_tot': [], 'vehicle_type': [],
                      'ID': []}
        for agg in range(agg_interval, len(self.traffic_parameters[0]) + agg_interval + 1, agg_interval):
            pre_interval = agg - agg_interval
            for ind, val in traj_df_sort[f'time_step_{edge_id[selected_art_sections[1] - 1]}'].iteritems():
                if pre_interval <= val < agg:
                    art_agg_df['time_step'].append(agg / agg_interval)
                    art_agg_df['tt_tot'].append(traj_df_sort['tt_art'][ind])
                    art_agg_df['tt_tot_seconds'].append(traj_df_sort['tt_art_seconds'][ind])
                    art_agg_df['speed_tot'].append(traj_df_sort['speed_art'][ind])
                    art_agg_df['vehicle_type'].append(traj_df_sort['vehicle_type'][ind])
                    art_agg_df['ID'].append(traj_df_sort['ID'][ind])
        art_agg_df = pd.DataFrame(art_agg_df)
        return traj_df, art_agg_df

    def plot_fd(self, colors=None, veh_area_colorbar=False, summary=True, individual_fd=False,
                save_figures=True, per_lane=True, filter_stopped=False, max_lim_flow=0, max_lim_speed=0, filename=0):
        figures = []
        n_det = self.numberofdetectors
        distance = self.dfi
        edge_id = self.traffic_counts['detectors']['index'].values
        loop_distance = self.traffic_counts['detectors']['loop_distance'].values
        parameters = self.traffic_parameters
        if filter_stopped:
            parameters = self.traffic_parameters_adj
            veh_area_colorbar = False
        str_lane = '_'
        str_legend = ''
        if per_lane:
            str_lane = '_lane_'
            str_legend = ' per lane'
        if not colors:
            colors = ['b', 'k']
        markersize = 5
        f_size = (12, 7)
        max_q = max(
            [max(j[f'flow{str_lane}{det}']) for det in range(1, n_det + 1)
             for i, j in enumerate(parameters)]) + 100
        max_v = max([max(j[f'speed_{det}']) for det in range(1, n_det + 1)
                     for i, j in enumerate(parameters)]) + 5
        max_lim_flow = max(max_q, max_lim_flow)
        max_lim_speed = max(max_v, max_lim_speed)
        for i, j in enumerate(parameters):
            if filename != 0:
                name_file = f'{filename} {i + 1}'
            else:
                name_file = f'edge {edge_id[i]}'
            if summary:
                fig, axes = plt.subplots(nrows=2, ncols=n_det, sharex='col', sharey='row', figsize=f_size)
                axes_list = [item for sublist in axes for item in sublist]
                m_q = max([max(j[f'flow{str_lane}{det}']) for det in range(1, n_det + 1)])
                m_v = max([max(j[f'speed_{det}']) for det in range(1, n_det + 1)])
                m_k = max([max(j[f'density{str_lane}{det}']) for det in range(1, n_det + 1)])
                diff_det = n_det
                for det in range(1, n_det + 1):
                    ax_1 = axes_list.pop(0)
                    ax_2 = axes_list.pop(diff_det - 1)
                    ax_1.scatter(j[f'density{str_lane}{det}'], j[f'flow{str_lane}{det}'],
                                 color=colors[1], s=markersize)
                    ax_2.scatter(j[f'density{str_lane}{det}'], j[f'speed_{det}'],
                                 color=colors[1], s=markersize)
                    if det == 1:
                        ax_1.set_ylabel(f'Flow q [veh/h{str_legend}]')
                        ax_2.set_ylabel('Speed v [km/h]')
                    ax_1.set_ylim(bottom=-50, top=m_q + 200)
                    ax_2.set_ylim(bottom=-2.5, top=m_v + 10)
                    ax_2.set_xlim(left=-10, right=m_k + 50)
                    ax_1.set_title(f'Detector {det}')
                    ax_2.set_xlabel(f'Density k [veh/km{str_legend}]')
                    ax_1.grid(True)
                    ax_2.grid(True)
                    diff_det -= 1
                fig.suptitle(f"FDs for {name_file} \n" + f"(loop distance: {loop_distance[i]} m, time step: "
                                                         f"{int(self.frequency / 1000)} "
                                                         f"sec, dfi: {distance} m)", fontsize=16, wrap=True)
                fig.align_ylabels()
                plt.tight_layout(rect=[0, 0.03, 1, 0.90])
                if save_figures:
                    plt.savefig(f"FD_summary_{name_file}_{int(self.frequency / 1000)}_lp"
                                f"{loop_distance[i]}_d{distance}.png")
            if not individual_fd:
                continue
            for det in range(1, n_det + 1):
                if veh_area_colorbar:
                    plt.figure()
                    kqFD = plt.scatter(j[f'density{str_lane}{det}'], j[f'flow{str_lane}{det}']
                                       , c=j[f'vehicle_area_{det}'])
                    plt.clim(0, 15)
                    cb1 = plt.colorbar()
                    cb1.set_label('Average vehicle area')
                    # plt.legend(loc='upper left')
                    plt.grid(True)
                    plt.title(f"FD flow-density {name_file} \n" +
                              f"(Detector {det}: loop distance {loop_distance[i]} m, time step: "
                              f"{int(self.frequency / 1000)} "
                              f"sec, dfi: {distance} m)", wrap=True)
                    plt.xlabel(f'Density k [veh/km{str_legend}]')
                    plt.ylabel(f'Flow q [veh/h{str_legend}]')
                    plt.axis([- 20, int(max(j[f'density{str_lane}{det}']) + 50),
                              - 50, max_lim_flow])
                    plt.tight_layout()
                    if save_figures:
                        plt.savefig(f"FD_kq_veh_area_det{det}_{name_file}_"
                                    f"{int(self.frequency / 1000)}_lp"
                                    f"{loop_distance[i]}_d{distance}.png")
                    plt.figure()
                    kuFD = plt.scatter(j[f'density{str_lane}{det}'], j[f'speed_{det}'],
                                       c=j[f'vehicle_area_{det}'])
                    plt.clim(0, 15)
                    cb2 = plt.colorbar()
                    cb2.set_label('Average vehicle area')
                    # plt.legend(loc='upper left')
                    plt.grid(True)
                    plt.title(f"FD speed-density {name_file} \n" +
                              f"(Detector {det}: loop distance {loop_distance[i]} m, time step: "
                              f"{int(self.frequency / 1000)} "
                              f"sec, dfi: {distance} m)", wrap=True)
                    plt.xlabel(f'Density k [veh/km{str_legend}]')
                    plt.ylabel('Speed v [km/h]')
                    plt.axis([- 20, int(max(j[f'density{str_lane}{det}']) + 50),
                              - 5, max_lim_speed])
                    plt.tight_layout()
                    if save_figures:
                        plt.savefig(f"FD_kv_veh_area_det{det}_{name_file}_"
                                    f"{int(self.frequency / 1000)}_lp"
                                    f"{loop_distance[i]}_d{distance}.png")
                else:
                    plt.figure()
                    kqFD = plt.scatter(j[f'density{str_lane}{det}'], j[f'flow{str_lane}{det}'],
                                       color=colors[0])
                    # plt.legend(loc='upper left')
                    plt.grid(True)
                    plt.title(f"FD flow-density edge {name_file} \n" +
                              f"(Detector {det}: loop distance {loop_distance[i]} m), time step: "
                              f"{int(self.frequency / 1000)} "
                              f"sec, dfi: {distance} m)", wrap=True)
                    plt.xlabel(f'Density k [veh/km{str_legend}]')
                    plt.ylabel(f'Flow q [veh/h{str_legend}]')
                    plt.axis([- 20, int(max(j[f'density{str_lane}{det}']) + 50),
                              - 50, max_lim_flow])
                    plt.tight_layout()
                    if save_figures:
                        plt.savefig(f"FD_kq_det{det}_{name_file}_"
                                    f"{int(self.frequency / 1000)}_lp"
                                    f"{loop_distance[i]}_d{distance}.png")
                    plt.figure()
                    kuFD = plt.scatter(j[f'density{str_lane}{det}'],
                                       j['speed_' + str(det)], color=colors[0])
                    # plt.legend(loc='upper left')
                    plt.grid(True)
                    plt.title(f"FD speed-density edge {name_file} \n" +
                              f"(Detector {det}: loop distance {loop_distance[i]} m, time step: "
                              f"{int(self.frequency / 1000)} "
                              f"sec, dfi: {distance} m))", wrap=True)
                    plt.xlabel(f'Density k [veh/km{str_legend}]')
                    plt.ylabel('Speed v [km/h]')
                    plt.axis([- 20, int(max(j[f'density{str_lane}{det}']) + 50),
                              - 5, max_lim_speed])
                    plt.tight_layout()
                    if save_figures:
                        plt.savefig(f"FD_kv_det{det}_{name_file}_"
                                    f"{int(self.frequency / 1000)}_lp"
                                    f"{loop_distance[i]}_d{distance}.png")
                figures.append([kqFD, kuFD])
        return figures

    min_length = 45  # For this research: 2 * distance from intersection + number of det * loop distance

    def arterial_statistics(self, minimal_length=min_length, art_name='arterial', aggregated_parameters=False,
                            adjusted_parameters=False, det_short_edges=1):
        art_dict = []
        parameters = self.traffic_parameters
        edge_length = list(self.traffic_counts['detectors']['length'].values)
        lanes = list(self.traffic_counts['detectors']['lanes_adj'].values)
        n_det = self.numberofdetectors
        if aggregated_parameters:
            parameters = self.traffic_parameters_agg
        elif adjusted_parameters:
            parameters = self.traffic_parameters_adj
        for ind, par in enumerate(parameters):
            if edge_length[ind] > minimal_length:
                df = par[['density_1', 'density_lane_1', 'flow_1', 'flow_lane_1', 'speed_1']].copy()
                for d in range(2, n_det + 1):
                    df = pd.concat([df, par[[f'density_{d}', f'density_lane_{d}', f'flow_{d}', f'flow_lane_{d}',
                                             f'speed_{d}']]], axis=1)
                ndf = df.describe()
                for d in range(1, n_det + 1):
                    ndf = ndf.rename(columns={f'density_{d}': f'{art_name}{ind + 1}_k{d}',
                                              f'density_lane_{d}': f'{art_name}{ind + 1}_lk{d}',
                                              f'flow_{d}': f'{art_name}{ind + 1}_q{d}',
                                              f'flow_lane_{d}': f'{art_name}{ind + 1}_lq{d}',
                                              f'speed_{d}': f'{art_name}{ind + 1}_v{d}'})
                art_dict.append(ndf)
            else:
                df = par[[f'density_{det_short_edges}', f'density_lane_{det_short_edges}', f'flow_{det_short_edges}',
                          f'flow_lane_{det_short_edges}', f'speed_{det_short_edges}']].copy()
                ndf = df.describe()
                ndf = ndf.rename(columns={f'density_{det_short_edges}': f'{art_name}{ind + 1}_k{det_short_edges}',
                                          f'density_lane_{det_short_edges}': f'{art_name}{ind + 1}_lk{det_short_edges}',
                                          f'flow_{det_short_edges}': f'{art_name}{ind + 1}_q{det_short_edges}',
                                          f'flow_lane_{det_short_edges}': f'{art_name}{ind + 1}_lq{det_short_edges}',
                                          f'speed_{det_short_edges}': f'{art_name}{ind + 1}_v{det_short_edges}'})
                art_dict.append(ndf)
        art_df = pd.concat(art_dict, axis=1)
        return art_df

    def share_ctmb(self):
        edge_id = list(self.traffic_counts['detectors']['index'].values)
        share = {'Car': [], 'Taxi': [], 'PTW': [], 'Bus': [], 'Bus_abs': []}
        for ind, ed in enumerate(edge_id):
            total = len(self.traffic_counts['vehicle_type'][f'vehicle_type_{edge_id[ind]}'])
            cnt = collections.Counter(self.traffic_counts['vehicle_type'][f'vehicle_type_{edge_id[ind]}'])
            share['Car'].append(round(cnt['Car'] / total * 100, 1))
            share['Taxi'].append(round(cnt['Taxi'] / total * 100, 1))
            share['PTW'].append(round(cnt['Motorcycle'] / total * 100, 1))
            share['Bus'].append(round(cnt['Bus'] / total * 100, 1))
            share['Bus_abs'].append(cnt['Bus'])
        share = pd.DataFrame(share)
        return share


def fit_curves(df_x, df_y, degree=2):
    k = df_x[:, np.newaxis]
    q = df_y[:, np.newaxis]
    if degree > 1:
        polynomial_features = PolynomialFeatures(degree=degree)
        k_poly = polynomial_features.fit_transform(k)
    else:
        k_poly = k
    model = LinearRegression()
    model.fit(k_poly, q)
    q_poly_pred = model.predict(k_poly)
    if degree > 1:
        # sort the values of x before line plot
        sort_axis = operator.itemgetter(0)
        sorted_zip = sorted(zip(k, q_poly_pred), key=sort_axis)
        k, q_poly_pred = zip(*sorted_zip)
    r2 = r2_score(df_y, q_poly_pred)
    return k, q_poly_pred, r2


def afd_estimates(list_arterial, arterial_names, specific_detector=0, aggregated=False, adjusted=False,
                  step_agg=6, stop_time=3):
    afd = []
    for ind, art in enumerate(list_arterial):
        if adjusted:
            art.traffic_parameters_adj = art.adjustment_stopped_vehicles(stop_time)
        if aggregated:
            art.aggregation_step(step=step_agg, adjusted=adjusted)
        if specific_detector:
            afd.append(art.arterial_parameters(specific_detector,
                                               aggregated_parameters=aggregated, adjusted_parameters=adjusted))
        else:
            afd.append(art.arterial_parameters_all(aggregated_parameters=aggregated, adjusted_parameters=adjusted))
    fig, ax = plt.subplots(nrows=2, ncols=int(len(afd) / 2))
    for el, val in enumerate(afd):
        if el < 4:
            ax[0, el].scatter(val.accumulation_arterial, val.average_speed_arterial, color='k', s=10)
            ax[0, el].set_title(arterial_names[el])
            ax[0, el].set_xlim(0, )
            ax[0, el].set_ylim(0, )
            ax[0, el].set_xlabel('Accumulation [veh]')
            ax[0, el].set_ylabel('Average speed [km/h]')
        else:
            ax[1, el - 4].scatter(val.accumulation_arterial, val.average_speed_arterial, color='k', s=10)
            ax[1, el - 4].set_title(arterial_names[el])
            ax[1, el - 4].set_xlim(0, )
            ax[1, el - 4].set_ylim(0, )
            ax[1, el - 4].set_xlabel('Accumulation [veh]')
            ax[1, el - 4].set_ylabel('Average speed [km/h]')
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig, ax = plt.subplots(nrows=2, ncols=int(len(afd) / 2))
    for el, val in enumerate(afd):
        if el < 4:
            ax[0, el].scatter(val.accumulation_arterial, val.production_arterial, color='k', s=10)
            ax[0, el].set_title(arterial_names[el])
            ax[0, el].set_xlim(0, )
            ax[0, el].set_ylim(0, )
            ax[0, el].set_xlabel('Accumulation [veh]')
            ax[0, el].set_ylabel('Production [veh km/h]')
        else:
            ax[1, el - 4].scatter(val.accumulation_arterial, val.production_arterial, color='k', s=10)
            ax[1, el - 4].set_title(arterial_names[el])
            ax[1, el - 4].set_xlim(0, )
            ax[1, el - 4].set_ylim(0, )
            ax[1, el - 4].set_xlabel('Accumulation [veh]')
            ax[1, el - 4].set_ylabel('Production [veh km/h]')
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    return afd


def mfd_estimation(list_arterials, aggregated=False, step_agg=6, adjusted=False, stop_time=3, specific_detector=0):
    mfd = []
    length_network = []
    for i, j in enumerate(list_arterials):
        length_network.append(sum(j.traffic_counts['detectors']['length'] * j.traffic_counts['detectors']['lanes_adj']))
        if adjusted:
            j.traffic_parameters_adj = j.adjustment_stopped_vehicles(stop_time=stop_time)
        if aggregated:
            j.aggregation_step(step_agg, adjusted=adjusted)
        if specific_detector:
            mfd.append(j.arterial_parameters(specific_detector,
                                             adjusted_parameters=adjusted, aggregated_parameters=aggregated))
        else:
            mfd.append(j.arterial_parameters_all(adjusted_parameters=adjusted, aggregated_parameters=aggregated))
    mfd_dict = {'accumulation': mfd[0].accumulation_arterial,
                'production': mfd[0].production_arterial}
    for i, j in enumerate(mfd):
        if i > 0:
            mfd_dict['accumulation'] += j.accumulation_arterial
            mfd_dict['production'] += j.production_arterial
    mfd_dict['average_speed'] = mfd_dict['production'] / mfd_dict['accumulation']
    mfd_df = pd.DataFrame(mfd_dict)
    return mfd_df


def tt_art_all(list_arterials, art_names, plot_tt=False, plot_tt_art=False, column='tt_seconds_unit'):
    tt_art = []
    len_art = []
    for i, j in enumerate(list_arterials):
        len_art.append(len(j.traffic_counts['detectors']))
        tt_art.append([j.travel_time_distribution(), j.matching_trajectories_arterial()])
    if plot_tt:
        for i, j in enumerate(tt_art):
            fig, ax = plt.subplots()
            val_mean = []
            val_95 = []
            val_c = []
            val_b = []
            val_t = []
            val_ptw = []
            for d in range(0, len_art[i]):
                df = j[0][d][0]
                df_c = df[df.vehicle_type == 'Car']
                df_b = df[df.vehicle_type == 'Bus']
                df_t = df[df.vehicle_type == 'Taxi']
                df_ptw = df[df.vehicle_type == 'Motorcycle']
                val_mean.append(df[column].mean())
                val_c.append(df_c[column].mean())
                val_b.append(df_b[column].mean())
                val_t.append(df_t[column].mean())
                val_ptw.append(df_ptw[column].mean())
                val_95.append(np.percentile(df[column], 95))
            ax.plot(val_mean, marker='o', color='k')
            # ax.plot(val_95, marker='o', color='r')
            ax.plot(val_c, marker='o', color='r', label='Car')
            ax.plot(val_b, marker='o', color='orange', label='Bus')
            ax.plot(val_t, marker='o', color='b', label='Taxi')
            ax.plot(val_ptw, marker='o', color='g', label='PTW')
            ax.legend()
            ax.set_title(art_names[i])
            ax.set_ylabel('Unit travel time [s/m]')
            ax.set_xlabel('Consecutive sections')
            ax.grid(True)
            ax.set_xticks([])
    return tt_art


# def arterial_fundamental_diagram(parameters_list):
# afd = []
# for a, b in enumerate(parameters_list):

# parameters = calculate_parameters(detector_counts, used_network)
# breakpoint()
"""
Calculate parameters:
- specified edges of detector counts
- with or without specific vehicle types
- with or without area adjustment
"""
### Omonoia arterial ###
# For different selections of vehicle types
# parameters = calculate_parameters(detector_counts, used_network)
# parameters_cb = calculate_parameters(detector_counts_cb, used_network)
# parameters_no_t = calculate_parameters(detector_counts_t, used_network)
# parameters_ao = calculate_parameters_area(detector_counts, used_network, lane_width)
"""
labels_cb = ['Original', 'All Modes', 'No PTWs', 'Adjusted']
labels_t = ['Original', 'All Modes', 'No Taxis', 'Adjusted']
#parameters_plots_modes_cb = save_plot_fd(detector_counts, parameters, parameters_list_mode_sel=parameters_cb,
                                         #labels=labels_cb)
#parameters_plots_modes_t = save_plot_fd(detector_counts, parameters, parameters_list_mode_sel=parameters_no_t,
                                        #labels=labels_t)
#parameters_plots_comparison = save_plot_fd(detector_counts, parameters_ao)
step = 0.5
lane_width = np.arange(3, 4+step, step).tolist()  # maximum width of identified vehicle, bus, in dataset
p_ao = []
for i, j in enumerate(lane_width):
    parameters_ao = calculate_parameters_area(detector_counts, used_network, j)
    p_ao.append(parameters_ao)
m = ['.', '^','s']
c = ['b', 'r', 'darkgreen']

for det in range(1,4):
    fig, ax = plt.subplots()
    ax.scatter(p_ao[0][0]['occupancy_' + str(det)], p_ao[0][0]['flow_occ_' + str(det)], color=c[0], marker=m[0])
    ax.scatter(p_ao[1][0]['occupancy_' + str(det)], p_ao[1][0]['flow_occ_' + str(det)], color=c[1],
               marker=m[1])
    ax.scatter(p_ao[2][0]['occupancy_' + str(det)], p_ao[2][0]['flow_occ_' + str(det)], color=c[2],
               marker=m[2])
    plt.show()
for key, value in detector_counts[2].items():
    print(key)
    print(collections.Counter(value))

end = time.time()
print(str(int(divmod(end - start, 60)[0])) + 'min' + str(int(divmod(end - start, 60)[1])) + 'sec')
"""
# with open('parameters_'+str(int(detector_counts[2]['frequency'][0]/1000)) +
# '_omonoia.pkl','wb') as a:
# pickle.dump(parameters,a)
# with open('parameters_'+str(int(detector_counts_cb[2]['frequency'][0]/1000)) +
# 'no_t_omonoia.pkl','wb') as a:
# pickle.dump(parameters_cb, a)
# with open('parameters_cb_traj_test_set.pkl', 'wb') as a:
# pickle.dump(parameters,a)
