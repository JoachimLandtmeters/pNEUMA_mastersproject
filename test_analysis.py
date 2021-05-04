"""
Test file for analysis of pNEUMA data
"""
from pneumapackage.settings import *
import pneumapackage.compute as cp
from pneumapackage.__init__ import read_pickle, write_pickle, path_data, path_results
import pneumapackage.iodata as rd

import test_network as tn
import test_data as td

import numpy as np
import pandas as pd
import leuvenmapmatching.util.dist_euclidean as distxy

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from tqdm.contrib import tenumerate
from tqdm import tqdm
import os

"""
Up until now we have list of dataframes, trajectories, with a column with the matched edge in the extracted OSM network
The line trajectories can be used to count vehicles crossing specific locations in the network, keeping the individual
information for more specific aggregations afterwards (augmented loop detector data)

Place detectors on the edges in the used network and select specific edges
--> using Qgis to select manually the needed edge ids

Input parameters:
- 20 m = width of detector edges, make sure they span the whole road
- 10 m = distance from intersection
- True = place double virtual loops
- 1 m = loop distance
- 2 = number of detectors on every link
"""


def test_crossings(line_traj, df_det, **kwargs):
    df_crossings = cp.vehicle_crossings(line_traj, df_det, **kwargs)
    return df_crossings


def get_traj_crossings(track, df_crossings):
    df_crossings = df_crossings[~df_crossings[track].isna()][track]
    return df_crossings


def get_vehicle_types(group_id):
    pid, _, _ = td.get_hdf_names(group_id)
    hdf_path = rd.get_hdf_path()
    vehicle_types = rd.get_from_hdf(hdf_path, key_id=pid, result='ids')
    vehicle_types = vehicle_types.loc[:, ['track_id', 'type']]
    return vehicle_types


def case_configuration(group_id, det_obj, edges):
    """
    Get all the needed information of the detectors for the chosen edges, as well as only those trajectories that map
    onto one of the edges.
    Parameters
    ----------
    group_id
    det_obj
    edges

    Returns
    -------

    """
    ds = det_obj.detector_selection(edges)
    id_ft_pan = list(set(det_obj.features.index.get_level_values(0)) & set(edges))
    id_ft_pan.sort()
    ds_ft = det_obj.features.loc[(id_ft_pan,)]
    ds_ft.attrs = det_obj.features.attrs
    lt = td.get_lt(group_id=group_id, edges=edges, gdf=True)
    return ds, ds_ft, lt


def edges_crossings(group_id, crossing_edges, case_number=1, det_obj=None, bearing_difference=90, strict_match=True,
                    folder=path_results):
    dataset_name = rd.get_path_dict()['groups'][group_id].replace('/', '_')
    try:
        df_ft = read_pickle(f'features_crossing_{dataset_name}_bd{bearing_difference}_case{case_number}',
                            os.path.join(folder, 'crossings'))
        df_det = read_pickle(f'detectors_crossing_{dataset_name}_bd{bearing_difference}_case{case_number}',
                             os.path.join(folder, 'crossings'))
    except FileNotFoundError:
        if det_obj is None:
            det_obj = tn.test_detectors(tn.test_network(), path_data)
        ds, ds_ft, lt = case_configuration(group_id, det_obj, edges=crossing_edges)
        # Determine crossings
        df_ft = cp.vehicle_crossings(lt, ds_ft, bearing_difference=bearing_difference, strict_match=strict_match)
        df_det = cp.vehicle_crossings(lt, ds, bearing_difference=bearing_difference, strict_match=strict_match)
        write_pickle(df_ft, f'features_crossing_{dataset_name}_bd{bearing_difference}_case{case_number}',
                     os.path.join(folder, 'crossings'))
        write_pickle(df_det, f'detectors_crossing_{dataset_name}_bd{bearing_difference}_case{case_number}',
                     os.path.join(folder, 'crossings'))
    return df_ft, df_det, dataset_name


def signal_timings(df_crossings, time_rows=('t1', 't2'), time_step=1000):
    df_det = df_crossings.sort_index()
    df_det = df_det.reset_index()
    df_sel = df_det.loc[df_det['detector'].isin(list(time_rows))]
    df_sel.set_index(['edge', 'detector'], inplace=True)
    df_sel = df_sel.transpose()
    max_time = int(max(df_sel.max()) + time_step)
    df_cycle = {'time_step': [], 'passing': []}
    for t in tqdm(range(time_step, max_time, time_step)):
        df = df_sel[(df_sel >= (t - time_step)) & (df_sel < t)]
        df_cycle['time_step'].append(t), df_cycle['passing'].append(df.count().values)
    df_cycle = pd.DataFrame(df_cycle['passing'], index=df_cycle['time_step'], columns=df_sel.columns)
    df_cycle.index.name = 'time_step'
    # df_cycle = df_st.mask(df_st > 0, 1)
    # df_cum = df_st.cumsum()
    return df_cycle


def cycle_times(df_cycle, edge, column=None, thresh_filter=10000, filter_step=3, thresh=5000):
    if column is None:
        column = 't2'
    tmp = df_cycle.loc[:, edge].copy()
    tmp.loc[:, 'green'] = 0
    tmp.loc[:, 'edge'] = edge
    step = list(set(np.diff(tmp.index)))[0]
    tmp2 = tmp.loc[list(set(np.r_[tmp[tmp[column] > 0].index, tmp.index[0], tmp.index[-1]]))].copy()
    tmp2.sort_index(inplace=True)
    tmp2.reset_index(inplace=True)
    tmp2.loc[:, 'filter_b'] = tmp2.loc[:, 'time_step'].diff(filter_step)
    tmp2.loc[:, 'filter_a'] = abs(tmp2.loc[:, 'time_step'].diff(-filter_step))
    filter_index = tmp2.loc[(tmp2.filter_b > thresh_filter) & (tmp2.filter_a > thresh_filter)].index
    tmp2.loc[filter_index, 'filter_b'] = 0
    tmp2.loc[filter_index, 'filter_a'] = 0
    tmp2 = tmp2[~tmp2.index.isin(tmp2[(tmp2.filter_a == 0) & (tmp2.filter_b == 0)].index)]
    tmp2.loc[:, 'before'] = tmp2.loc[:, 'time_step'].diff(1)
    tmp2.loc[:, 'after'] = abs(tmp2.loc[:, 'time_step'].diff(-1))
    tmp2.loc[tmp2.before <= thresh, 'before'] = 0
    tmp2.loc[tmp2.after <= thresh, 'after'] = 0
    tmp2 = tmp2.loc[tmp2[column] > 0]
    tmp2.loc[:, 'green_start'] = 0
    tmp2.loc[:, 'green_end'] = 0
    tmp2.loc[tmp2.before > 0, 'green_start'] = 1
    tmp2.loc[tmp2.after > 0, 'green_end'] = 1
    tmp2 = tmp2[tmp2.index.isin(tmp2[(tmp2.green_start > 0) | (tmp2.green_end > 0)].index)]
    if len(tmp2.loc[(tmp2.green_start > 0) & (tmp2.green_end > 0)]):
        print('Adjust filters')
        raise ValueError('Invalid instances detected')
    tmp2 = tmp2[~tmp2.index.isin(tmp2[(tmp2.green_start > 0) & (tmp2.green_end > 0)].index)]
    tmp2.set_index('time_step', inplace=True)
    tmp2.loc[:, 'green_time'] = 0
    tmp2.loc[:, 'red_time'] = tmp2.before
    index_greens = []
    ls_tmp = []
    row = 0
    for i, j in tmp2.iterrows():
        if row == 0:
            if j['green_end'] > 0:
                index_greens.extend(np.arange(tmp.index[0], i + step, step).tolist())
                tmp2.loc[i, 'green_time'] = i - tmp.index[0] - step
            else:
                ls_tmp.append(i)
            row += 1
        elif row == len(tmp2) - 1:
            if j['green_start'] > 0:
                index_greens.extend(np.arange(i, tmp.index[-1] + step, step).tolist())
                tmp2.loc[i, 'green_time'] = tmp.index[-1] - i
            else:
                ls_tmp.append(i)
                index_greens.extend(np.arange(ls_tmp[0], ls_tmp[1] + step, step).tolist())
                tmp2.loc[i, 'green_time'] = ls_tmp[1] - ls_tmp[0]
                ls_tmp = []
        else:
            if j['green_end'] > 0:
                ls_tmp.append(i)
                index_greens.extend(np.arange(ls_tmp[0], ls_tmp[1] + step, step).tolist())
                tmp2.loc[i, 'green_time'] = ls_tmp[1] - ls_tmp[0]
                ls_tmp = []
            else:
                ls_tmp.append(i)
            row += 1
    tmp.loc[index_greens, 'green'] = 1
    return tmp2, tmp


def create_cumulative(tuple_crossings, edge_selection, turn='other', time_step=1000, plot=False, statistics=False):
    assert turn in ['incoming', 'outgoing', 'turn_right', 'turn_left', 'straight', 'other']
    df_det = tuple_crossings[1]
    data_title = tuple_crossings[2]
    df_det = df_det.sort_index()
    df_sel = df_det.loc[edge_selection, :]
    df = df_sel.dropna(axis=1)
    df = df.transpose()
    max_time = int(max(df.max()) + time_step)
    df_st = {'time_step': [], 'count': []}
    df_tt = df.astype('float64')
    df_tt = df_tt.assign(travel_time=df_tt[edge_selection[-1]] - df_tt[edge_selection[0]])
    for t in range(time_step, max_time, time_step):
        tmp = df[(df >= (t - time_step)) & (df < t)]
        df_st['time_step'].append(t), df_st['count'].append(tmp.count().values)
    df_st = pd.DataFrame(df_st['count'], index=df_st['time_step'],
                         columns=[f'count_{i[0]}_{i[1]}' for i in edge_selection])
    df_st = df_st.assign(veh_diff=df_st[f'count_{edge_selection[0][0]}_{edge_selection[0][1]}'] -
                                  df_st[f'count_{edge_selection[-1][0]}_{edge_selection[-1][1]}'])
    for i in edge_selection:
        df_st.loc[:, f'cumulative_{i[0]}_{i[1]}'] = df_st[f'count_{i[0]}_{i[1]}'].cumsum()
    df_tt = df_tt.assign(travel_time_sec=df_tt.travel_time / 1000)
    if statistics:
        print(f'Basic statistics of travel time from {edge_selection[0]} to {edge_selection[-1]}: '
              f'{df_tt.travel_time_sec.describe()}')
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        ind_link = 0
        for i in edge_selection:
            ax.plot(df_st.index / 1000, df_st[f'count_{i[0]}_{i[1]}'],
                    color=qual_colorlist[ind_link], label=f'{i[0]}_{i[1]}')
            ind_link += 1
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Vehicles passing [veh]')
        plt.close()
        fig, ax = plt.subplots()
        ind_link = 0
        for i in edge_selection:
            ax.plot(df_st.index / 1000, df_st[f'count_{i[0]}_{i[1]}'].cumsum(), color=qual_colorlist[ind_link],
                    label=f'{i[0]}_{i[1]}')
            ind_link += 1
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Cumulative count [veh]')
        ax.set_title(f'{data_title}_{turn}')
        fig.savefig(f'{data_title}_{edge_selection[0][0]}_{edge_selection[-1][0]}_{turn}')
        fig, ax = plt.subplots()
        ax.plot(df_st.index / 1000, df_st['veh_diff'].cumsum(), label='vehicle accumulation',
                color=qual_colorlist[0])
        ax.plot(df_st.index / 1000, df_st[f'count_{edge_selection[-1][0]}_{edge_selection[-1][1]}'],
                label='vehicles passing downstream',
                color=qual_colorlist[5])
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Vehicles [veh]')
        ax.set_title(f'{data_title} {turn} accumulation')
        fig.savefig(f'{data_title}_{edge_selection[0][0]}_{edge_selection[-1][0]}_accumulation')
    return df_st, df_tt


def test_cycle_times(group_id, crossing_edges, **kwargs):
    _, df_crossings, _ = edges_crossings(group_id, crossing_edges, **kwargs)
    df_cycle = signal_timings(df_crossings)
    return df_cycle


def test_cumulative(group_id, crossing_edges, edge_selection, **kwargs):
    tuple_crossings = edges_crossings(group_id, crossing_edges, **kwargs)
    df_st, df_tt = create_cumulative(tuple_crossings, edge_selection)
    return df_st, df_tt


def create_output_table(df_crossing_sel, group_id, save_csv=False, filename=None):
    hdf_path = rd.get_hdf_path()
    group_path = rd.get_group(group_id)
    df_dict = {'track_id': [], 'from': [], 'to': [], 't_from': [], 't_to': [], 'delta_t': []}
    for i in df_crossing_sel:
        df = df_crossing_sel[i][df_crossing_sel[i].notna()]
        if len(df) % 2 == 0:
            nr = int(len(df) / 2)
            df = df.sort_values()
            df_idx = df.index.get_level_values(0).to_list()
            df_val = df.values
            df_dict['track_id'].extend([i] * nr)
            df_dict['from'].extend(df_idx[::2])
            df_dict['t_from'].extend(df_val[::2])
            df_dict['to'].extend(df_idx[1::2])
            df_dict['t_to'].extend(df_val[1::2])
            df_dict['delta_t'].extend(df_val[1::2] - df_val[::2])
        else:
            continue
    df = pd.DataFrame(df_dict)
    tr_id = rd.get_from_hdf(hdf_path, key_id=group_path + '/all_id', result='ids')
    df = df.merge(tr_id[['track_id', 'type']], how='left', on='track_id')
    if save_csv:
        fn = filename
        if filename is None:
            fn = 'traj_data.csv'
        df.to_csv(path_data + fn)
    return df


def create_xt(edge, group_id, network_df, crossing_edges, show_det=None, plot=False, colormap='gist_rainbow',
              lines=False, veh_type=None, psize=1, bearing_difference=90,
              strict_match=True, folder=path_results, **kwargs):
    edge_length = network_df.loc[network_df['_id'] == edge, 'length'].values[0]
    vt_str = "all"
    _, df_det, data_title = edges_crossings(group_id, crossing_edges, bearing_difference=bearing_difference,
                                            strict_match=strict_match)
    df_sel = df_det.loc[edge]
    df_sel = df_sel.dropna(axis=1, how='any')
    lt = td.get_lt_from_id(df_sel.columns.to_list(), group_id=group_id, gdf=False, **kwargs)
    df_xt = pd.DataFrame()
    s = {'pos_start': [], 'pos_end': [], 'track_id': []}
    e1 = (network_df.loc[network_df['_id'] == edge, ['x1', 'y1']].values[0])
    e2 = (network_df.loc[network_df['_id'] == edge, ['x2', 'y2']].values[0])
    df_transpose = df_sel.transpose()
    c1 = [(xy) for xy in zip(df_transpose.loc[:, 'cross_x1'], df_transpose.loc[:, 'cross_y1'])]
    c2 = [(xy) for xy in zip(df_transpose.loc[:, 'cross_x2'], df_transpose.loc[:, 'cross_y2'])]
    for ind, el in enumerate(lt.index.get_level_values(0).unique()):
        tmp = lt.loc[(el, slice(df_sel.loc['rid1', el], int(df_sel.loc['rid2', el] + 1))), :].copy()
        tmp2 = tmp.apply(help_proj, e1=e1, e2=e2, axis=1)
        _, t1 = distxy.project(e1, e2, c1[ind])
        _, t2 = distxy.project(e1, e2, c2[ind])
        s['pos_start'].append(t1), s['pos_end'].append(t2), s['track_id'].append(el)
        # tmp['proj'] = tmp2
        tmp_dist, _, tmp_proj = zip(*tmp2)
        tmp['lateral_dist'] = tmp_dist
        tmp['proj'] = tmp_proj
        tmp['avg_speed'] = tmp['line_length_yx'] * 3.6
        df_xt = pd.concat([df_xt, tmp], axis=0)
    df_xt.loc[:, 'proj_m'] = df_xt.loc[:, 'proj'] * edge_length
    s2 = pd.DataFrame(s, index=s['track_id'])
    if veh_type is not None:
        assert isinstance(veh_type, (str, list))
        if isinstance(veh_type, str):
            veh_type = [veh_type]
        vt = get_vehicle_types(group_id)
        if set(veh_type).issubset(set(vt.type.values.unique())):
            vt_sel = vt.loc[vt.type.isin(veh_type)]
            df_xt = df_xt.loc[df_xt.index.get_level_values(0).isin(vt_sel.track_id.values)]
        else:
            raise Exception(f"Vehicle type not recognized, should be subset of: {set(vt.type.values.unique())}")
        vt_str = ""
        tmp_str = [w for i in veh_type for w in i if w.isupper()]
        vt_str = vt_str.join(tmp_str)
    if plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        if lines:
            norm = plt.Normalize(0, 50)
            # c_map = cm.ScalarMappable(cmap=colormap, norm=norm)
            for i in df_xt.index.get_level_values(0).unique():
                points = np.array([df_xt.loc[i, 'time'], df_xt.loc[i, 'proj_m']]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, linewidths=psize, cmap=colormap, norm=norm)
                lc.set_array(df_xt.loc[i, 'avg_speed'])
                im = ax.add_collection(lc)
                ax.autoscale()
        else:
            im = ax.scatter(df_xt.time, df_xt.proj_m, s=psize, c=df_xt.speed_1, vmin=0, vmax=50, cmap=colormap)
        if show_det is not None:
            ft = show_det.features.loc[show_det.features.index.get_level_values(0) == edge]
            if len(ft) > 0:
                c_dict = {'crossing': 'dimgrey', 'traffic_signals': 'darkorange'}
                for r, v in ft.iterrows():
                    ax.hlines(v.proj_feature * edge_length, xmin=0, xmax=df_xt.time.max(), linestyles='dashed',
                              colors=c_dict[v.feature], label=v.feature)
            det_loc = show_det.det_loc.loc[show_det.det_loc._id == edge]
            for d in range(1, show_det.n_det + 1):
                ax.hlines(det_loc[f'proj_det{d}'] * edge_length, xmin=0, xmax=df_xt.time.max(), linestyles='dashed',
                          colors='k', label=f'detector {d}')
        ax.grid(True)
        ax.set_title(f'X-T for link {edge}')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Distance (m)')
        if show_det is not None:
            ax.legend()
        fig.suptitle(f"{rd.get_path_dict()['groups'][group_id]}")
        fig.colorbar(im, ax=ax)
        fig.savefig(os.path.join(folder, "plots", f"xt_{rd.get_path_dict()['groups'][group_id].replace('/', '_')}_"
                                                  f"{edge}_{vt_str}_bd{bearing_difference}"))
    return df_xt, s2


def create_xt_arterial(specified_detectors, group_id, network_df, crossing_edges, show_det=None, plot=False,
                       colormap='gist_rainbow', lines=False, veh_type=None, psize=1, bearing_difference=90,
                       strict_match=True, folder=path_results, **kwargs):
    edge_lengths = network_df.loc[network_df['_id'].isin(specified_detectors[0]), 'length'].sum()
    length_factor = edge_lengths / len(specified_detectors[0])
    vt_str = "all"
    _, df_det, data_title = edges_crossings(group_id, crossing_edges=crossing_edges, bearing_difference=bearing_difference,
                                            strict_match=strict_match)
    df_sel = df_det.loc[specified_detectors[0]]
    if specified_detectors[1][0] > 1:
        id1 = df_sel.loc[specified_detectors[0][0]].dropna(axis=1, how='all').columns.to_list()
        id2 = df_sel.loc[specified_detectors[0][1:]].dropna(axis=1, how='any').columns.to_list()
        id3 = list(set(id1).intersection(id2))
        id3.sort()
        df_sel = df_sel.loc[:, id3]
    else:
        df_sel = df_sel.dropna(axis=1, how='any')
    lt = td.get_lt_from_id(df_sel.columns.to_list(), group_id=group_id, gdf=False, **kwargs)
    df_xt = pd.DataFrame()
    s = {'pos_start': [], 'pos_end': [], 't_start': [], 't_end': [], 'track_id': []}
    df_transpose = df_sel.transpose()
    c1 = [(xy) for xy in zip(df_transpose.loc[:, (specified_detectors[0][0], f'cross_x{specified_detectors[1][0]}')],
                             df_transpose.loc[:, (specified_detectors[0][0], f'cross_y{specified_detectors[1][0]}')])]
    c2 = [(xy) for xy in zip(df_transpose.loc[:, (specified_detectors[0][-1], f'cross_x{specified_detectors[1][-1]}')],
                             df_transpose.loc[:, (specified_detectors[0][-1], f'cross_y{specified_detectors[1][-1]}')])]
    ct1 = df_transpose.loc[:, (specified_detectors[0][0], f't{specified_detectors[1][0]}')].values
    ct2 = df_transpose.loc[:, (specified_detectors[0][-1], f't{specified_detectors[1][-1]}')].values
    ed1 = network_df.loc[network_df['_id'].isin(specified_detectors[0]), ['x1', 'y1']]
    ed2 = network_df.loc[network_df['_id'].isin(specified_detectors[0]), ['x2', 'y2']]
    e1 = []
    e2 = []
    for i, j in enumerate(specified_detectors[0]):
        e1.append(ed1.loc[j].values)
        e2.append(ed2.loc[j].values)
    error_traj = []
    for ind, el in enumerate(lt.index.get_level_values(0).unique()):
        tmp = lt.loc[(el, slice(df_sel.loc[(specified_detectors[0][0], f'rid{specified_detectors[1][0]}'), el],
                                int(df_sel.loc[(specified_detectors[0][-1], f'rid{specified_detectors[1][-1]}'), el]
                                    + 1))), :].copy()
        tmp['proj'] = 0
        tmp['avg_speed'] = tmp['line_length_yx'] * 3.6
        _, t1 = distxy.project(e1[0], e2[0], c1[ind])
        _, t2 = distxy.project(e1[-1], e2[-1], c2[ind])
        s['pos_start'].append(t1), s['pos_end'].append(t2 + len(specified_detectors[0]) - 1), s['track_id'].append(el)
        s['t_start'].append(ct1[ind]), s['t_end'].append(ct2[ind])
        while True:
            try:
                for i, j in enumerate(specified_detectors[0]):
                    tmp2 = tmp.apply(help_proj, e1=e1[i], e2=e2[i], axis=1)
                    _, _, tmp_proj = zip(*tmp2)
                    tmp['proj'] += tmp_proj
                break
            except TypeError:
                print(el)
                error_traj.append(el)
                break
        if len(tmp) < 1:
            print(el)
            continue
        df_xt = pd.concat([df_xt, tmp], axis=0)
    df_xt.loc[:, 'proj_m'] = df_xt.loc[:, 'proj'] * length_factor
    s = pd.DataFrame(s, index=s['track_id'])
    s.loc[:, 'tt_ms'] = s.t_end - s.t_start
    s.loc[:, 'dist_m'] = (s.pos_end - s.pos_start) * length_factor
    if veh_type is not None:
        assert isinstance(veh_type, (str, list))
        if isinstance(veh_type, str):
            veh_type = [veh_type]
        vt = get_vehicle_types(group_id)
        if set(veh_type).issubset(set(vt.type.values.unique())):
            vt_sel = vt.loc[vt.type.isin(veh_type)]
            df_xt = df_xt.loc[df_xt.index.get_level_values(0).isin(vt_sel.track_id.values)]
        else:
            raise Exception(f"Vehicle type not recognized, should be subset of: {set(vt.type.values.unique())}")
        vt_str = ""
        tmp_str = [w for i in veh_type for w in i if w.isupper()]
        vt_str = vt_str.join(tmp_str)
    if plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        if lines:
            norm = plt.Normalize(0, 50)
            # c_map = cm.ScalarMappable(cmap=colormap, norm=norm)
            for i in df_xt.index.get_level_values(0).unique():
                points = np.array([df_xt.loc[i, 'time'], df_xt.loc[i, 'proj_m']]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, linewidths=psize, cmap=colormap, norm=norm)
                lc.set_array(df_xt.loc[i, 'avg_speed'])
                im = ax.add_collection(lc)
                ax.autoscale()
        else:
            im = ax.scatter(df_xt.time, df_xt.proj_m, s=psize, c=df_xt.speed_1, vmin=0, vmax=50, cmap=colormap)
        if show_det is not None:
            ft = show_det.features.loc[show_det.features.index.get_level_values(0).isin(specified_detectors[0])]
            if len(ft) > 0:
                c_dict = {'crossing': 'dimgrey', 'traffic_signals': 'darkorange'}
                for r, v in enumerate(ft.iterrows()):
                    v_add = specified_detectors[0].index(v[0][0])
                    ax.hlines((v[1].proj_feature + len(specified_detectors[0][:v_add])) * length_factor, xmin=0,
                              xmax=df_xt.time.max(),
                              linestyles='dashed', colors=c_dict[v[1].feature], label=v[1].feature)
            det_loc = show_det.det_loc.loc[show_det.det_loc._id.isin([specified_detectors[0][0],
                                                                      specified_detectors[0][-1]])]
            for r, d in enumerate(specified_detectors[1]):
                if r == 0:
                    ax.hlines(det_loc[det_loc._id == specified_detectors[0][0]].loc[:, f'proj_det{d}'].values[0]
                              * length_factor,
                              xmin=0, xmax=df_xt.time.max(), linestyles='dashed', colors='darkgreen',
                              label=f'start')
                else:
                    ax.hlines((det_loc[det_loc._id == specified_detectors[0][-1]].loc[:, f'proj_det{d}'].values[0]
                               + len(specified_detectors[0][:-1])) * length_factor, xmin=0, xmax=df_xt.time.max(),
                              linestyles='dashed',
                              colors='darkred', label=f'end')
        ax.grid(True)
        ax.set_title(f'X-T for link {specified_detectors[0]}')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Distance (m)')
        if show_det is not None:
            ax.legend()
        fig.suptitle(f"{rd.get_path_dict()['groups'][group_id]}")
        fig.colorbar(im, ax=ax)
        fig.savefig(os.path.join(folder, "plots", f"xt_{rd.get_path_dict()['groups'][group_id].replace('/', '_')}_"
                                                  f"{specified_detectors[0][0]}_{specified_detectors[0][-1]}_"
                                                  f"arterial_{vt_str}_bd{bearing_difference}"))
    return df_xt, error_traj, s


def get_tt_from_xt(df_xt, bins=20):
    df = pd.DataFrame({'track_id': df_xt.index.get_level_values(0).unique(), 'tt': 0})
    tt = [df_xt.loc[i, 'time'].iloc[-1] - df_xt.loc[i, 'time'].iloc[0] for i in df.track_id]
    df.loc[:, 'tt'] = tt
    df.loc[:, 'tts'] = df.tt / 1000
    plt.hist(df.tts, bins=bins, edgecolor='k')
    plt.title('Travel time')
    plt.xlabel('Seconds')
    return df


def help_proj(row, e1, e2, delta=0.0):
    p = (row['x_1'], row['y_1'])
    dist, p_int, t = distxy.distance_point_to_segment(s1=e1, s2=e2, p=p, delta=delta)
    return dist, p_int, t

