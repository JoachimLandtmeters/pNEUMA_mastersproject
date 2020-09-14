"""
Plotting for different traffic characteristics and sensitivity analysis
- scatter
- time series
- curve fit

Needed files:
- parameters file
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections
import time
from textwrap import wrap


# Scatter plot


def plot_fd(counts_dict, colors=None, veh_area_colorbar=False, summary=False, individual_fd=True,
            save_figures=True, max_lim_flow=0, max_lim_speed=0, filename=0):
    figures = []
    n_det = counts_dict['counts']['info']['number_of_det'][0]
    distance = counts_dict['counts']['info']['distance_det'][0]
    edge_id = counts_dict['counts']['detectors']['index'].values
    loop_distance = counts_dict['counts']['detectors']['loop_distance'].values
    if not colors:
        colors = ['b', 'k']
    markersize = 5
    f_size = (12, 7)
    max_q = max(
        [max(j[f'flow_lane_{det}']) for det in range(1, n_det + 1)
         for i, j in enumerate(counts_dict['parameters'])]) + 100
    max_v = max([max(j[f'speed_{det}']) for det in range(1, n_det + 1)
                 for i, j in enumerate(counts_dict['parameters'])]) + 5
    max_lim_flow = max(max_q, max_lim_flow)
    max_lim_speed = max(max_v, max_lim_speed)
    for i, j in enumerate(counts_dict['parameters']):
        if filename != 0:
            name_file = f'{filename} {i + 1}'
        else:
            name_file = f'edge {edge_id[i]}'
        if summary:
            fig, axes = plt.subplots(nrows=2, ncols=n_det, sharex='col', sharey='row', figsize=f_size)
            axes_list = [item for sublist in axes for item in sublist]
            m_q = max([max(j[f'flow_lane_{det}']) for det in range(1, n_det + 1)])
            m_v = max([max(j[f'speed_{det}']) for det in range(1, n_det + 1)])
            m_k = max([max(j[f'density_lane_{det}']) for det in range(1, n_det + 1)])
            diff_det = n_det
            for det in range(1, n_det + 1):
                ax_1 = axes_list.pop(0)
                ax_2 = axes_list.pop(diff_det - 1)
                ax_1.scatter(j[f'density_lane_{det}'], j[f'flow_lane_{det}'],
                             color=colors[1], s=markersize)
                ax_2.scatter(j[f'density_lane_{det}'], j[f'speed_{det}'],
                             color=colors[1], s=markersize)
                if det == 1:
                    ax_1.set_ylabel('Flow q [veh/h per lane]')
                    ax_2.set_ylabel('Speed v [km/h]')
                ax_1.set_ylim(bottom=-50, top=m_q + 200)
                ax_2.set_ylim(bottom=-2.5, top=m_v + 10)
                ax_2.set_xlim(left=-10, right=m_k + 50)
                ax_1.set_title(f'Detector {det}')
                ax_2.set_xlabel('Density k [veh/km per lane]')
                ax_1.grid(True)
                ax_2.grid(True)
                diff_det -= 1
            fig.suptitle(f"FDs for {name_file} \n" + f"(loop distance: {loop_distance[i]} m, time step: "
                                                     f"{int(counts_dict['counts']['info']['frequency'][0] / 1000)} "
                                                     f"sec, dfi: {distance} m)", fontsize=16, wrap=True)
            fig.align_ylabels()
            plt.tight_layout(rect=[0, 0.03, 1, 0.90])
            if save_figures:
                plt.savefig(f"FD_summary_{name_file}_{int(counts_dict['counts']['info']['frequency'][0] / 1000)}_lp"
                            f"{loop_distance[i]}_d{distance}.png")
        if not individual_fd:
            continue
        for det in range(1, n_det + 1):
            if veh_area_colorbar:
                plt.figure()
                kqFD = plt.scatter(j[f'density_lane_{det}'], j[f'flow_lane_{det}']
                                   , c=j[f'vehicle_area_{det}'])
                plt.clim(0, 15)
                cb1 = plt.colorbar()
                cb1.set_label('Average vehicle area')
                # plt.legend(loc='upper left')
                plt.grid(True)
                plt.title(f"FD flow-density {name_file} \n" +
                          f"(Detector {det}: loop distance {loop_distance[i]} m, time step: "
                          f"{int(counts_dict['counts']['info']['frequency'][0] / 1000)} "
                          f"sec, dfi: {distance} m)", wrap=True)
                plt.xlabel('Density k [veh/km per lane]')
                plt.ylabel('Flow q [veh/h per lane]')
                plt.axis([- 20, int(max(j[f'density_lane_{det}']) + 50),
                          - 50, max_lim_flow])
                plt.tight_layout()
                if save_figures:
                    plt.savefig(f"FD_kq_veh_area_det{det}_{name_file}_"
                                f"{int(counts_dict['counts']['info']['frequency'][0] / 1000)}_lp"
                                f"{loop_distance[i]}_d{distance}.png")
                plt.figure()
                kuFD = plt.scatter(j[f'density_lane_{det}'], j[f'speed_{det}'],
                                   c=j[f'vehicle_area_{det}'])
                plt.clim(0, 15)
                cb2 = plt.colorbar()
                cb2.set_label('Average vehicle area')
                # plt.legend(loc='upper left')
                plt.grid(True)
                plt.title(f"FD speed-density {name_file} \n" +
                          f"(Detector {det}: loop distance {loop_distance[i]} m, time step: "
                          f"{int(counts_dict['counts']['info']['frequency'][0] / 1000)} "
                          f"sec, dfi: {distance} m)", wrap=True)
                plt.xlabel('Density k [veh/km per lane]')
                plt.ylabel('Speed v [km/h]')
                plt.axis([- 20, int(max(j[f'density_lane_{det}']) + 50),
                          - 5, max_lim_speed])
                plt.tight_layout()
                if save_figures:
                    plt.savefig(f"FD_kv_veh_area_det{det}_{name_file}_"
                                f"{int(counts_dict['counts']['info']['frequency'][0] / 1000)}_lp"
                                f"{loop_distance[i]}_d{distance}.png")
            else:
                plt.figure()
                kqFD = plt.scatter(j[f'density_lane_{det}'], j[f'flow_lane_{det}'],
                                   color=colors[0])
                # plt.legend(loc='upper left')
                plt.grid(True)
                plt.title(f"FD flow-density edge {name_file} \n" +
                          f"(Detector {det}: loop distance {loop_distance[i]} m), time step: "
                          f"{int(counts_dict['counts']['info']['frequency'][0] / 1000)} "
                          f"sec, dfi: {distance} m)", wrap=True)
                plt.xlabel('Density k [veh/km per lane]')
                plt.ylabel('Flow q [veh/h per lane]')
                plt.axis([- 20, int(max(j[f'density_lane_{det}']) + 50),
                          - 50, max_lim_flow])
                plt.tight_layout()
                if save_figures:
                    plt.savefig(f"FD_kq_det{det}_{name_file}_"
                                f"{int(counts_dict['counts']['info']['frequency'][0] / 1000)}_lp"
                                f"{loop_distance[i]}_d{distance}.png")
                plt.figure()
                kuFD = plt.scatter(j['density_lane_' + str(det)],
                                   j['speed_' + str(det)], color=colors[0])
                # plt.legend(loc='upper left')
                plt.grid(True)
                plt.title(f"FD speed-density edge {name_file} \n" +
                          f"(Detector {det}: loop distance {loop_distance[i]} m, time step: "
                          f"{int(counts_dict['counts']['info']['frequency'][0] / 1000)} "
                          f"sec, dfi: {distance} m))", wrap=True)
                plt.xlabel('Density k [veh/km per lane]')
                plt.ylabel('Speed v [km/h]')
                plt.axis([- 20, int(max(j[f'density_lane_{det}']) + 50),
                          - 5, max_lim_speed])
                plt.tight_layout()
                if save_figures:
                    plt.savefig(f"FD_kv_det{det}_{name_file}_"
                                f"{int(counts_dict['counts']['info']['frequency'][0] / 1000)}_lp"
                                f"{loop_distance[i]}_d{distance}.png")
            figures.append([kqFD, kuFD])
    return figures


def mixed_traffic(parameters, modal_share_limit, veh_type, detector_number, label, colors, name_arterial):
    df_info = pd.DataFrame({'vehicle type': veh_type, 'limit': modal_share_limit, 'detector': detector_number},
                           index=[0])
    param_low = parameters[parameters[f'{veh_type}_{detector_number}'] <= modal_share_limit]
    param_high = parameters[parameters[f'{veh_type}_{detector_number}'] > modal_share_limit]
    plt.figure()
    plt.scatter(param_low[f'density_lane_{detector_number}'], param_low[f'flow_lane_{detector_number}'],
                color=colors[0], label=f'{label} <= {modal_share_limit} %',
                s=param_low[f'{veh_type}_{detector_number}'] / max(param_high[f'{veh_type}_{detector_number}']) * 30)
    plt.scatter(param_high[f'density_lane_{detector_number}'], param_high[f'flow_lane_{detector_number}'],
                color=colors[1], label=f'{label} > {modal_share_limit} %',
                s=param_high[f'{veh_type}_{detector_number}'] / max(param_high[f'{veh_type}_{detector_number}']) * 30)
    plt.title(f"FD flow-density with modal shares ({name_arterial})")
    plt.xlabel('Density k [veh/km per lane]')
    plt.ylabel('Flow q [veh/h per lane]')
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    return [param_low, param_high, df_info]


def plotfd_mode_comparison(parameter_df, parameter_df_mode, detector=1, labels=None, per_lane=True):
    colors = ['b', 'red']
    marker = 30
    font = 16
    f_size = (12, 7)
    if per_lane:
        str_lane = '_lane_'
        str_label = 'per lane'
    else:
        str_lane = '_'
        str_label = ''
    if labels is None:
        labels = ['df 1', 'df 2']
    fig, ax = plt.subplots(figsize=f_size)
    ax.scatter(parameter_df[f'density{str_lane}{detector}'], parameter_df[f'flow{str_lane}{detector}'],
               s=marker, color=colors[0], label=labels[0], alpha=0.6)
    ax.scatter(parameter_df_mode[f'density{str_lane}{detector}'], parameter_df_mode[f'flow{str_lane}{detector}'],
               s=marker, color=colors[1], label=labels[1], marker='^')
    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlabel(f'Density [veh/km {str_label}]')
    ax.set_ylabel(f'Flow [veh/h {str_label}]')
    fig.suptitle('flow-density', fontsize=font)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig, ax = plt.subplots(figsize=f_size)
    ax.scatter(parameter_df[f'density{str_lane}{detector}'], parameter_df[f'speed_{detector}'],
               s=marker, color=colors[0], label=labels[0], alpha=0.6)
    ax.scatter(parameter_df_mode[f'density{str_lane}{detector}'], parameter_df_mode[f'speed_{detector}'],
               s=marker, color=colors[1], label=labels[1], marker='^')
    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlabel(f'Density [veh/km {str_label}]')
    ax.set_ylabel('Speed [km/h]')
    fig.suptitle('speed-density', fontsize=font)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    # Boxplots
    fig, ax = plt.subplots(ncols=2, nrows=3, sharey='row', figsize=f_size)
    columns = [f'density{str_lane}{detector}', f'flow{str_lane}{detector}', f'speed_{detector}']
    box_labels = [f'Density [veh/km {str_label}]', f'Flow [veh/h {str_label}]', f'Speed [km/h]']
    for i in range(0, len(ax)):
        ax[i, 0].boxplot(parameter_df[columns[i]])
        ax[i, 1].boxplot(parameter_df_mode[columns[i]])
        ax[i, 0].set_title(labels[0])
        ax[i, 1].set_title(labels[1])
        ax[i, 0].grid(True)
        ax[i, 1].grid(True)
        ax[i, 0].set_ylabel(box_labels[i])
    fig.suptitle('Boxplots for traffic characteristics', fontsize=font)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])


def plot_time_series(parameter_df, freq, detector=1):
    font = 16
    f_size = (12, 7)
    fig, ax = plt.subplots(figsize=f_size, nrows=3, sharex=True)
    ax[0].plot(parameter_df[f'density_{detector}'], marker='o', color='k')
    ax[0].set_ylabel('density [veh/km]')
    ax[0].grid(True)
    ax[1].plot(parameter_df[f'flow_{detector}'], marker='o', color='k')
    ax[1].set_ylabel('flow [veh/h]')
    ax[1].grid(True)
    ax[2].plot(parameter_df[f'speed_{detector}'], marker='o', color='k')
    ax[2].set_ylabel('speed [km/h]')
    ax[2].grid(True)
    ax[2].set_xlabel(f'time steps [{round(float(freq/1000), 1)} sec]')
    fig.suptitle('Time series for traffic characteristics', fontsize=font)
    fig.align_ylabels()
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])


def plot_column_for_all_detectors(parameter_df, freq, column, n_det, ylab, title):
    font = 16
    f_size = (12, 7)
    fig, ax = plt.subplots(figsize=f_size, nrows=n_det, sharex=True, sharey=True)
    for det in range(1, n_det+1):
        ax[det-1].plot(parameter_df[f'{column}_{det}'], marker='o', color='k')
        ax[det-1].set_title(f'Detector {det}')
        ax[det-1].set_ylabel(f'{ylab}')
        ax[det-1].grid(True)
    ax[n_det-1].set_xlabel(f'time steps [{round(float(freq/1000), 1)} sec]')
    fig.suptitle(title, fontsize=font)
    fig.align_ylabels()
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])


def plot_vehicle_mix_traffic_characteristics(list_parameter_df_mode, column, colors, labels, axis_labels,
                                             h_lines=None, v_lines=None):
    figs = (12, 5)
    col_tot = 0
    fig, ax = plt.subplots(figsize=figs)
    for ind, param in enumerate(list_parameter_df_mode):
        col_tot += param[column]
        ax.plot(param[column], color=colors[ind], label=labels[ind])
    ax.plot(col_tot, color='k', label='Total')
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.legend()
    if h_lines is not None:
        ax.hlines(h_lines, 0, 1, transform=ax.get_xaxis_transform(), linestyles='dashed', color='grey', lw=2)
    if v_lines is not None:
        ax.vlines(v_lines, 0, 1, transform=ax.get_xaxis_transform(), linestyles='dashed', color='grey', lw=2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])


def plot_arterial_statistics(art_df, art_name, characteristic='_k', variable='mean', label='unit',
                             save=True):
    list_col = []
    variables = {}
    for i, j in enumerate(list(art_df)):
        if art_name in j and characteristic in j:
            list_col.append(j)
    if type(variable) is not list:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(art_df[list_col].loc[variable], marker='o', color='k', lw=0.75)
        variables[f'{art_name}_{variable}'] = art_df[list_col].loc[variable]
    else:
        colors = ['k', 'b', 'r', 'g', 'orange', 'pink', 'purple']
        fig, ax = plt.subplots(figsize=(12, 7))
        for e, v in enumerate(variable):
            ax.plot(art_df[list_col].loc[v], marker='o', color=colors[e], lw=0.75, label=v)
            variables[f'{art_name}_{v}'] = art_df[list_col].loc[v]
        ax.legend()
    ax.grid(True)
    ax.set_ylabel(label)
    char_mod = characteristic + '1'
    ticks = [j for i, j in enumerate(list_col) if art_name in j and char_mod in j]
    tick_labels = [f'{i+1}' for i, j in enumerate(ticks)]
    plt.xticks(ticks, tick_labels)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.title(art_name)
    if save:
        plt.savefig(f"Arterial_{art_name}_{characteristic}_.png")
    variables = pd.DataFrame(variables)
    return variables


def fitted_curves():
    print('skip')
# def traffic_scatter(parameters_list):


# curve fit

# def traffic_curve_fit():
"""
fig, ax = plt.subplots(ncols=2, nrows=4, sharey=True, figsize=(12,7))
for i, (k,v) in enumerate(tt_art.items()):
    if i < 2:  
        v[0].hist(column='tt_art_seconds', ax=ax[0,i], grid=True)
        ax[0, i].set_title(k)
        ax[0, i].set_xlabel('Unit travel time [s/m]')
    elif i < 4:
        v[0].hist(column='tt_art_seconds', ax=ax[1, i-2], grid=True)
        ax[1, i-2].set_title(k)
        ax[1, i-2].set_xlabel('Unit travel time [s/m]')
    elif i < 6:
        v[0].hist(column='tt_art_seconds', ax=ax[2, i-4], grid=True)
        ax[2, i-4].set_title(k)
        ax[2, i-4].set_xlabel('Unit travel time [s/m]')
    else:
        v[0].hist(column='tt_art_seconds', ax=ax[3, i-6], grid=True)
        ax[3, i-6].set_title(k)
        ax[3, i-6].set_xlabel('Unit travel time [s/m]')
plt.tight_layout(rect=[0, 0.03, 1, 0.90])
"""