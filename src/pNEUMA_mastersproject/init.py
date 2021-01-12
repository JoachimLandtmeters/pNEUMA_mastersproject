from create_network import *
from map_matching_trajectories import *
from make_detector import *
from calc_parameters import *
import visualization as vz

tic = time.time()

"""
frequency = 10000
n_det = 3
double = True
dfi = 15
det_width = 15
loop_distance = 5
"""

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

toc = time.time()
print(f'{int(divmod(toc - tic, 60)[0])} min {int(divmod(toc - tic, 60)[1])} sec')
