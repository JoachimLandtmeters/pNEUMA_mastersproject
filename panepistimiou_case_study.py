"""
Panepistimiou Case Study

- intersection with incoming and outgoing links
- detectors placed generically and at special features of the network, i.e. traffic signals
- Using QGis to select right links and specific detectors --> manual process

"""
from pneumapackage.settings import *
from pneumapackage.__init__ import path_data, path_case_studies

import test_network as tn
import test_analysis as ta

from pathlib import Path
import os
from tqdm.contrib import tenumerate
import time

Path(path_case_studies + "/panepistimiou").mkdir(parents=True, exist_ok=True)
path_pan = os.path.join(path_case_studies, 'panepistimiou')
save_general = True

group_id = input('Enter group ID: ')
crossing_edges = pan_id  # Edges for which the crossings of the detectors are determined

omonoia = ([48, 1309], (1, 2))
panepistimiou = ([1034, 42, 46], (1, 2))
octobriou = ([1129], (1, 2))
vt = ['Car', 'Taxi', 'Bus', 'Medium Vehicle', 'Heavy Vehicle']
bearing_difference = 90

network = tn.test_network()
detectors = tn.test_detectors(network, path_data)
df_crossings = ta.edges_crossings(group_id, crossing_edges, bearing_difference=bearing_difference)
vehicle_types = ta.get_vehicle_types(group_id)

if save_general:
    print('Save general files:')
    fn = 'network.csv'
    network.network_edges.to_csv(os.path.join(path_pan, fn))
    fn = 'detectors.csv'
    detectors.det_edges_all.to_csv(os.path.join(path_pan, fn))
    fn = 'features.csv'
    detectors.features.to_csv(os.path.join(path_pan, fn))

fn = f'{group_id}_vehicle_types.csv'
vehicle_types.to_csv(os.path.join(path_pan, fn))
fn = f'{group_id}_trajectory_crossings_bd{bearing_difference}.csv'
df_crossings[1].to_csv(os.path.join(path_pan, fn))

"""
Cycle times
For Panepistimiou cycle times are determined for all incoming edges
Note: the incoming edge from the south, edge 2544, has not enough traffic to determine the cycles with enough certainty

Edges: 46, 1129, 2544
"""
print('Start cycle times')
tic = time.time()

edges_cycles = [46, 1129, 2544]
df_cycle = ta.signal_timings(df_crossings[1])
thresh_filter = 10000
filter_step = 3
thresh = 5000
for i, j in tenumerate(edges_cycles):
    print(f'Edge: {j}')
    while True:
        try:
            cycle1, cycle2 = ta.cycle_times(df_cycle, j, thresh_filter=thresh_filter, filter_step=filter_step,
                                            thresh=thresh)
            break
        except ValueError:
            thresh += 1000
            print(f'Thresh increased to {thresh} ms')
    fn1 = f'{group_id}_cycle_times_steps_changes_{j}.csv'
    fn2 = f'{group_id}_cycle_times_steps_all_{j}.csv'
    cycle1.to_csv(os.path.join(path_pan, fn1))
    cycle2.to_csv(os.path.join(path_pan, fn2))

toc = time.time()
print(f'End cycle times, took {toc-tic} sec')
"""
Cumulative tables
For Panepistimiou cumulative tables are determined for incoming and outgoing edges

Incoming edges: 1034, 42, 46, 1129, 2204, 2544
Outgoing edges: 49, 48, 1309, 47, 1308
"""
print('Start cumulative tables')
tic = time.time()

in_1 = [(1129, 't1'), (1129, 't2')]
in_2 = [(1034, 't1'), (42, 't2'), (46, 't2')]
in_3 = [(2204, 't1'), (2204, 't2'), (2544, 't2')]

out_1 = [(49, 't1'), (49, 't2')]
out_2 = [(48, 't1'), (48, 't2'), (1309, 't2')]
out_3 = [(47, 't2'), (1308, 't1'), (1308, 't2')]

cum_edges = [in_1, in_2, in_3, out_1, out_2, out_3]

for i, j in tenumerate(cum_edges):
    df_cum, _ = ta.create_cumulative(df_crossings, j)
    fn = f'{group_id}_cumulative_{j[0][0]}_{j[-1][0]}.csv'
    df_cum.to_csv(os.path.join(path_pan, fn))

toc = time.time()
print(f'End cumulative tables, took {toc-tic} sec')
"""
XT-tables
For Panepistimiou xt-tables are determined for incoming and outgoing edges

Incoming edges: 1034, 42, 46, 1129, 2204, 2544
Outgoing edges: 49, 48, 1309, 47, 1308
"""
print('Start xt-tables')
tic = time.time()

det_numbers = (1, 2)
xt_1 = ([1129], det_numbers)
xt_2 = ([1034, 42, 46], det_numbers)
xt_3 = ([2204, 2544], det_numbers)
xt_4 = ([49], det_numbers)
xt_5 = ([48, 1309], det_numbers)
xt_6 = ([47, 1308], det_numbers)

xt_edges = [xt_1, xt_2, xt_3, xt_4, xt_5, xt_6]

for i, j in tenumerate(xt_edges):
    df_xt = ta.create_xt_arterial(j, group_id, network_df=network.network_edges, crossing_edges=crossing_edges)
    fn1 = f'{group_id}_xt_{j[0][0]}_{j[0][-1]}.csv'
    fn2 = f'{group_id}_travel_time_{j[0][0]}_{j[0][-1]}.csv'
    df_xt[0].to_csv(os.path.join(path_pan, fn1))
    df_xt[2].to_csv(os.path.join(path_pan, fn2))

toc = time.time()
print(f'End xt-tables, took {toc-tic} sec')
