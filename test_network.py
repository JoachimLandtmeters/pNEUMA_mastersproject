"""
Test file for creating and loading all network inputs needed
Author: Joachim Landtmeters
"""
from pneumapackage.settings import *
import pneumapackage.network as cn
import pneumapackage.virtualdetector as md
from pneumapackage.__init__ import read_pickle, write_pickle, path_data

import time


"""
Extract network graph from OSM
- choose filter, predefined by osmnx pacakge or customized
- keep all nodes with relevant attribute values before simplification
- make geodataframes of graph

customized filter makes the inclusion of pedestrian streets possible, extracting all paths would give an abundance of
irrelevant features which is not preferred. Nevertheless, the implication of missing edges is strong because of the 
map matching procedure used
"""


def test_network(bbox=bb_athens, path=path_data, osm_filter=adj_filter, save_shp=False, save_pickle=True, reload=False):
    tic = time.time()
    print('Start: …load network ')
    if reload:
        network_df = cn.CreateNetwork(bounding_box=bbox, custom_filter=osm_filter)
        _ = network_df.network_dfs()
        if save_pickle:
            write_pickle(network_df, 'network', path)
    else:
        try:
            network_df = read_pickle('network', path)
        except FileNotFoundError:
            network_df = cn.CreateNetwork(bounding_box=bbox, custom_filter=osm_filter)
            _ = network_df.network_dfs()
            if save_pickle:
                write_pickle(network_df, 'network', path)

    if save_shp:
        network_df.save_graph_to_shp()
        print('Shapefiles stored')
    toc = time.time()
    print(f'Network loaded, took {toc - tic} sec')
    return network_df


def test_detectors(network_df, path, number_detectors=n_det, detector_length=len_det, distance_from_intersection=dfi,
                   double_loops=None, lonlat=False, save_shp=False):
    double_l = False
    dl_tmp = 0
    if double_loops is not None:
        if type(double_loops) in (int, float):
            if double_loops > 0:
                double_l = double_loops
            else:
                print(f'unrealistic length: {double_loops}, default of 1 m is used')
                double_l = 1
            dl_tmp = double_l
        else:
            raise TypeError('"double_loops" parameter not of the right type, should be integer or float')
    name = f'{number_detectors}_{distance_from_intersection}_{detector_length}_{dl_tmp}'
    try:
        det = read_pickle(f'detectors_{name}', path=path)
    except FileNotFoundError:
        det = md.Detectors(network_df.network_edges, length_detector=detector_length, dfi=distance_from_intersection,
                           n_det=number_detectors, double_loops=double_l, lonlat=lonlat,
                           gdf_special=network_df.node_tags)
        det.detector_projected()
        det.features_projected()
        write_pickle(det, f'detectors_{name}', path=path)
        print('Virtual detectors created for edges')
        if save_shp:
            det.detector_to_shapefile()
            print('Shapefiles stored')
    return det
