"""
Created at EPFL 2020

@author: Joachim Landtmeters
Placing detectors on edges

Input requirements:
- dataframe with edges, needed column names = 'N1', 'N2', 'length'
- Linestring column in x-y coordinates (needed for interpolate function) with name = 'geom'
Note: Linestring object can follow curves, not just straight lines between begin and end node
"""
# Choice between two implementations:
# - single detectors
# - multiple detectors at once

import pneumapackage.compassbearing as cpb
from pneumapackage.__init__ import path_results
from pneumapackage.settings import *

import geopandas as gpd
import pandas as pd
import osmnx as ox
from pyproj import Proj
from shapely.geometry import Point, LineString
import leuvenmapmatching.util.dist_euclidean as distxy

from pathlib import Path
import os


# Single implementation
# gdf = dataframe with edges with Linestring object in x-y coordinates
# distance = specified distance of point object from start node of linestring object
# relative = place point object on relative position wrt to length of edge
# reverse = specified distance of point object starting from end node of linestring object


class Detectors:

    def __init__(self, gdf_netw, n_det, length_detector, dfi, double_loops, lonlat=False, gdf_special=None):
        self.network = gdf_netw
        self.n_det = n_det
        self.dfi = dfi
        self.len_det = length_detector
        if type(double_loops) in (int, float):
            self.double_loops = True
            self.loop_width = double_loops
        else:
            self.double_loops = False
            self.loop_width = 0
        self.det_loc = make_double_detector(self.network, self.dfi, n_det=self.n_det, loop_width=self.loop_width,
                                            make_double_loops=self.double_loops)
        self.det_loc_latlon = get_xy_to_crs_double_loops(self.det_loc, n_det=self.n_det, double_loops=self.double_loops,
                                                         lonlat=lonlat)
        self.det_edges = make_detector_edges(self.det_loc_latlon, self.len_det, double_loops=self.double_loops)
        self.features = edge_features(gdf_special, length_detector, lonlat=lonlat)
        self.det_edges_all = self.det_edges[0]
        self.det_edges_all_ts = {}
        self.det_selection = {}

    def info(self):
        det_info = {'number_detectors': self.n_det, 'distance_from_intersection': self.dfi,
                    'length_detector': self.len_det, 'double_loops': self.double_loops, 'loop_width': self.loop_width}
        return det_info

    def detector_selection(self, index_list):
        det_sel = self.det_edges_all[self.det_edges_all['_id'].isin(index_list)]
        self.det_selection = det_sel
        return det_sel

    def detector_projected(self):
        det_loc = self.det_loc
        tmp_det = pd.merge(det_loc, self.network[['_id', 'x1', 'y1', 'x2', 'y2']], how='left', on='_id')
        for ind in range(1, self.n_det + 1):
            tmp_det[f'crd{ind}'] = [t.coords[0] for t in tmp_det[f'detector {ind}']]
            det_proj = tmp_det.apply(help_det_proj, column_name=f'crd{ind}', axis=1)
            det_loc[f'proj_det{ind}'] = det_proj
        self.det_loc = det_loc

    def features_projected(self):
        ft = self.features
        ft = ft.reset_index()
        tmp_ft = pd.merge(ft, self.network[['_id', 'x1', 'y1', 'x2', 'y2']], how='left', on='_id')
        ft_proj = tmp_ft.apply(help_det_proj, column_name='xy', axis=1)
        ft['proj_feature'] = ft_proj
        ft.set_index(['_id', 'index'], inplace=True)
        self.features = ft

    def detector_to_shapefile(self, det_sel=False, filename=None, folder=path_results):
        detector = self.det_edges_all
        fn = ''
        if filename is not None:
            fn = filename
        if det_sel:
            detector = self.det_selection
        Path(folder + "/shapefiles").mkdir(parents=True, exist_ok=True)
        for det in range(1, self.n_det + 1):
            det_gdf = gpd.GeoDataFrame(detector[['_id', 'n1', 'n2']], geometry=detector[f'det_edge_{det}'])
            det_gdf_shp = det_gdf.copy()
            det_gdf_shp.crs = 'epsg:4326'
            shp_fn = os.path.join(folder, 'shapefiles', f'detector_{det}{fn}')
            det_gdf_shp.to_file(filename=shp_fn)
            if self.double_loops:
                det_bis_gdf = gpd.GeoDataFrame(detector[['_id', 'n1', 'n2']], geometry=detector[f'det_edge_{det}bis'])
                det_bis_gdf_shp = det_bis_gdf.copy()
                det_bis_gdf_shp.crs = 'epsg:4326'
                shp_fnbis = os.path.join(folder, 'shapefiles', f'detector_{det}bis{fn}')
                det_bis_gdf_shp.to_file(filename=shp_fnbis)


def make_detector(gdf, distance, relative=False, reverse=False):
    if distance < 0:
        raise Exception('distance should be positive. The value was: {}'.format(distance))
    if relative:
        if 1 < distance:
            raise Exception('distance should be lower or equal to 1 to be relative. '
                            'The value was: {}'.format(distance))
    if gdf.crs.to_epsg() == 4326:
        gdf = ox.project_gdf(gdf)
    name = []
    id_b = []
    if not reverse:
        for i, j in gdf.iterrows():
            if relative:
                d = gdf.loc[i, 'geometry'].interpolate(distance, normalized=relative)
            elif gdf['length'][i] > distance:
                d = gdf.loc[i, 'geometry'].interpolate(distance, normalized=relative)
            else:
                d = gdf.loc[i, 'geometry'].interpolate(0.1, normalized=True)
            id = gdf.loc[i, ['n1', 'n2']]
            name.append(d)
            id_b.append(id)
    else:
        for i, j in gdf.iterrows():
            if gdf['length'][i] > distance:
                d = gdf.loc[i, 'geometry'].interpolate(gdf.loc[i, 'length'] - distance, normalized=relative)
            else:
                d = gdf.loc[i, 'geometry'].interpolate(0.9, normalized=True)
            id = gdf.loc[i, ['n1', 'n2']]
            name.append(d)
            id_b.append(id)
    name = pd.DataFrame(name, columns=['detector_1'])
    id_b = pd.DataFrame(id_b, columns=['n1', 'n2'])
    name = pd.concat([name, id_b], axis=1)
    name_2 = gpd.GeoDataFrame(name, geometry=name.loc[:, 'detector_1'])
    return name_2


# Multiple detectors
# Better implementation and possibility to include double loops
# gdf = dataframe with edges with Linestring object in x-y coordinates
# distance = specified distance of point object from start node of linestring object
# n_det = number of detectors to place on the edges ( >1: always detector at begin and end of edge)
# (continued) other detectors are placed in between these two detectors with equal spacing between them
# make_double_loops = construct double loops by placing an extra detector right behind every initial detector


def make_double_detector(gdf, distance, n_det=1, make_double_loops=True, loop_width=1):
    if distance < 0:
        raise Exception('distance should be positive. The value was: {}'.format(distance))
    if gdf.crs.to_epsg() == 4326:
        gdf = ox.project_gdf(gdf)  # project unprojected geodataframe of network edges
    name = []
    id_b = []
    det_bearing = []
    new_loop_distance = []
    if make_double_loops:
        name_double = []
    if n_det > 1:
        for i, j in gdf.iterrows():
            det_loc_edge = [0] * n_det
            dist_help = max(1, 0.05 * gdf['length'][i])  # To get bearing of local part of the edge's length
            dist_help_rel = 0.1
            det_help = [0] * n_det
            # Create double loops
            if make_double_loops:
                det_loc_edge_double = [0] * n_det
                det_help = [0] * n_det * 2
            if gdf['length'][i] > (distance * 2) and gdf['length'][i] > loop_width:
                # assure order of detectors on edge (begin not after end)
                det_loc_edge[0] = gdf.loc[i, 'geometry'].interpolate(distance)  # Begin detector, first in list
                det_help[0] = gdf.loc[i, 'geometry'].interpolate(distance + dist_help)
                if make_double_loops:
                    det_loc_edge_double[0] = gdf.loc[i, 'geometry'].interpolate(distance + loop_width)
                    det_help[n_det] = gdf.loc[i, 'geometry'].interpolate(distance + loop_width + dist_help)
                inter_length = (gdf['length'][i] - distance * 2)
                prev_dist = [distance]
                relative_distance = 1 / (n_det - 1)
                inter_det_length = relative_distance * inter_length
                tag = False
                for ind in range(1, len(det_loc_edge)):
                    new_distance = (inter_det_length * ind) + distance
                    det_loc_edge[ind] = gdf.loc[i, 'geometry'].interpolate(new_distance)
                    det_help[ind] = gdf.loc[i, 'geometry'].interpolate(new_distance + dist_help)
                    if make_double_loops:
                        inter_length = (gdf['length'][i] - (distance + loop_width) * 2)
                        inter_det_length = relative_distance * inter_length
                        if 0.5 * loop_width < inter_det_length:  # only one loop distance needed, 2 is too strict
                            new_distance = (inter_det_length * ind) + distance + loop_width
                            det_loc_edge[ind] = gdf.loc[i, 'geometry'].interpolate(new_distance - 0.5 * loop_width)
                            det_help[ind] = gdf.loc[i, 'geometry'].interpolate(
                                new_distance - 0.5 * loop_width + dist_help)
                            det_loc_edge_double[ind] = gdf.loc[i, 'geometry'].interpolate(new_distance +
                                                                                          0.5 * loop_width)
                            det_help[n_det + ind] = gdf.loc[i, 'geometry'].interpolate(new_distance
                                                                                       + 0.5 * loop_width + dist_help)
                            if len(det_loc_edge) - ind == 1:  # Last detector
                                det_loc_edge[ind] = gdf.loc[i, 'geometry'].interpolate(new_distance)
                                det_help[ind] = gdf.loc[i, 'geometry'].interpolate(new_distance + dist_help)
                                det_loc_edge_double[ind] = gdf.loc[i, 'geometry'].interpolate(
                                    new_distance + loop_width)
                                det_help[n_det + ind] = gdf.loc[i, 'geometry'].interpolate(new_distance
                                                                                           + loop_width + dist_help)
                        else:
                            # Place identical detectors (keep the number of detectors to have a valid dataframe)
                            # rel=(prev_dist[ind-1]+inter_det_length*0.4)/gdf['length'][i]
                            # det_loc_edge_double[ind-1]=gdf.loc[i,'geom'].interpolate(rel,normalized=True)
                            # det_help[n_det+ind - 1] = gdf.loc[i, 'geom'].\
                            # interpolate(rel+dist_help_rel, normalized=True)
                            tag = True
                            rel = 0.5 * gdf['length'][i] - (loop_width * 0.5)  # Place a detector in middle of link
                            det_loc_edge[ind] = gdf.loc[i, 'geometry'].interpolate(rel)
                            det_loc_edge_double[ind] = gdf.loc[i, 'geometry'].interpolate(rel + loop_width)
                            det_help[ind] = gdf.loc[i, 'geometry']. \
                                interpolate(rel + dist_help)
                            det_help[n_det + ind] = gdf.loc[i, 'geometry']. \
                                interpolate(rel + loop_width + dist_help)
                    prev_dist.append(new_distance)
                if make_double_loops:  # Place double loops for end detector on edge
                    if tag:
                        dist_special = 0.5 * gdf['length'][i] - (loop_width * 0.5)
                        det_loc_edge[0] = gdf.loc[i, 'geometry'].interpolate(dist_special)
                        det_help[0] = gdf.loc[i, 'geometry'].interpolate(dist_special + dist_help)
                        det_loc_edge_double[0] = gdf.loc[i, 'geometry'].interpolate(dist_special + loop_width)
                        det_help[n_det] = gdf.loc[i, 'geometry'].interpolate(dist_special + loop_width + dist_help)
                        new_loop_distance.append(loop_width)
                    else:
                        new_loop_distance.append(loop_width)
                    name_double.append(det_loc_edge_double)
                det_bearing.append(det_help)
                name.append(det_loc_edge)
            else:
                d_rel = 0.2  # Define relative distance instead of initial distance
                det_loc_edge[0] = gdf.loc[i, 'geometry'].interpolate(d_rel, normalized=True)
                det_help[0] = gdf.loc[i, 'geometry'].interpolate(d_rel + dist_help_rel, normalized=True)
                rel = (1 - d_rel * 2) / (n_det - 1)
                prev_dist = [d_rel]
                for ind in range(1, len(det_loc_edge)):
                    new_distance = rel * ind + d_rel
                    det_loc_edge[ind] = gdf.loc[i, 'geometry'].interpolate(new_distance, normalized=True)
                    det_help[ind] = gdf.loc[i, 'geometry'].interpolate(new_distance + dist_help_rel, normalized=True)
                    # if make_double_loops:
                    # rel_dist=rel*0.2 + prev_dist[ind-1]
                    # det_loc_edge_double[ind - 1] = gdf.loc[i, 'geom'].interpolate(rel_dist, normalized=True)
                    # det_help[n_det + ind - 1] = gdf.loc[i, 'geom']. \
                    # interpolate(rel_dist + dist_help_rel, normalized=True)
                    prev_dist.append(new_distance)
                if make_double_loops:
                    if gdf['length'][i] > (loop_width + 2 * dist_help):
                        dist_special = 0.5 * gdf['length'][i] - (loop_width * 0.5)
                        for ind in range(0, len(det_loc_edge)):
                            det_loc_edge[ind] = gdf.loc[i, 'geometry'].interpolate(dist_special)
                            det_help[ind] = gdf.loc[i, 'geometry'].interpolate(dist_special + dist_help)
                            det_loc_edge_double[ind] = gdf.loc[i, 'geometry'].interpolate(dist_special + loop_width)
                            det_help[n_det + ind] = gdf.loc[i, 'geometry'].interpolate(dist_special +
                                                                                       loop_width + dist_help)
                        new_loop_distance.append(loop_width)
                    else:
                        dist_special = 0.5
                        rel_loop = 0.1
                        for ind in range(0, len(det_loc_edge)):
                            det_loc_edge[ind] = gdf.loc[i, 'geometry'].interpolate(dist_special - rel_loop,
                                                                                   normalized=True)
                            det_help[ind] = gdf.loc[i, 'geometry'].interpolate(dist_special - rel_loop + dist_help_rel,
                                                                               normalized=True)
                            det_loc_edge_double[ind] = gdf.loc[i, 'geometry'].interpolate(dist_special + rel_loop,
                                                                                          normalized=True)
                            det_help[n_det + ind] = gdf.loc[i, 'geometry'].interpolate(dist_special +
                                                                                       rel_loop + dist_help_rel,
                                                                                       normalized=True)
                        new_loop_distance.append(2 * rel_loop * gdf['length'][i])
                    name_double.append(det_loc_edge_double)
                det_bearing.append(det_help)
                name.append(det_loc_edge)
            id = gdf.loc[i, ['_id', 'n1', 'n2', 'highway', 'lanes', 'length']]
            id_b.append(id)
    else:
        for i, j in gdf.iterrows():
            det_help = [0]
            dist_help = min(1, 0.05 * gdf['length'][i])
            dist_help_rel = 0.1
            if gdf['length'][i] > distance:
                d = gdf.loc[i, 'geometry'].interpolate(distance)
                det_help[0] = gdf.loc[i, 'geometry'].interpolate(distance + dist_help)
                if make_double_loops:
                    if gdf['length'][i] - 2 * distance > loop_width:
                        d_2 = gdf.loc[i, 'geometry'].interpolate(distance + loop_width)
                        det_help_double = gdf.loc[i, 'geometry'].interpolate(distance + loop_width + dist_help)
                        name_double.append(d_2)
                        new_loop_distance.append(loop_width)
                    else:
                        rel = gdf['length'][i] - distance
                        d_2 = gdf.loc[i, 'geometry'].interpolate(rel)
                        det_help_double = gdf.loc[i, 'geometry'].interpolate(rel + dist_help)
                        name_double.append(d_2)
                        new_loop_distance.append(gdf['length'][i] - distance)
                    det_help.append(det_help_double)
                name.append(d)
                det_bearing.append(det_help)
            else:
                d = gdf.loc[i, 'geometry'].interpolate(0.5, normalized=True)
                det_help[0] = gdf.loc[i, 'geometry'].interpolate(0.5 + dist_help_rel, normalized=True)
                if make_double_loops:
                    if gdf['length'][i] * 0.5 > loop_width:
                        rel = loop_width / gdf['length'][i] + 0.5
                        d_2 = gdf.loc[i, 'geometry'].interpolate(rel, normalized=True)
                        det_help_double = gdf.loc[i, 'geometry'].interpolate(rel + dist_help_rel, normalized=True)
                        name_double.append(d_2)
                        new_loop_distance.append(loop_width)
                    else:
                        d_2 = gdf.loc[i, 'geometry'].interpolate(0.6, normalized=True)
                        det_help_double = gdf.loc[i, 'geometry'].interpolate(0.6 + dist_help_rel, normalized=True)
                        name_double.append(d_2)
                        new_loop_distance.append(0.1 * gdf['length'][i])
                    det_help.append(det_help_double)
                name.append(d)
                det_bearing.append(det_help)
            id = gdf.loc[i, ['_id', 'n1', 'n2', 'highway', 'lanes', 'length']]
            id_b.append(id)
    cols = ['detector ' + str(i + 1) for i in range(0, n_det)]
    cols_loops = ['detector ' + str(i + 1) + 'bis' for i in range(0, n_det)]
    name = pd.DataFrame(name, columns=cols)
    id_b = pd.DataFrame(id_b, columns=['_id', 'n1', 'n2', 'highway', 'lanes', 'length'])
    id_b.reset_index(drop=True, inplace=True)
    if make_double_loops:
        cols_bearing = ['bearing ' + str(i + 1) for i in range(0, n_det * 2)]
        det_bearing = pd.DataFrame(det_bearing, columns=cols_bearing)
        name_double = pd.DataFrame(name_double, columns=cols_loops)
        new_loop_distance = pd.DataFrame(new_loop_distance, columns=['loop_distance'])
        name = pd.concat([id_b, name, name_double, det_bearing, new_loop_distance], axis=1)
    else:
        cols_bearing = ['bearing ' + str(i + 1) for i in range(0, n_det)]
        det_bearing = pd.DataFrame(det_bearing, columns=cols_bearing)
        name = pd.concat([id_b, name, det_bearing], axis=1)
    name_2 = gpd.GeoDataFrame(name, geometry=name.loc[:, 'detector 1'])
    return name_2


# Transform x-y coordinates back to lon-lat in WGS84


def get_xy_to_crs(gdf):
    a = pd.Series(gdf['detector_1'])
    a_x = list(a.apply(lambda p: p.x))
    a_y = list(a.apply(lambda p: p.y))
    p = Proj(proj='utm', zone=34, ellps='WGS84', preserve_units=False)
    lon, lat = p(a_x, a_y, inverse=True)
    c = {'lon': lon, 'lat': lat}
    df = pd.DataFrame(c)
    gdf = gdf.drop(['detector_1', 'geometry'], axis=1)
    gdf = pd.concat([df, gdf], axis=1)
    geom = [Point(xy) for xy in zip(gdf.lon, gdf.lat)]
    gdf = gpd.GeoDataFrame(gdf, crs='WGS84', geometry=geom)
    return gdf


def get_xy_to_crs_double_loops(gdf, n_det=1, double_loops=True, lonlat=False):
    new_gdf = []
    individ_gdf = []
    for i in range(1, n_det + 1):
        a = pd.Series(gdf['detector ' + str(i)])
        a_x = list(a.apply(lambda p: p.x))
        a_y = list(a.apply(lambda p: p.y))
        p_d = pd.Series(gdf['bearing ' + str(i)])
        p_x = list(p_d.apply(lambda p: p.x))
        p_y = list(p_d.apply(lambda p: p.y))
        p = Proj(proj='utm', zone=34, ellps='WGS84', preserve_units=False)
        # Detector coordinates transformation
        lon, lat = p(a_x, a_y, inverse=True)
        c = {'lon': lon, 'lat': lat}
        df = pd.DataFrame(c)
        df['lonlat'] = list(zip(df.lon, df.lat))
        df['xy'] = list(zip(a_x, a_y))
        # Bearing points coordinates transformation
        p_lon, p_lat = p(p_x, p_y, inverse=True)
        p_c = {'lon_p': p_lon, 'lat_p': p_lat}
        df_p = pd.DataFrame(p_c)
        p1 = [tuple(xy) for xy in zip(df.lat, df.lon)]
        p2 = [tuple(xy) for xy in zip(df_p.lat_p, df_p.lon_p)]
        bearing = [cpb.calculate_initial_compass_bearing(p1[j], p2[j]) for j in range(0, len(p1))]
        gdf = gdf.drop(['detector ' + str(i)], axis=1)
        gdf = gdf.drop(['bearing ' + str(i)], axis=1)
        if lonlat:
            geom = [Point(xy) for xy in df.lonlat]
            df[f'detector_{i}'] = df.lonlat
        else:
            geom = [Point(xy) for xy in df.xy]
            df[f'detector_{i}'] = df.xy
        gdf = pd.concat([gdf, df[f'detector_{i}']], axis=1)
        gdf.insert(len(gdf.columns), 'detector_bearing_' + str(i), bearing)
        gdf_ind = gdf[['_id', 'n1', 'n2', 'detector_' + str(i), 'detector_bearing_' + str(i)]]
        gdf_ind = gpd.GeoDataFrame(gdf_ind, crs='WGS84', geometry=geom)
        individ_gdf.append(gdf_ind)
    if double_loops:
        for i in range(1, n_det + 1):
            a = pd.Series(gdf['detector ' + str(i) + 'bis'])
            a_x = list(a.apply(lambda p: p.x))
            a_y = list(a.apply(lambda p: p.y))
            p_d = pd.Series(gdf['bearing ' + str(i + n_det)])
            p_x = list(p_d.apply(lambda p: p.x))
            p_y = list(p_d.apply(lambda p: p.y))
            p = Proj(proj='utm', zone=34, ellps='WGS84', preserve_units=False)
            lon, lat = p(a_x, a_y, inverse=True)
            c = {'lon': lon, 'lat': lat}
            df = pd.DataFrame(c)
            df['lonlat'] = list(zip(df.lon, df.lat))
            df['xy'] = list(zip(a_x, a_y))
            # Bearing points coordinates transformation
            p_lon, p_lat = p(p_x, p_y, inverse=True)
            p_c = {'lon_p': p_lon, 'lat_p': p_lat}
            df_p = pd.DataFrame(p_c)
            p1 = [tuple(xy) for xy in zip(df.lat, df.lon)]
            p2 = [tuple(xy) for xy in zip(df_p.lat_p, df_p.lon_p)]
            bearing = [cpb.calculate_initial_compass_bearing(p1[j], p2[j]) for j in range(0, len(p1))]
            gdf = gdf.drop(['detector ' + str(i) + 'bis'], axis=1)
            gdf = gdf.drop(['bearing ' + str(i + n_det)], axis=1)
            if lonlat:
                geom = [Point(xy) for xy in df.lonlat]
                df[f'detector_{i}bis'] = df.lonlat
            else:
                geom = [Point(xy) for xy in df.xy]
                df[f'detector_{i}bis'] = df.xy
            gdf = pd.concat([gdf, df[f'detector_{i}bis']], axis=1)
            gdf.insert(len(gdf.columns), 'detector_bearing_' + str(i) + 'bis', bearing)
            gdf_ind = gdf[['_id', 'n1', 'n2', 'detector_' + str(i) + 'bis', 'detector_bearing_' + str(i) + 'bis']]
            gdf_ind = gpd.GeoDataFrame(gdf_ind, crs='WGS84', geometry=geom)
            individ_gdf.append(gdf_ind)
    gdf = gdf.drop(['geometry'], axis=1)
    if lonlat:
        gdf.attrs['crs'] = crs_pneuma
        gdf.attrs['lonlat'] = True
    else:
        gdf.attrs['crs'] = crs_pneuma_proj
        gdf.attrs['lonlat'] = False
    gdf.attrs['n_det'] = n_det
    new_gdf.append(gdf)
    new_gdf.append(individ_gdf)
    return new_gdf


def make_detector_edges(gdf, distance, double_loops=False, b_begin=True, b_end=False):
    if type(gdf) is list:
        gdf = gdf[0]  # Selecting the dataframe with all detectors together (subject to output of previous functions)
    n_det = gdf.attrs['n_det']
    lonlat = gdf.attrs['lonlat']
    gdf_edges = []
    gdf_edges_all = gdf[['_id', 'n1', 'n2', 'highway', 'lanes', 'length']]
    if double_loops:
        gdf_edges_all = gdf[['_id', 'n1', 'n2', 'highway', 'lanes', 'length', 'loop_distance']]
    gdf_edges.append(gdf_edges_all)
    for ind in range(1, n_det + 1):
        e = []
        f = []
        b_b = []
        b_e = []
        for i, j in gdf.iterrows():
            b1 = gdf[f'detector_bearing_{ind}'][i] - 90
            b2 = gdf[f'detector_bearing_{ind}'][i] + 90
            if double_loops:
                if b_end and not b_begin:
                    b1 = gdf[f'detector_bearing_{ind}bis'][i] - 90
                    b2 = gdf[f'detector_bearing_{ind}bis'][i] + 90
            point = (gdf[f'detector_{ind}'][i][0], gdf[f'detector_{ind}'][i][1])
            point1 = cpb.get_coordinates(b1, point, distance, lonlat=lonlat)
            point2 = cpb.get_coordinates(b2, point, distance, lonlat=lonlat)
            d_edge = LineString([point1, point2])
            e.append(d_edge)
            b_b.append((b1 + 360) % 360)
            if double_loops:
                if b_end:
                    b1 = gdf[f'detector_bearing_{ind}bis'][i] - 90
                    b2 = gdf[f'detector_bearing_{ind}bis'][i] + 90
                point = (gdf[f'detector_{ind}bis'][i][0], gdf[f'detector_{ind}bis'][i][1])
                point1 = cpb.get_coordinates(b1, point, distance, lonlat=lonlat)
                point2 = cpb.get_coordinates(b2, point, distance, lonlat=lonlat)
                d_edge = LineString([point1, point2])
                f.append(d_edge)
                b_e.append((b1 + 360) % 360)
        gdf_edges_all.insert(len(gdf_edges_all.columns), f'det_edge_{ind}', e)
        if double_loops:
            gdf_edges_all.insert(len(gdf_edges_all.columns), f'det_edge_{ind}bis', f)
        gdf_edges_all.insert(len(gdf_edges_all.columns), f'det_bearing_{ind}', b_b)
        if double_loops:
            gdf_edges_all.insert(len(gdf_edges_all.columns), f'det_bearing_{ind}bis', b_e)
        gdf_edge = gdf[['n1', 'n2']]
        gdf_edge = gpd.GeoDataFrame(gdf_edge, geometry=e)
        gdf_edges.append(gdf_edge)
        if double_loops:
            gdf_edge = gdf[['n1', 'n2']]
            gdf_edge_bis = gpd.GeoDataFrame(gdf_edge, geometry=f)
            gdf_edges.append(gdf_edge_bis)
        gdf_edges_all.attrs['crs'] = gdf.attrs['crs']
        gdf_edges_all.attrs['lonlat'] = gdf.attrs['lonlat']
        gdf_edges_all.attrs['n_det'] = gdf.attrs['n_det']
        gdf_edges[0] = gdf_edges_all
    g = [tuple(xy) for xy in zip(gdf_edges_all['n1'], gdf_edges_all['n2'])]
    gdf_edges[0].insert(len(gdf_edges_all.columns), 'edge', g)
    print(gdf_edges_all.attrs)
    return gdf_edges


def edge_features(gdf_special, distance, lonlat=False, select_traffic_signals=True):
    if gdf_special is not None:
        assert {'_id', 'lat', 'lon', 'x', 'y', 'bearing', 'highway'}.issubset(set(gdf_special.columns))
        gdf_special = gdf_special.rename(columns={'highway': 'feature'})
        gdf_special = gdf_special.assign(lonlat=[tuple(xy) for xy in zip(gdf_special.lon, gdf_special.lat)],
                                         xy=[tuple(xy) for xy in zip(gdf_special.x, gdf_special.y)])
        gdf_special.reset_index(inplace=True)
        gdf_special.set_index(['_id', 'index'], inplace=True)
        gdf_special.sort_index(inplace=True)
        gdf_special = gdf_special[~gdf_special.junction]
        ft1 = gdf_special['bearing'] - 90
        ft2 = gdf_special['bearing'] + 90
        if lonlat:
            df_ft = gdf_special.lonlat
            crs = crs_pneuma
        else:
            df_ft = gdf_special.xy
            crs = crs_pneuma_proj
        point1 = [cpb.get_coordinates(ft1[i], df_ft.loc[i], distance, lonlat=lonlat) for i in gdf_special.index]
        point2 = [cpb.get_coordinates(ft2[i], df_ft.loc[i], distance, lonlat=lonlat) for i in gdf_special.index]
        ft_geom = [LineString(xy) for xy in zip(point1, point2)]
        gdf_special = gdf_special.drop('geometry', axis=1)

        df_features = gpd.GeoDataFrame(gdf_special, crs=crs, geometry=ft_geom)
        if select_traffic_signals:
            df_features = df_features[(df_features['feature'] == 'traffic_signals')
                                      | (df_features['crossing'] == 'traffic_signals') |
                                      (df_features['traffic_signals'].isin(['signal', 'traffic_lights']))]
            df_features['det_signal1'] = df_features['geometry']
        df_features.attrs['lonlat'] = lonlat
        df_features.attrs['crs'] = crs
        df_features.attrs['n_det'] = 1
    else:
        df_features = []
    return df_features


def help_det_proj(row, column_name):
    p = row[column_name]
    e1 = (row['x1'], row['y1'])
    e2 = (row['x2'], row['y2'])
    _, t = distxy.project(s1=e1, s2=e2, p=p)
    return t