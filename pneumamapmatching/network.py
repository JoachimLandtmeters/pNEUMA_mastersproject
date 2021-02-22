"""
Created on Wed Oct  9 14:10:17 2019

@author: Joachim Landtmeters
Building the graph of Athens network by using osmnx package
"""

from pneumapackage.settings import *

import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import networkx as nx
import pandas as pd
import geopandas as gpd
from collections import Counter, OrderedDict
from shapely.geometry import Point, LineString, Polygon
import math
import pyproj
import itertools

from operator import itemgetter
from statistics import mean
import numpy as np
from pylab import *
import pickle
import json


class Box:

    def __init__(self, bbox, epsg_proj=None):
        """

        :param bbox: ordered North, South, East, West (lat, lat, lon, lon)
        """
        self.bounding_box = bbox
        self.north = bbox[0]
        self.south = bbox[1]
        self.east = bbox[2]
        self.west = bbox[3]
        self.crs_proj = epsg_proj
        self.corners_lonlat = self.get_corners_lonlat()
        self.corners_proj = self.get_corners_proj()
        self.crs_lonlat = crs_pneuma

    def get_lat(self):
        return [self.north, self.south]

    def get_lon(self):
        return [self.east, self.west]

    def get_x(self):
        xs = [i[0] for i in self.corners_proj]
        return xs

    def get_y(self):
        ys = [i[1] for i in self.corners_proj]
        return ys

    def get_corners_lonlat(self):
        pts = [(r[0], r[1]) for r in itertools.product(self.get_lon(), self.get_lat())]
        pts = [pts[0], pts[1], pts[3], pts[2]]
        return pts

    def get_corners_proj(self):
        pts, proj = project_point(self.corners_lonlat, epsg_proj=self.crs_proj, return_proj=True)
        self.crs_proj = proj
        return pts

    def get_polygon(self, lonlat=False):
        pts = self.corners_proj
        if lonlat:
            pts = self.corners_lonlat
        bb_polygon = Polygon(pts)
        return bb_polygon


class CreateNetwork:

    def __init__(self, bounding_box, network_type='drive_service', crs='epsg:4326', tags_nodes=None, tags_edges=None,
                 simplify_strict=False, custom_filter=None, truncate_by_edge=False):
        # researched area (bounding box)
        self.bounding_box = bounding_box
        self.network_type = network_type
        self.custom_filter = custom_filter
        self.strict = simplify_strict
        self.tags_nodes = tags_nodes
        self.tags_edges = tags_edges
        self.crs = crs
        if tags_edges is None:
            self.tags_edges = ['bridge', 'tunnel', 'oneway', 'lanes', 'name',
                               'highway', 'busway', 'busway:both', 'busway:left', 'busway:right',
                               'maxspeed', 'service', 'access', 'area',
                               'landuse', 'width', 'est_width', 'junction', 'surface', 'turn']
        if tags_nodes is None:
            self.tags_nodes = ['highway', 'public_transport', 'traffic_signals', 'crossing']
        # download the road network from OSM
        ox.config(useful_tags_way=self.tags_edges, useful_tags_node=self.tags_nodes)
        self.graph_latlon = ox.graph_from_bbox(self.bounding_box[0], self.bounding_box[1], self.bounding_box[2],
                                               self.bounding_box[3], network_type=self.network_type,
                                               custom_filter=self.custom_filter, simplify=self.strict,
                                               truncate_by_edge=truncate_by_edge)
        self.graph_xy = ox.project_graph(self.graph_latlon)
        self.graph_raw = self.graph_latlon
        self.network_edges = pd.DataFrame()
        self.network_nodes = pd.DataFrame()
        self.used_network = pd.DataFrame()
        self.mm_id = 0
        self.node_tags = node_tags(self.graph_raw, tag='highway')

    def network_dfs(self):
        g = self.graph_latlon
        if not self.strict:
            g = ox.simplify_graph(g, strict=self.strict)
        g = ox.add_edge_bearings(g, precision=1)
        n, e = ox.graph_to_gdfs(g)
        e = e.reset_index()  # Method graph_to_gdfs changed to multiindex df
        network_edges, network_nodes_small = dbl_cleaning(ndf=n, edf=e)
        network_edges = network_edges.join(network_nodes_small, on='u')
        network_edges = network_edges.rename(columns={'u': 'n1', 'y': 'lat1', 'x': 'lon1'})
        network_edges = network_edges.join(network_nodes_small, on='v')
        network_edges = network_edges.rename(columns={'v': 'n2', 'y': 'lat2', 'x': 'lon2'})
        x1, y1 = zip(*project_point(list(zip(network_edges.lon1, network_edges.lat1))))
        x2, y2 = zip(*project_point(list(zip(network_edges.lon2, network_edges.lat2))))
        network_edges = network_edges.assign(x1=x1, y1=y1, x2=x2, y2=y2)
        network_edges['edge'] = list(zip(network_edges['n1'].values, network_edges['n2'].values))
        network_edges.reset_index(inplace=True)  # From hereon the unique index of an edge is just its position in df
        network_edges = network_edges.rename(columns={'index': '_id'})
        self.graph_latlon = g
        self.graph_xy = ox.project_graph(self.graph_latlon)
        self.network_edges = network_edges
        self._get_network_nodes(network_edges)
        # link node_tags to specific edge, osmid not unique over edges after simplification
        nearest = ox.get_nearest_edges(self.graph_xy, self.node_tags.x.to_list(), self.node_tags.y.to_list(),
                                       method='kdtree', dist=1)
        n1, n2, _ = zip(*nearest)
        test_b1 = network_edges[['_id','edge', 'bearing']][network_edges.edge.isin(list(zip(n1, n2)))].values
        test_b2 = network_edges[['_id','edge', 'bearing']][network_edges.edge.isin(list(zip(n2, n1)))].values
        self.node_tags['edge'] = [ij for ij in zip(n1, n2)]
        self.node_tags.reset_index(inplace=True)
        self.node_tags = self.node_tags.merge(self.network_edges[['edge', 'bearing']], on='edge',
                                              suffixes=('','_edge'))
        diff_b = abs(self.node_tags['bearing']-self.node_tags['bearing_edge'])
        for i, j in diff_b.iteritems():
            if (j > 45) and not self.node_tags.junction[i]:
                self.node_tags.at[i, 'edge'] = (self.node_tags.at[i, 'edge'][1], self.node_tags.at[i, 'edge'][0])
        self.node_tags.drop('bearing_edge', axis=1, inplace=True)
        self.node_tags = self.node_tags.merge(self.network_edges[['_id', 'edge', 'bearing']], on='edge',
                                              suffixes=('', '_edge'))
        diff_b2 = abs(self.node_tags['bearing'] - self.node_tags['bearing_edge'])
        breakpoint()
        # check if nearest edge is in right direction, problem with two way streets

        self.node_tags.set_index('index', inplace=True)
        self.node_tags.sort_index(inplace=True)

    def plot_dbl(self, new_added=False):
        network_matrix = self.network_edges
        fig, ax = plt.subplots()
        network_matrix.plot(ax=ax, edgecolor='lightgrey')
        network_matrix[network_matrix['dbl_left']].plot(ax=ax, edgecolor='r', linewidth=3, label='DBL: Contra flow')
        network_matrix[network_matrix['dbl_right']].plot(ax=ax, edgecolor='g', linewidth=3, label='DBL: With flow')
        network_matrix[np.logical_and(network_matrix['dbl_right'], network_matrix['dbl_left'])].plot(
            ax=ax, edgecolor='purple', linewidth=3, label='DBL: Both directions')
        if new_added:
            str_new = 'new_edge'
            network_matrix[network_matrix['osmid'] == str_new].plot(ax=ax, edgecolor='y', linewidth=3,
                                                                    label='Newly Added')
        ax.legend(loc='upper left')
        fig.suptitle('Dedicated bus lanes in Athens research area')
        plt.show()

    def plot_network_lanes(self):
        # Plot graph with number of lanes, colours for categorisation of roads
        G = self.graph_latlon
        edge_lanes = list(G.edges.data('lanes', default='0.5'))
        n_lanes = [x[2] for x in edge_lanes]
        for num, i in enumerate(n_lanes):
            t = type(i)
            if t is list:
                n_lanes[num] = [float(y) for y in n_lanes[num]]
                n_lanes[num] = mean(n_lanes[num])
                print(num)
            else:
                n_lanes[num] = float(n_lanes[num])
        n_lanes = [float(x) for x in n_lanes]
        ## Creating a pos_list based on longitude and latitude
        labels = nx.get_edge_attributes(G, 'lanes')
        colors = ['lightgrey', 'r', 'orange', 'y', 'blue', 'g', 'm', 'c', 'pink', 'darkred']
        keys = list(Counter(n_lanes).keys())
        keys.sort()
        col_dict = OrderedDict(zip(keys, colors))
        print(col_dict)
        lane_colors = [col_dict[x] for x in n_lanes]
        fig, ax = ox.plot_graph(G, edge_linewidth=n_lanes, edge_color=lane_colors,
                                show=False, close=False, node_size=1)
        markersize = 6
        legend_elements = [0] * len(keys)
        for k, v in col_dict.items():
            idx = keys.index(k)
            if float(k) < 1:
                label = 'NaN'
                idx = 0
            elif float(k) == 1:
                label = ' 1 lane'
                idx = 1
            elif float(k) > int(k):
                label = f'{int(k)} to {int(k) + 1} lanes (list)'
            else:
                label = f'{int(k)} lanes'
            legend_elements[idx] = Line2D([0], [0], marker='s', color="#061529", label=label,
                                          markerfacecolor=col_dict[k], markersize=markersize)
        ax.legend(handles=legend_elements, frameon=True, framealpha=0.7, loc='lower left',
                  fontsize=6)
        fig.suptitle('Athens network with colors and width of edges wrt lanes')
        plt.show()

    def _get_network_nodes(self, network_edges):
        n1 = network_edges[['n1', 'lat1', 'lon1', 'x1', 'y1']]
        n2 = network_edges[['n2', 'lat2', 'lon2', 'x2', 'y2']]
        n2 = n2.rename(columns={'n2': 'n1', 'lat2': 'lat1', 'lon2': 'lon1', 'x2': 'x1', 'y2': 'y1'})
        n = pd.concat([n1, n2], axis=0)
        n.drop_duplicates(inplace=True)
        self.network_nodes = n

    def add_used_network(self, used_network):
        self.used_network = used_network

    def add_mapmatch_tag(self, tag):
        self.mm_id = tag

    def save_graph_to_shp(self, path='data/shapefiles', latlon=True):
        g = self.graph_xy
        if latlon:
            g = self.graph_latlon
        ox.save_graph_shapefile(g, filepath=path)


def node_tags(g, tag='highway'):
    n = [n for n in g.nodes(data=True) if tag in n[1].keys()]
    nid, val = zip(*n)
    ndf = pd.DataFrame(val, index=nid)
    ndf = ndf.rename(columns={'x': 'lon', 'y': 'lat'})
    x, y = zip(*project_point(list(zip(ndf.lon, ndf.lat))))
    ndf = ndf.assign(x=x, y=y, junction=list(ndf.street_count > 2))
    g = ox.add_edge_bearings(g, precision=1)
    edf = ox.graph_to_gdfs(g, nodes=False, edges=True)
    edf.reset_index(inplace=True)
    edf = edf[['u', 'v', 'osmid', 'bearing']]
    edf = edf.explode('osmid', ignore_index=True)
    edf.set_index('u', inplace=True)
    ndf = pd.merge(ndf, edf[['osmid', 'bearing']], how='left', left_index=True, right_index=True)
    # check for na values
    ndf_na = ndf[ndf.osmid.isna()].copy()
    ndf.drop(ndf_na.index, inplace=True)
    ndf_na.drop(['osmid', 'bearing'], axis=1, inplace=True)
    edf_na = edf.reset_index()
    edf_na.set_index('v', inplace=True)
    ndf_na = pd.merge(ndf_na, edf_na[['osmid', 'bearing']], how='left', left_index=True, right_index=True)
    ndf = pd.concat([ndf, ndf_na], axis=0)
    ndf = ndf.astype({'osmid': 'int64'})
    ndf = gpd.GeoDataFrame(ndf, geometry=gpd.points_from_xy(ndf.lon, ndf.lat))
    return ndf


def dbl_cleaning(ndf, edf):
    if 'busway:left' and 'busway:right' not in edf.columns:
        network_edges = edf.loc[:, ['u', 'v', 'oneway', 'osmid', 'highway', 'length', 'bearing', 'geometry', 'lanes']]
        network_nodes_small = ndf.loc[:, ['y', 'x']]
        return network_edges, network_nodes_small
    new_rows = []
    left_na = pd.isna(edf['busway:left'])
    right_na = pd.isna(edf['busway:right'])
    edf = edf.assign(dbl_left=~left_na)
    edf = edf.assign(dbl_right=~right_na)
    # Temporal addition to change all dbl in network
    for r, v in edf.iterrows():
        if v.busway == 'opposite_lane' and not v.dbl_left:
            edf.loc[r, 'dbl_left'] = True
    edf = edf.drop(['busway:left', 'busway:right'], axis=1)
    dbl_bool = np.logical_and(edf['dbl_left'].values, edf['oneway'].values)
    gdf_val = edf[['u', 'v', 'bearing']].values
    new_index = len(edf)
    for row, val in edf.iterrows():
        if dbl_bool[row]:
            new_row = val.copy()
            new_row['u'] = int(gdf_val[row][1])
            new_row['v'] = int(gdf_val[row][0])
            new_row['lanes'] = 1
            new_row['bearing'] = gdf_val[row][2] - 180
            new_row['osmid'] = -1
            new_row['geometry'] = [LineString([ndf['geometry'][gdf_val[row][1]],
                                               ndf['geometry'][gdf_val[row][0]]])]
            new_row = gpd.GeoDataFrame(dict(new_row), index=[new_index])
            new_index += 1
            new_rows.append(new_row)
    if new_rows:
        new_rows = pd.concat(new_rows, axis=0)
        edf = pd.concat([edf, new_rows], axis=0)
        edf.set_index(['u', 'v', 'key'], inplace=True)
        new_graph = ox.graph_from_gdfs(ndf, edf)
        ndf, edf = ox.graph_to_gdfs(new_graph)
        edf.reset_index(inplace=True)
    network_edges = edf.loc[:, ['u', 'v', 'oneway', 'osmid', 'highway', 'length', 'bearing', 'geometry',
                                'lanes', 'dbl_left', 'dbl_right']]
    network_nodes_small = ndf.loc[:, ['y', 'x']]
    return network_edges, network_nodes_small


def project_point(point, lonlat=False, epsg_proj=None, hemisphere='north', return_proj=False):
    """

    :param return_proj:
    :param hemisphere:
    :param epsg_proj:
    :param lonlat:
    :param point: longitude-latitude or x-y coordinates
    :param lonlat: project from or to latlon, i.e. True transforms xy into latlon
    :return: projected point (x,y) or (lon, lat)
    """

    if epsg_proj is not None:
        try:
            crs = pyproj.CRS.from_epsg(epsg_proj)
        except pyproj.exceptions.CRSError:
            raise ValueError('EPSG code is not valid')
        epsg_proj = crs
    else:
        if lonlat:
            raise ValueError('Projected EPSG-code unknown for these projected coordinates, not possible to reproject')
        else:
            lon = point[0]
            if isinstance(point, list) and len(point) > 1:
                lon = np.mean([i[0] for i in point])
            utm_zone = math.floor((lon + 180) / 6) + 1
            if hemisphere == 'north':
                nth = True
                sth = False
            else:
                nth = False
                sth = True
            tmp_crs = pyproj.CRS({'proj': 'utm', 'zone': utm_zone, 'north': nth, 'south': sth})
            epsg_proj = pyproj.CRS(tmp_crs.to_epsg())
    t = pyproj.Transformer.from_crs(crs_pneuma, epsg_proj, always_xy=True)
    t_direction = 'FORWARD'
    if lonlat:
        t_direction = 'INVERSE'
    if isinstance(point, list) and len(point) > 1:
        ls_points = list(t.itransform(point, direction=t_direction))
        if return_proj:
            return ls_points, epsg_proj
        else:
            return ls_points
    else:
        cd = t.transform(point[0], point[1], direction=t_direction)
        if return_proj:
            return cd, epsg_proj
        else:
            return cd


def project_gdf(gdf, epsg_proj=None):
    gdf = gdf.to_crs(epsg_proj)
    return gdf
