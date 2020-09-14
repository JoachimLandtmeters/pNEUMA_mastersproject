"""
Created on Wed Oct  9 14:10:17 2019

@author: Joachim Landtmeters
Building the graph of Athens network by using osmnx package
"""

import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import networkx as nx
import pandas as pd
import geopandas as gpd
from collections import Counter, OrderedDict
from shapely.geometry import Point, LineString
from operator import itemgetter
from statistics import mean
import numpy as np
from pylab import *
import pickle
import json

# north, south, east, west = 37.9936, 37.9738, 23.7424, 23.7201 Coordinates for Athens region
bb = 37.9936, 37.9738, 23.7424, 23.7201
# custom filter to include pedestrian streets
adj_filter = '["area"!~"yes"]["highway"!~"cycleway|footway|path|steps' \
'|track|corridor|elevator|escalator|proposed|construction|bridleway' \
'|abandoned|platform|raceway"]["motor_vehicle"!~"no"]["motorcar"!~"no"]' \
'["access"!~"private"]["service"!~"parking|parking_aisle|private|emergency_access"]'


class CreateNetwork:

    def __init__(self, bounding_box, network_type='drive_service', crs='epsg:4326', tags=None, simplify_strict=False,
                 traffic_signals=True, custom_filter=None):
        # researched area (bounding box)
        self.bounding_box = bounding_box
        self.network_type = network_type
        self.custom_filter = custom_filter
        self.strict = simplify_strict
        self.tags = tags
        self.crs = crs
        if tags is None:
            self.tags = ['bridge', 'tunnel', 'oneway', 'lanes', 'ref', 'name',
                         'highway', 'busway:both', 'busway:left', 'busway:right',
                         'maxspeed', 'service', 'access', 'area',
                         'landuse', 'width', 'est_width', 'junction', 'surface']
        # download the road network from OSM
        ox.settings.useful_tags_path = self.tags
        if custom_filter:
            self.graph = ox.graph_from_bbox(self.bounding_box[0], self.bounding_box[1], self.bounding_box[2],
                                            self.bounding_box[3], custom_filter=self.custom_filter
                                            , simplify=self.strict)
        else:
            self.graph = ox.graph_from_bbox(self.bounding_box[0], self.bounding_box[1], self.bounding_box[2],
                                        self.bounding_box[3], network_type=self.network_type , simplify=self.strict)
        if not self.strict:
            self.graph = ox.simplify_graph(self.graph, strict=False)
        self.graph_latlon = ox.project_graph(self.graph, to_crs=self.crs)
        if traffic_signals:

            with open('export.geojson') as f:  # Traffic signals in bounding box extracted via Overpass Turbo query
                ts = json.load(f)

            df_ts = pd.DataFrame(ts['features'])
            dict_traffic_signals = {'coordinates': [], 'highway': [], 'geometry': []}
            for ind, val in df_ts.iterrows():
                for i, j in enumerate(df_ts):
                    if j == 'properties':
                        dict_traffic_signals['highway'].append(val[j]['highway'])
                    if j == 'geometry':
                        dict_traffic_signals['coordinates'].append(val[j]['coordinates'])
                        dict_traffic_signals['geometry'].append(Point(val[j]['coordinates'][0],
                                                                      val[j]['coordinates'][1]))
            self.traffic_signals = gpd.GeoDataFrame(dict_traffic_signals, geometry=dict_traffic_signals['geometry'])
        self.network_matrix = pd.DataFrame()

    def network_dfs(self, print_info=False):
        G = self.graph_latlon
        new_graph = ox.add_edge_bearings(G)
        edge_bus_lanes_left = list(new_graph.edges.data('busway:left', default=False))
        edge_bus_lanes_right = list(new_graph.edges.data('busway:right', default=False))
        left = [j[2] for i, j in enumerate(edge_bus_lanes_left)]
        right = [j[2] for i, j in enumerate(edge_bus_lanes_right)]
        n, e = ox.graph_to_gdfs(new_graph)
        e = e.assign(dbl_left=left)
        e = e.assign(dbl_right=right)
        e = e.drop(['busway:left', 'busway:right'], axis=1)
        dbl_bool = np.logical_and(e['dbl_left'].values, e['oneway'].values)
        gdf_val = e[['u', 'v', 'bearing']].values
        new_rows = []
        new_index = len(e)
        for row, val in e.iterrows():
            if dbl_bool[row]:
                if print_info:
                    print(row)
                new_row = val.copy()
                new_row['u'] = int(gdf_val[row][1])
                new_row['v'] = int(gdf_val[row][0])
                new_row['lanes'] = 1
                new_row['bearing'] = gdf_val[row][2] - 180
                new_row['osmid'] = 'new_edge'
                new_row['geometry'] = [LineString([n['geometry'][gdf_val[row][1]],
                                                   n['geometry'][gdf_val[row][0]]])]
                # print(dict(new_row), dict(val))
                new_row = gpd.GeoDataFrame(dict(new_row), index=[new_index])
                new_index += 1
                new_rows.append(new_row)
        if new_rows:
            new_rows = pd.concat(new_rows, axis=0)
            if print_info:
                print(new_rows)
            e = pd.concat([e, new_rows], axis=0)
            new_graph = ox.gdfs_to_graph(n, e)
            n, e = ox.graph_to_gdfs(new_graph)
            network_matrix = e.loc[:,
                         ['u', 'v', 'oneway', 'osmid', 'highway', 'length', 'bearing', 'geometry', 'lanes',
                          'dbl_left', 'dbl_right']]
            network_nodes_small = n.loc[:, ['y', 'x']]
        else:
            network_matrix = e.loc[:,
                             ['u', 'v', 'oneway', 'osmid', 'highway', 'length', 'bearing', 'geometry', 'lanes',
                              'dbl_left', 'dbl_right']]
            network_nodes_small = n.loc[:, ['y', 'x']]
        network_matrix = network_matrix.join(network_nodes_small, on='u')
        network_matrix = network_matrix.rename(columns={'u': 'N1', 'y': 'Lat1', 'x': 'Long1'})
        network_matrix = network_matrix.join(network_nodes_small, on='v')
        network_matrix = network_matrix.rename(columns={'v': 'N2', 'y': 'Lat2', 'x': 'Long2'})
        cols = ['osmid', 'N1', 'Lat1', 'Long1', 'N2', 'Lat2', 'Long2', 'length', 'lanes', 'oneway', 'bearing', 'highway'
            , 'dbl_left', 'dbl_right', 'geometry']
        network_matrix = network_matrix[cols]  # rearranging columns (reader's convenience)
        network_matrix.reset_index(inplace=True)  # From hereon the unique index of an edge is just its position in df
        self.graph_latlon = new_graph
        self.graph = ox.project_graph(self.graph_latlon)
        self.network_matrix = network_matrix
        return network_matrix, new_graph, new_rows

    def link_traffic_signals_to_edge(self):
        G = self.graph_latlon
        gdf_netw = self.network_matrix
        gdf_ts = self.traffic_signals
        lon = [j[0] for i, j in gdf_ts['coordinates'].iteritems()]
        lat = [j[1] for i, j in gdf_ts['coordinates'].iteritems()]
        nearest = ox.get_nearest_edges(G, lon, lat, method='balltree')
        nearest = [tuple(j) for i, j in enumerate(nearest)]
        gdf_ts['edge'] = nearest
        gdf_ts['N1'] = [j[0] for i, j in enumerate(nearest)]
        gdf_ts['N2'] = [j[1] for i, j in enumerate(nearest)]
        df_ts_e = pd.merge(gdf_netw[['N1', 'N2', 'geometry']], gdf_ts[['N1', 'N2']], how='inner',
                           on=['N1', 'N2'])
        return df_ts_e

    def plot_dbl(self):
        network_matrix = self.network_matrix
        fig, ax = plt.subplots()
        network_matrix.plot(ax=ax, edgecolor='lightgrey')
        network_matrix[network_matrix['dbl_left'] == 'opposite_lane']. \
            plot(ax=ax, edgecolor='r', linewidth=3, label='DBL: Contra flow')
        network_matrix[network_matrix['dbl_left'] == 'opposite_way']. \
            plot(ax=ax, edgecolor='r', linewidth=3)
        network_matrix[network_matrix['dbl_right'] == 'lane'].plot(ax=ax, edgecolor='g', linewidth=3,
                                                                   label='DBL: With flow')
        network_matrix[(network_matrix['dbl_right'] == 'lane') & (network_matrix['dbl_left'] == 'opposite_lane')].plot(
            ax=ax, edgecolor='purple', linewidth=3, label='DBL: Both directions')
        network_matrix[(network_matrix['dbl_right'] == 'lane') & (network_matrix['dbl_left'] == 'opposite_way')].plot(
            ax=ax, edgecolor='purple', linewidth=3)
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
        colors = ['lightgrey', 'r', 'k', 'pink', 'blue', 'orange', 'g', 'm', 'c', 'darkred', 'pink']
        keys = list(Counter(n_lanes).keys())
        keys.sort()
        col_dict = OrderedDict(zip(keys, colors))
        print(col_dict)
        lane_colors = [col_dict[x] for x in n_lanes]
        fig, ax = ox.plot_graph(G, edge_linewidth=n_lanes, edge_color=lane_colors,
                                show=False, close=False, node_size=1, fig_height=7, fig_width=7, margin=0)
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
                label = f'{int(k)} to {int(k)+1} lanes (list)'
            else:
                label = f'{int(k)} lanes'
            legend_elements[idx] = Line2D([0], [0], marker='s', color="#061529", label=label,
                                          markerfacecolor=col_dict[k], markersize=markersize)
        ax.legend(handles=legend_elements, frameon=True, framealpha=0.7, loc='lower left',
                  fontsize=6)
        fig.suptitle('Athens network with colors and width of edges wrt lanes')
        plt.show()

    def save_graph_to_shp(self):
        network_nodes, _ = ox.graph_to_gdfs(self.graph_latlon)
        network_matrix_shp = self.network_matrix.copy()
        network_nodes_shp = network_nodes.copy()
        network_matrix_shp.crs = 'epsg:4326'
        network_nodes_shp.crs = 'epsg:4326'
        ox.save_load.save_gdf_shapefile(network_matrix_shp, filename="athens_network", folder="athens")
        ox.save_load.save_gdf_shapefile(network_nodes_shp, filename="athens_nodes", folder="athens")



# Show network with strict and non-strict simplification
# ox.plot_graph(research_area, show=False, close=False)
# plt.title('Strict simplification')
# ox.plot_graph(research_area_nonstrict, show=False, close=False)
# plt.title('Non-strict simplification')

# Projection to Lat-Long in WGS-84


# Add edge bearings as attribute and build matrix with nodes of edges and their respective Lat-Long coordinates

""" 
with open('network_matrix_it2_nonstrict.pkl', 'wb') as g:
    pickle.dump(network_matrix, g)

with open('athens_network_it2_nonstrict.pkl', 'wb') as h:
    pickle.dump(athens_graph, h)
"""