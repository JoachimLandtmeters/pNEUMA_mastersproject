"""
Visualize network with trajectories on map
"""
from bokeh.io import show, output_file, output_notebook
from bokeh.models import HoverTool, CustomJS, TapTool, OpenURL, Label
from bokeh.tile_providers import get_provider, Vendors
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models.glyphs import MultiLine, Circle, X, Asterisk, Patches
from pneumapackage.iodata import *
from pyproj import Proj
import leuvenmapmatching.util.dist_euclidean as distxy
import folium
from tqdm.contrib import tenumerate

colors = {'Car': 'darkgreen', 'Bus': 'r', 'Motorcycle': 'b', 'Medium Vehicle': 'y', 'Heavy Vehicle': 'pink',
          'Taxi':'darkorange'}
athens = 37.983810, 23.727539


def getLineCoords(row, geom, coord_type):
    """Returns a list of coordinates ('x' or 'y') of a LineString geometry"""
    if coord_type == 'x':
        return list( row[geom].coords.xy[0] )
    elif coord_type == 'y':
        return list( row[geom].coords.xy[1] )


def bokeh_gdfTransform(gdf):
    m_gdf = gdf.drop('geometry', axis=1).copy()
    m_gdf = ColumnDataSource(m_gdf)
    return m_gdf


def plot_trajectories(traj, notebook=False, plot_size=1300, colors=None):
    if notebook:
        plot_size = 900
    plot = figure(plot_height=plot_size,
                  plot_width=plot_size, x_axis_type="mercator", y_axis_type="mercator",
                  aspect_ratio=1, toolbar_location='below')
    #if type(traj) is list:
        #for ind, tr in enumerate(traj):
            #plot.multi_line('x', 'y', source=tr, color=colors[tr['type'][0]], line_width=3)
    #else:
        #plot.multi_line('x', 'y', source=traj, color=colors[traj['type'][0]], line_width=3)
    tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)
    plot.add_tile(tile_provider)
    if notebook:
        output_notebook()
    else:
        output_file(f"trajectories_pneuma.html")
    show(plot)


def traj_folium(location, traj, background='Stamen Toner', zoom=10, colors=None):
    m = folium.Map(location=location, tiles=background,
                   zoom_start=zoom, control_scale=True)
    tll = [[tuple(xy) for xy in zip(tt.lat, tt.lon)] for i, tt in tenumerate(traj)]
    veh_type = [tt['type'][0] for i, tt in tenumerate(traj)]
    for ind, tr in tenumerate(tll):
        if colors is None:
            folium.PolyLine(tr,
                            color='r',
                            weight=2,
                            opacity=0.8).add_to(m)
        else:
            folium.PolyLine(tr,
                    color=colors[veh_type[ind]],
                    weight=2,
                    opacity=0.8).add_to(m)
    outfp = "athens3.html"
    m.save(outfp)


def make_xt_plot(df_crossings, gdf_traj):
    pass

"""
output_file("tile.html")

tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)

# range bounds supplied in web mercator coordinates
p = figure(x_range=(-2000000, 6000000), y_range=(-1000000, 7000000),
           x_axis_type="mercator", y_axis_type="mercator")
p.add_tile(tile_provider)

show(p)
"""