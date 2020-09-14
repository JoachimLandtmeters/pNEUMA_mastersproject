# Reading csv file with trajectories of Athens area
# Making list of all trajectory dataframes --> from seperate files to one file
# Calculating bearing for every datapoint in individual trajectory
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString
import pickle
import matplotlib.pyplot as plt
import compassbearing
from tqdm import tqdm
from tqdm.contrib import tenumerate

path_to_export = "~/Desktop/Joachim/Master Logistics and Traffic/Erasmus Lausanne/" \
                 "Masters Project/athens_project/dataset_csv/"
trajectories = []
crs = 'WGS84'  # Lat-Lon in right crs, used for geodataframe column
number_of_csvfiles = 10660
for id in tqdm(range(1, number_of_csvfiles)):
    # range dependent on csv file, saving the index of python file with trajectory extraction
    t = pd.read_csv(f'{path_to_export}trajectory{id}.csv')
    t = t.rename(columns={'Latitude [deg]': 'Lat', 'Longitude [deg]': 'Lon'})
    geometry = [Point(yx) for yx in zip(t.Lon, t.Lat)]
    # t=t.drop(['Lat','Lon'], axis=1) X and Y needed for get_nearest_edge
    t = GeoDataFrame(t, crs=crs, geometry=geometry)
    t = t[:-1]
    t['Tracked Vehicle'].fillna(t['Tracked Vehicle'].values[0], inplace=True)
    trajectories.append(t)

column_number = 14  # Specific number puts column at preferred place in dataframe
for i, j in tenumerate(trajectories):
    bearing = []
    traj_latlon = j[['Lat', 'Lon']].values
    for e, f in j.iterrows():
        if e < len(j) - 1:
            A = (traj_latlon[e][0], traj_latlon[e][1])
            B = (traj_latlon[e + 1][0], traj_latlon[e + 1][1])
            comp = compassbearing.calculate_initial_compass_bearing(A, B)
            if comp == 0:
                if not bearing:
                    r = 2
                    while A == (traj_latlon[e + r][0], traj_latlon[e + r][1]) \
                            and r + e + 1 < len(j):
                        r = r + 1
                    if A == (traj_latlon[e + r][0], traj_latlon[e + r][1]):
                        bearing.append(999)  # Static vehicle
                    else:
                        C = (traj_latlon[e + r][0], traj_latlon[e + r][1])
                        comp_1 = compassbearing.calculate_initial_compass_bearing(A, C)
                        bearing.append(comp_1)
                else:
                    comp_2 = bearing[e - 1]
                    bearing.append(comp_2)
            else:
                bearing.append(comp)
        elif len(j) > 1:
            comp = bearing[e - 1]
            bearing.append(comp)
        else:
            bearing.append(999)  # Static vehicle
    j.insert(column_number, "bearing", bearing)

static_vehicle = []
trajectories_moving = []
for i, j in tenumerate(trajectories):
    j = j.rename(columns={'Traveled Dist.[m]': 'traveled_dist',
                          'Speed[km / h]': 'speed', 'Tan.Accel.[ms - 2]': 'tan_accel',
                          'Lat.Accel.[ms - 2]': 'lat_accel', 'Time[ms]': 'time'})
    if j['bearing'].nunique() == 1:
        static_vehicle.append(j)
    else:
        trajectories_moving.append(j)

"""
with open('trajects.pkl', 'wb') as f:
    pickle.dump(trajectories, f)
with open('trajectories_moving.pkl', 'wb') as f:
    pickle.dump(trajectories_moving, f)
with open('static_vehicles.pkl', 'wb') as f:
    pickle.dump(static_vehicle, f)
"""