# LICENSE: public domain
import math
from shapely.geometry import Point
def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.

    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))

    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees

    :Returns:
      The bearing in degrees

    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def get_coordinates(bearing, point, distance):
    # Bearing clockwise from north to east
    bearing=math.radians(bearing)
    lat=math.radians(point[0])
    dx=distance*math.sin(bearing)
    dy=distance*math.cos(bearing)
    polar=110540 # Polar circumference for one degree latitude for WGS-84 ellipsoid
    equatorial=111320 # Equatorial circumference for one degree longitude for WGS-84 ellipsoid
    d_lon=dx/(equatorial*math.cos(lat))
    d_lat=dy/polar

    lon2=point[1]+d_lon
    lat2=point[0]+d_lat

    new_point=Point(lon2,lat2)
    return new_point
