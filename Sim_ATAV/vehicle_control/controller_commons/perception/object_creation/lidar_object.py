"""Defines LidarObject class.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class LidarObject(object):
    OBJECT_CAR = 0
    OBJECT_PEDESTRIAN = 1
    OBJECT_BIKE = 2
    OBJECT_TRUCK = 3

    """LidarObject class defines features of the object detected by LIDAR."""
    def __init__(self, lidar_cluster, object_class, relative_position):
        self.lidar_cluster = lidar_cluster
        self.object_class = object_class
        self.relative_position = relative_position

