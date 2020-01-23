"""Defines RadarObject class.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class RadarObject(object):
    OBJECT_CAR = 0
    OBJECT_PEDESTRIAN = 1
    OBJECT_BIKE = 2
    OBJECT_TRUCK = 3

    """RadarObject class defines features of the object detected by radar."""
    def __init__(self, radar_target, object_class, relative_position):
        self.radar_target = radar_target
        self.object_class = object_class
        self.relative_position = relative_position

