"""Defines WebotsRoad Class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import math


class WebotsRoad(object):
    """User Configurable Road Structure to use in Webots environment"""
    def __init__(self, number_of_lanes=2):
        self.def_name = "STRROAD"
        self.road_type = "StraightRoadSegment"
        self.rotation = [0, 1, 0, math.pi/2]
        self.position = [0, 0.02, 0]
        self.number_of_lanes = number_of_lanes
        self.width = self.number_of_lanes * 3.5
        self.length = 1000
        self.right_border_bounding_object = True
        self.left_border_bounding_object = True
        self.extra_road_parameters = []

    def get_lane_width(self):
        """Get the width of one lane for this road."""
        return self.width / self.number_of_lanes
