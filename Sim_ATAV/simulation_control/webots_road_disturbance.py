"""Defines WebotsRoadDisturbance Class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class WebotsRoadDisturbance(object):
    """User Configurable Road Disturbance (small triangles on the road) to use in Webots"""

    TRIANGLE_DOUBLE_SIDED = 0
    TRIANGLE_FULL_LENGTH = 1
    TRIANGLE_LEFT_HALF = 2
    TRIANGLE_RIGHT_HALF = 3

    def __init__(self):
        self.disturbance_id = 1
        self.disturbance_type = self.TRIANGLE_FULL_LENGTH
        self.rotation = [0, 1, 0, 0]
        self.position = [0, 0, 0]
        self.width = 3.5
        self.length = 100
        self.height = 0.06
        self.surface_height = 0.02
        self.inter_object_spacing = 1.0
