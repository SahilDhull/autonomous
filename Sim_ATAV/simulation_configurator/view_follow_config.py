"""Defines ViewFollowConfig class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class ViewFollowConfig(object):
    """Configuration structure for Simulation View Follow point. (To follow a vehicle etc.)"""
    def __init__(self, item_type=None, item_index=None, position=None, rotation=None):
        self.item_type = item_type
        self.item_index = item_index
        self.position = position  # position is 1x3 array: x,y,z
        self.rotation = rotation  # orientation is 1x4 array: x,y,z for defining rotation axis, theta for rotation angle
