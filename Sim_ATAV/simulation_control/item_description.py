"""Define ItemDescription class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class ItemDescription(object):
    """ItemDescription class holds a reference to a simulation item
    and possibly a state of the item"""
    ITEM_TYPE_TIME = 1
    ITEM_TYPE_VEHICLE = 2
    ITEM_TYPE_PEDESTRIAN = 3
    ITEM_TYPE_VEHICLE_DET_PERF = 4
    ITEM_TYPE_PED_DET_PERF = 5
    ITEM_TYPE_VEHICLE_CONTROL = 6
    ITEM_TYPE_DET_EVAL = 7
    VISIBILITY_EVAL = 8

    VEHICLE_CONTROL_THROTTLE = 0
    VEHICLE_CONTROL_STEERING = 1
    VEHICLE_CONTROL_THROTTLE_ALT = 2
    VEHICLE_CONTROL_STEERING_ALT = 3

    ITEM_INDEX_ALL = 255

    def __init__(self, item_type=None, item_index=None, item_state_index=None):
        self.item_type = item_type
        self.item_index = item_index
        self.item_state_index = item_state_index
