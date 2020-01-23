"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.vehicle_control.controller_commons import controller_commons


class EgoState(object):
    SPEED_MS = 0
    SPEED_KMH = 1
    SPEED_MPH = 2
    ACC_MS = 3
    POSITION_X = 4
    POSITION_Y = 5
    POSITION_Z = 6
    OTHER = 100

    def __init__(self):
        self.states_dict = {}
        self.shortcuts_dict = {}

    def set_state(self, state_name, init_value, state_type):
        self.states_dict[state_name] = init_value
        if state_type is not None:
            self.shortcuts_dict[state_type] = state_name

    def get_state(self, state_name):
        state_val = None
        if state_name in self.states_dict:
            state_val = self.states_dict[state_name]
        return state_val

    def get_speed_ms(self):
        if self.SPEED_MS in self.shortcuts_dict:
            name = self.shortcuts_dict[self.SPEED_MS]
            speed = self.get_state(name)
        elif self.SPEED_KMH in self.shortcuts_dict:
            name = self.shortcuts_dict[self.SPEED_KMH]
            speed_kmh = self.get_state(name)
            speed = controller_commons.kmh_to_ms(speed_kmh)
        else:
            speed = 0.0
        return speed
