"""Defines WebotsControllerParameter class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import copy


class WebotsControllerParameter(object):
    """WebotsControllerParameter class defines parameters used by the vehicle controller"""

    def __init__(self, vehicle_id=None, parameter_name='', parameter_data=[]):
        self.vehicle_id = vehicle_id
        self.parameter_name = copy.copy(parameter_name)
        self.parameter_data = copy.copy(parameter_data)

    def get_parameter_name(self):
        return self.parameter_name

    def set_parameter_name(self, parameter_name):
        self.parameter_name = copy.copy(parameter_name)

    def get_parameter_data(self):
        return self.parameter_data

    def set_parameter_data(self, parameter_data):
        self.parameter_data = copy.copy(parameter_data)

    def get_vehicle_id(self):
        return self.vehicle_id

    def set_vehicle_id(self, vehicle_id):
        self.vehicle_id = vehicle_id
