"""Defines WebotsSensorField and WebotsSensor classes
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class WebotsSensorField(object):
    """WebotsSensorField Class holds sensor field name and value for each setting of a sensor"""
    def __init__(self):
        self.field_name = ""
        self.field_val = ""


class WebotsSensor(object):
    """WebotsSensor Class defines a sensor to be used in Webots environment"""
    FRONT = 0
    CENTER = 1
    LEFT = 2
    RIGHT = 3
    TOP = 4
    LEFT_FRONT = 5
    LEFT_REAR = 6
    RIGHT_FRONT = 7
    RIGHT_REAR = 8
    REAR = 9

    def __init__(self):
        self.sensor_type = ""
        self.sensor_location = 0
        self.sensor_fields = []

    def add_sensor_field(self, field_name, field_val):
        """Adds a new sensor field (a parameter for a sensor) to the current sensor"""
        self.sensor_fields.append(WebotsSensorField())
        self.sensor_fields[-1].field_name = field_name
        self.sensor_fields[-1].field_val = field_val
