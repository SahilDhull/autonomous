"""Defines WebotsPedestrian Class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import math
from Sim_ATAV.common.coordinate_system import CoordinateSystem


class WebotsPedestrian(object):
    """User Configurable Pedestrian Structure to use in Webots environment"""

    STATE_ID_VELOCITY_X = 1
    STATE_ID_VELOCITY_Y = 2
    STATE_ID_VELOCITY_Z = 3
    STATE_ID_ACCELERATION_X = 4
    STATE_ID_ACCELERATION_Y = 5
    STATE_ID_ACCELERATION_Z = 6
    STATE_ID_JERK_X = 7
    STATE_ID_JERK_Y = 8
    STATE_ID_JERK_Z = 9
    STATE_ID_ACCELERATION = 10
    STATE_ID_SPEED = 11
    STATE_ID_POSITION_X = 12
    STATE_ID_POSITION_Y = 13
    STATE_ID_POSITION_Z = 14

    def __init__(self):
        self.def_name = "PEDESTRIAN"
        self.node = None
        self.translation = None
        self.name = None
        self.ped_id = 0
        self.current_orientation = math.pi/2.0
        self.rotation = [0, 1, 0, math.pi/2.0]
        self.current_position = [0, 0, 0]
        self.shirt_color = [0.25, 0.55, 0.2]
        self.pants_color = [0.24, 0.25, 0.5]
        self.shoes_color = [0.28, 0.15, 0.06]
        self.controller = "void"
        self.trajectory = []
        self.target_speed = 0.0
        self.current_velocity = [0, 0, 0]
        self.speed = 0
        self.controller_parameters = []
        self.controller_arguments = []
        self.signal = []
        self.previous_position = None
        self.previous_velocity = 0.0

    def get_pedestrian_box_corners(self):
        pts = {}
        if self.target_speed > 0.1:  # If walking, front and rear distances will vary, but let's take average
            LEN_FRONT = 0.45
            LEN_REAR = -0.35
        else:
            LEN_FRONT = 0.2
            LEN_REAR = -0.2
        LEN_LEFT = 0.2
        LEN_RIGHT = -0.2
        LEN_TOP = 0.53
        LEN_BOTTOM = -1.27
        pts[0] = [LEN_RIGHT, LEN_BOTTOM, LEN_FRONT]  # front-right-bottom corner
        pts[1] = [LEN_RIGHT, LEN_TOP, LEN_FRONT]  # front-right-top corner
        pts[2] = [LEN_LEFT, LEN_BOTTOM, LEN_FRONT]  # front-left-bottom corner
        pts[3] = [LEN_LEFT, LEN_TOP, LEN_FRONT]  # front-left-top corner
        pts[4] = [LEN_RIGHT, LEN_BOTTOM, LEN_REAR]  # rear-right-bottom corner
        pts[5] = [LEN_RIGHT, LEN_TOP, LEN_REAR]  # rear-right-top corner
        pts[6] = [LEN_LEFT, LEN_BOTTOM, LEN_REAR]  # rear-left-bottom corner
        pts[7] = [LEN_LEFT, LEN_TOP, LEN_REAR]  # rear-left-top corner
        return pts

    def get_pedestrian_state_with_id(self, state_id):
        """Returns the value of the current state which corresponds to the given state_id"""
        state_value = 0.0
        if state_id == self.STATE_ID_VELOCITY_X:
            state_value = self.current_velocity[CoordinateSystem.X_AXIS]
        elif state_id == self.STATE_ID_VELOCITY_Y:
            state_value = self.current_velocity[CoordinateSystem.Y_AXIS]
        elif state_id == self.STATE_ID_VELOCITY_Z:
            state_value = self.current_velocity[CoordinateSystem.Z_AXIS]
        elif state_id == self.STATE_ID_SPEED:
            state_value = self.speed
        elif state_id == self.STATE_ID_POSITION_X:
            state_value = self.current_position[CoordinateSystem.X_AXIS]
        elif state_id == self.STATE_ID_POSITION_Y:
            state_value = self.current_position[CoordinateSystem.Y_AXIS]
        elif state_id == self.STATE_ID_POSITION_Z:
            state_value = self.current_position[CoordinateSystem.Z_AXIS]
        return state_value
