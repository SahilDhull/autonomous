"""Defines WebotsVehicle Class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

from Sim_ATAV.common.coordinate_system import CoordinateSystem


class WebotsVehicle(object):
    """WebotsVehicle class defines user configurable vehicle to use in Webots environment"""

    CITROEN_C_ZERO = 201
    CITROEN_C_ZERO_SIMPLE = 202
    TOYOTA_PRIUS = 203
    TOYOTA_PRIUS_SIMPLE = 204
    BMW_X5 = 205
    BMW_X5_SIMPLE = 206
    BUS = 207
    LINCOLN_MKZ = 209
    LINCOLN_MKZ_SIMPLE = 210
    RANGE_ROVER = 211
    RANGE_ROVER_SIMPLE = 212
    ACKERMANN_VEHICLE = 255

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
    STATE_ID_ORIENTATION = 15
    STATE_ID_JERK = 16
    STATE_ID_ANGULAR_VELOCITY_X = 17
    STATE_ID_ANGULAR_VELOCITY_Y = 18
    STATE_ID_ANGULAR_VELOCITY_Z = 19
    STATE_ID_YAW_RATE = 20

    DUMMY = 0
    VUT = 1

    def __init__(self):
        self.node = None
        self.translation = None
        self.name = None
        self.vhc_id = 0
        self.def_name = ""
        self.vehicle_model = "AckermannVehicle"
        self.vehicle_model_id = self.ACKERMANN_VEHICLE
        self.color = [1, 1, 1]
        self.controller = "void"
        self.is_controller_name_absolute = False
        self.vehicle_parameters = []
        self.controller_parameters = []
        self.controller_arguments = []
        self.sensor_array = []
        self.signal = []
        self.half_width = 0.0
        self.rear_axis_to_rear_length = 0.0
        self.rear_axis_to_front_length = 0.0
        self.height_from_axis = 0.0
        # States ---
        self.current_position = [0, 0.3, 0]
        self.current_orientation = 0.0
        self.previous_orientation = None
        self.rotation = [0, 1, 0, 0]
        self.angular_velocity_3d = None
        self.current_velocity = None
        self.previous_velocity = None
        self.yaw_rate = 0.0
        self.speed = 0
        self.acceleration_3d = None
        self.previous_acceleration_3d = None
        self.acceleration = 0.0
        self.jerk_3d = None
        self.jerk = 0.0
        self.state_record_time = 0.0
        # ---

    def set_vehicle_model(self, vhc_model_name):
        """Sets the model of the vehicle per the given vhc_model_name"""
        self.vehicle_model = vhc_model_name[:]
        self.set_vehicle_dimensions()
        if 'Citroen' in self.vehicle_model:
            self.vehicle_model_id = self.CITROEN_C_ZERO
        elif 'Bmw' in self.vehicle_model:
            self.vehicle_model_id = self.BMW_X5
        elif 'Toyota' in self.vehicle_model:
            self.vehicle_model_id = self.TOYOTA_PRIUS
        elif 'Lincoln' in self.vehicle_model:
            self.vehicle_model_id = self.LINCOLN_MKZ
        elif 'RangeRover' in self.vehicle_model:
            self.vehicle_model_id = self.RANGE_ROVER
        elif 'Bus' in self.vehicle_model:
            self.vehicle_model_id = self.BUS
        else:
            self.vehicle_model_id = self.ACKERMANN_VEHICLE

    def get_vehicle_box_corners(self):
        """Get the vehicle bounding box corners."""
        return self.get_vehicle_box_corners_by_model()

    def set_vehicle_dimensions(self):
        """Set dimensions of the vehicle using the model name."""
        if 'Citroen' in self.vehicle_model:
            self.half_width = 0.75
            self.rear_axis_to_rear_length = 0.35
            self.rear_axis_to_front_length = 3.0
            self.height_from_axis = 1.35
        elif 'Bmw' in self.vehicle_model:
            self.half_width = 0.865
            self.rear_axis_to_rear_length = 0.9
            self.rear_axis_to_front_length = 3.7
            self.height_from_axis = 1.45
        elif 'Toyota' in self.vehicle_model:
            self.half_width = 0.85
            self.rear_axis_to_rear_length = 0.7
            self.rear_axis_to_front_length = 3.6
            self.height_from_axis = 1.3
        elif 'Lincoln' in self.vehicle_model:
            self.half_width = 0.9
            self.rear_axis_to_rear_length = 1.03
            self.rear_axis_to_front_length = 3.87
            self.height_from_axis = 1.16
        elif 'RangeRover' in self.vehicle_model:
            self.half_width = 0.9
            self.rear_axis_to_rear_length = 1.0
            self.rear_axis_to_front_length = 3.5
            self.height_from_axis = 1.3
        elif 'Bus' in self.vehicle_model:
            self.half_width = 1.37
            self.rear_axis_to_rear_length = 6.2
            self.rear_axis_to_front_length = 11.5
            self.height_from_axis = 2.6
        else:
            self.half_width = 0.9
            self.rear_axis_to_rear_length = 0.7
            self.rear_axis_to_front_length = 3.6
            self.height_from_axis = 1.3

    def get_vehicle_box_corners_by_model(self):
        """Returns the corner points of a box bounding the vehicle shape."""
        pts = {}
        if self.half_width == 0.0 or self.rear_axis_to_front_length == 0.0 or self.rear_axis_to_rear_length == 0.0:
            self.set_vehicle_dimensions()
        # Vehicles 0 point is the midpoint of the rear axle.
        # Axises: [x, y, z]
        # x: towards right of the screen,
        # y: from ground towards air,
        # z: towards bottom of the screen
        # When vehicle is not rotated:
        # +ve x is towards vehicle's left,
        # +ve y is towards vehicle's top,
        # +ve z is towards vehicle's front
        len_bottom = -0.3
        pts[0] = [-self.half_width, len_bottom, self.rear_axis_to_front_length]  # front-right-bottom corner
        pts[1] = [-self.half_width, self.height_from_axis, self.rear_axis_to_front_length]  # front-right-top corner
        pts[2] = [self.half_width, len_bottom, self.rear_axis_to_front_length]  # front-left-bottom corner
        pts[3] = [self.half_width, self.height_from_axis, self.rear_axis_to_front_length]  # front-left-top corner
        pts[4] = [-self.half_width, len_bottom, -self.rear_axis_to_rear_length]  # rear-right-bottom corner
        pts[5] = [-self.half_width, self.height_from_axis, -self.rear_axis_to_rear_length]  # rear-right-top corner
        pts[6] = [self.half_width, len_bottom, -self.rear_axis_to_rear_length]  # rear-left-bottom corner
        pts[7] = [self.half_width, self.height_from_axis, -self.rear_axis_to_rear_length]  # rear-left-top corner
        return pts

    def get_vehicle_critical_points_by_model(self, vehicle_model_id):
        """Returns the critical points defining the vehicle shape
           according to the given vehicle_model_id"""
        pts = {}
        # Vehicles 0 point is the midpoint of the rear axle.
        # Axises: [x, y, z]
        # x: towards right of the screen,
        # y: from ground towards air,
        # z: towards bottom of the screen
        # Vehicle's non-rotated positions: (front sides are looking towards bottom of the screen)
        if vehicle_model_id == self.CITROEN_C_ZERO or \
           vehicle_model_id == self.CITROEN_C_ZERO_SIMPLE:
            pts[0] = [-0.74, 0.05, 3.075]  # front-right corner
            pts[1] = [-0.74, 0.05, -0.425]  # rear-right corner
            pts[2] = [0.74, 0.05, 3.075]  # front-left corner
            pts[3] = [0.74, 0.05, -0.425]  # rear-left corner
            pts[4] = [0, 0.05, -0.425]  # Mid rear
            pts[5] = [0, 0.05, 3.075]  # Mid front
            pts[6] = [-0.74, 0.05, 1.325]  # Mid right
            pts[7] = [0.74, 0.05, 1.325]  # Mid left
        elif vehicle_model_id == self.TOYOTA_PRIUS or vehicle_model_id == self.TOYOTA_PRIUS_SIMPLE:
            pts[0] = [-0.875, 0.05, 3.635]  # front-right corner
            pts[1] = [-0.875, 0.05, -0.85]  # rear-right corner
            pts[2] = [0.875, 0.05, 3.635]  # front-left corner
            pts[3] = [0.875, 0.05, -0.85]  # rear-left corner
            pts[4] = [0, 0.05, -0.85]  # Mid rear
            pts[5] = [0, 0.05, 3.635]  # Mid front
            pts[6] = [-0.875, 0.05, 1.3925]  # Mid right
            pts[7] = [0.875, 0.05, 1.3925]  # Mid left
        elif vehicle_model_id == self.BMW_X5 or vehicle_model_id == self.BMW_X5_SIMPLE:
            pts[0] = [-1.1, 0.05, 3.85]  # front-right corner
            pts[1] = [-1.1, 0.05, -1.01]  # rear-right corner
            pts[2] = [1.1, 0.05, 3.85]  # front-left corner
            pts[3] = [1.1, 0.05, -1.01]  # rear-left corner
            pts[4] = [0, 0.05, -1.01]  # Mid rear
            pts[5] = [0, 0.05, 3.85]  # Mid front
            pts[6] = [-1.1, 0.05, 1.425]  # Mid right
            pts[7] = [1.1, 0.05, 1.425]  # Mid left
        elif vehicle_model_id == self.BUS:
            pts[0] = [-1.37, 0.05, 11.5]  # front-right corner
            pts[1] = [-1.37, 0.05, -6.2]  # rear-right corner
            pts[2] = [1.37, 0.05, 11.5]  # front-left corner
            pts[3] = [1.37, 0.05, -6.2]  # rear-left corner
            pts[4] = [0, 0.05, -6.2]  # Mid rear
            pts[5] = [0, 0.05, 11.5]  # Mid front
            pts[6] = [-1.37, 0.05, 2.65]  # Mid right
            pts[7] = [1.37, 0.05, 2.65]  # Mid left
        else:  # ACKERMANN_VEHICLE
            pts[0] = [-0.95, 0.05, 4.4]  # front-right corner
            pts[1] = [-0.95, 0.05, -0.4]  # rear-right corner
            pts[2] = [0.95, 0.05, 4.4]  # front-left corner
            pts[3] = [0.95, 0.05, -0.4]  # rear-left corner
            pts[4] = [0, 0.05, -0.4]  # Mid rear
            pts[5] = [0, 0.05, 4.4]  # Mid front
            pts[6] = [-0.95, 0.05, 2.0]  # Mid right
            pts[7] = [0.95, 0.05, 2.0]  # Mid left
        return pts

    def get_vehicle_critical_points(self):
        """Returns the critical points defining the vehicle shape for the current vehicle"""
        pts = self.get_vehicle_critical_points_by_model(self.vehicle_model_id)
        return pts

    def get_vehicle_state_with_id(self, state_id):
        """Returns the value of the current state which corresponds to the given state_id"""
        state_value = 0.0
        if state_id == self.STATE_ID_POSITION_X:
            state_value = self.current_position[CoordinateSystem.X_AXIS]
        elif state_id == self.STATE_ID_POSITION_Y:
            state_value = self.current_position[CoordinateSystem.Y_AXIS]
        elif state_id == self.STATE_ID_POSITION_Z:
            state_value = self.current_position[CoordinateSystem.Z_AXIS]
        elif state_id == self.STATE_ID_VELOCITY_X:
            state_value = self.current_velocity[CoordinateSystem.X_AXIS] if self.current_velocity is not None else 0.0
        elif state_id == self.STATE_ID_VELOCITY_Y:
            state_value = self.current_velocity[CoordinateSystem.Y_AXIS] if self.current_velocity is not None else 0.0
        elif state_id == self.STATE_ID_VELOCITY_Z:
            state_value = self.current_velocity[CoordinateSystem.Z_AXIS] if self.current_velocity is not None else 0.0
        elif state_id == self.STATE_ID_SPEED:
            state_value = self.speed
        elif state_id == self.STATE_ID_ACCELERATION_X:
            state_value = self.acceleration_3d[CoordinateSystem.X_AXIS] if self.acceleration_3d is not None else 0.0
        elif state_id == self.STATE_ID_ACCELERATION_Y:
            state_value = self.acceleration_3d[CoordinateSystem.Y_AXIS] if self.acceleration_3d is not None else 0.0
        elif state_id == self.STATE_ID_ACCELERATION_Z:
            state_value = self.acceleration_3d[CoordinateSystem.Z_AXIS] if self.acceleration_3d is not None else 0.0
        elif state_id == self.STATE_ID_ACCELERATION:
            state_value = self.acceleration
        elif state_id == self.STATE_ID_JERK_X:
            state_value = self.jerk_3d[CoordinateSystem.X_AXIS] if self.jerk_3d is not None else 0.0
        elif state_id == self.STATE_ID_JERK_Y:
            state_value = self.jerk_3d[CoordinateSystem.Y_AXIS] if self.jerk_3d is not None else 0.0
        elif state_id == self.STATE_ID_JERK_Z:
            state_value = self.jerk_3d[CoordinateSystem.Z_AXIS] if self.jerk_3d is not None else 0.0
        elif state_id == self.STATE_ID_JERK:
            state_value = self.jerk
        elif state_id == self.STATE_ID_ORIENTATION:
            state_value = self.current_orientation
        elif state_id == self.STATE_ID_ANGULAR_VELOCITY_X:
            state_value = self.angular_velocity_3d[CoordinateSystem.X_AXIS] if self.angular_velocity_3d is not None \
                else 0.0
        elif state_id == self.STATE_ID_ANGULAR_VELOCITY_Y:
            state_value = self.angular_velocity_3d[CoordinateSystem.Y_AXIS] if self.angular_velocity_3d is not None \
                else 0.0
        elif state_id == self.STATE_ID_ANGULAR_VELOCITY_Z:
            state_value = self.angular_velocity_3d[CoordinateSystem.Z_AXIS] if self.angular_velocity_3d is not None \
                else 0.0
        elif state_id == self.STATE_ID_YAW_RATE:
            state_value = self.yaw_rate
        return state_value
