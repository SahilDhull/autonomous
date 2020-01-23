"""Defines BaseCarController class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import math
import sys


try:
    from vehicle import Car
except ImportError:
    sys.stderr.write("WARNING! 'Car' module from Webots 'vehicle' could not be imported.\n")
    sys.exit(0)


# ********************************************************************************
# This controller is the base vehicle controller
# ********************************************************************************
class BaseCarController(Car):
    """BaseCarController is the base class for a vehicle controller. It has basic and common
    definitions for a car, and it provides wrapper functions for Webots API"""

    # Vehicle control modes:
    CONTROL_MODE_SPEED = 0
    CONTROL_MODE_TORQUE = 1

    # Vehicle engine types:
    ENG_TYPE_COMBUSTION = 0
    ENG_TYPE_ELECTRIC = 1
    ENG_TYPE_PARALLEL_HYBRID = 2
    ENG_TYPE_SPLIT_HYBRID = 3

    def __init__(self, car_model):
        self.debug_mode = False
        print('BaseController: Car Model: {}'.format(car_model))
        Car.__init__(self)
        self.car_model = car_model
        self.set_car_specific_parameters()
        if self.debug_mode:
            print("BaseCarController: initialized")

    def set_car_specific_parameters(self):
        """Sets some vehicle specific parameters that can be used by the controllers."""
        if 'Citroen' in self.car_model:
            self.engine_type = 'e'  # 'e'lectric, 'c'ombustion, 'h'ybrid
            self.engine_split_ratio = 0.0
            self.engine_split_rpm = 0.0
            self.engine_param_a = 0.0
            self.engine_param_b = 0.0
            self.engine_param_c = 0.0
            self.engine_min_rpm = 0.0
            self.engine_max_rpm = 8000.0
            self.tire_radius = 0.27
            self.gear_ratios = [6.2329]
            self.engine_max_torque = 150.0
            self.engine_max_power = 75000.0
            self.brake_coefficient = 500.0
            self.mass = 1200.0
            self.length = 3.5
            self.wheel_base_length = 2.6
            self.length_front = self.wheel_base_length / 2
            self.length_rear = self.wheel_base_length - self.length_front
            self.inertia = 2534.0
            self.tire_stiffness = 25000.0  # 867.0 #49675.5
            if self.debug_mode:
                print("BaseCarController: Citroen (CZero) Settings applied.")
        elif 'Toyota' in self.car_model:
            self.engine_type = 'h'  # 'e'lectric, 'c'ombustion, 'h'ybrid
            self.engine_split_ratio = 0.2778
            self.engine_split_rpm = 3000.0
            self.engine_param_a = 65.0
            self.engine_param_b = 0.0225
            self.engine_param_c = -0.0000025
            self.engine_min_rpm = 1200.0
            self.engine_max_rpm = 6500.0
            self.tire_radius = 0.27
            self.gear_ratios = [6.0]
            self.engine_max_torque = 350.0
            self.engine_max_power = 33000.0
            self.brake_coefficient = 500.0
            self.mass = 1805.0
            self.length = 4.0
            self.wheel_base_length = 3.2
            self.length_front = self.wheel_base_length / 2
            self.length_rear = self.wheel_base_length - self.length_front
            self.inertia = 3400.0
            self.tire_stiffness = 25000.0
            if self.debug_mode:
                print("BaseCarController: Toyota (Prius) Settings applied.")
        else:
            self.engine_type = 'c'  # 'e'lectric, 'c'ombustion, 'h'ybrid
            self.engine_split_ratio = 0.0
            self.engine_split_rpm = 0.0
            self.engine_param_a = 150.0
            self.engine_param_b = 0.1
            self.engine_param_c = 0.0
            self.engine_min_rpm = 1000.0
            self.engine_max_rpm = 4000.0
            self.tire_radius = 0.27
            self.gear_ratios = [10.0, 7.0, 5.0, 2.5, 1.0]
            self.engine_max_torque = 250.0
            self.engine_max_power = 50000.0
            self.brake_coefficient = 500.0
            self.mass = 2000.0
            self.length = 5.0
            self.wheel_base_length = 3.0
            self.length_front = self.wheel_base_length / 2
            self.length_rear = self.wheel_base_length - self.length_front
            self.inertia = 4346.1
            self.tire_stiffness = 25000.0
            if self.debug_mode:
                print("BaseCarController: BMW (X5) Settings applied.")

    def get_torque(self):
        """Compute and return current torque."""
        torque = 0.0
        if self.getControlMode() == self.CONTROL_MODE_TORQUE:
            rpm = self.getRpm()
            if (self.engine_type == 'e') or (self.engine_type == 'h'):
                if rpm > 1.0:
                    torque = min(self.engine_max_torque,\
                        self.engine_max_power*60.0/(2.0*math.pi*rpm))
                else:
                    torque = float(self.engine_max_torque)
            if self.engine_type == 'c':
                if rpm < self.engine_min_rpm:
                    rpm = self.engine_min_rpm
                if rpm > self.engine_max_rpm:
                    torque = 0.0
                else:
                    torque = self.engine_param_c*(rpm**2) +\
                             self.engine_param_b*rpm +\
                             self.engine_param_a
            if self.engine_type == 'h':
                if rpm < self.engine_min_rpm:
                    torque_c = 0.0
                else:
                    torque_c = self.engine_param_c*(self.engine_split_rpm**2) +\
                               self.engine_param_b*self.engine_split_rpm +\
                               self.engine_param_a
                torque = torque + (1.0-self.engine_split_ratio)*torque_c
        return torque

    def set_throttle_and_steering_angle(self, throttle, angle):
        """Set the vehicle throttle and steering angle.
        Between -1 and 1 for both throttle and steering angle.
        Applies brakes if throttle is less than 0.
        Applicable only when control mode is CONTROL_MODE_TORQUE."""
        self.setSteeringAngle(angle)
        if throttle >= 0.0:
            throttle_setting = min(throttle, 1.0)
            self.setThrottle(throttle_setting)
            self.setBrakeIntensity(0.0)
        else:
            self.setThrottle(0.0)
            self.setBrakeIntensity(min(-throttle, 1.0))

    def set_target_speed_and_angle(self, speed, angle):
        """Set the vehicle target speed and steering angle.
        Between -1 and 1 for steering angle.
        Applicable only when control mode is CONTROL_MODE_SPEED."""
        self.setSteeringAngle(angle)
        self.setCruisingSpeed(speed)

    def set_gear(self, gear):
        """Change gear."""
        self.setGear(gear)

    def set_throttle(self, throttle):
        """Set throttle pedal position (between 0 and 1)."""
        self.setThrottle(throttle)

    def get_gear(self):
        """Get current gear."""
        return self.getGear()

    def get_steering_angle(self):
        """Get current steering angle (between -1 and 1)."""
        return self.getSteeringAngle()

    def get_current_speed(self):
        """Get current speed."""
        return self.getCurrentSpeed()

    def get_throttle(self):
        """Set current throttle pedal position (between 0 and 1)."""
        return self.getThrottle()

    def get_brake(self):
        """Get current brake intensity."""
        return self.getBrakeIntensity()

    def set_brake(self, brake_intensity):
        """Set brake intensity"""
        self.setBrakeIntensity(brake_intensity)

    def get_rpm(self):
        """Get current engine rpm.
        Applicable only when control mode is CONTROL_MODE_TORQUE."""
        if self.getControlMode() == self.CONTROL_MODE_TORQUE:
            ret_val = self.getRpm()
        else:
            ret_val = 0.0
        return ret_val

    def get_control_mode(self):
        """Get current control mode (CONTROL_MODE_TORQUE or CONTROL_MODE_SPEED)."""
        return self.getControlMode()

    def get_sim_time(self):
        """Get current simulation time."""
        return self.getTime()

    def start_car(self):
        """Puts the car into first gear and sets throttle pedal to 0."""
        self.setThrottle(0.0)
        self.setGear(1)
        gear = self.getGear()
        if self.debug_mode:
            print("Starting CAR: Gear: {}".format(gear))

    def run(self):
        """Runs the controller.
        This controller runs the car with full throttle."""
        self.start_car()
        self.setBrakeIntensity(0.0)

        while self.step() >= 0:
            self.setThrottle(1.0)
