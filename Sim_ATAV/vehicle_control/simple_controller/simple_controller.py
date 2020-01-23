"""Defines SimpleController class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from Sim_ATAV.vehicle_control.base_controller.base_controller import BaseCarController


class SimpleController(BaseCarController):
    """SimpleController class is a car controller class for Webots."""
    def __init__(self, controller_parameters):
        (car_model, target_throttle) = controller_parameters
        BaseCarController.__init__(self, car_model)
        self.target_throttle = float(target_throttle)
        print("SimpleController Initialized: {}, {}".format(car_model, self.target_throttle))

    def run(self):
        """Runs the controller.
        This controller is a simple controller for vehicles.
        Drives the car straight with given target throttle"""
        self.start_car()
        while True:
            self.step()
            self.set_throttle_and_steering_angle(self.target_throttle, 0.0)
