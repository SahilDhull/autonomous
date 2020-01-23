"""Defines TrialController class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from Sim_ATAV.vehicle_control.base_controller.base_controller import BaseCarController


class TrialController(BaseCarController):
    """This controller is for trying out new stuff."""
    def __init__(self, controller_parameters):
        (car_model, target_throttle) = controller_parameters
        BaseCarController.__init__(self, car_model)
        self.camera_device_name = "camera"
        self.camera = None
        self.target_throttle = float(target_throttle)
        print("TrialController Initialized: {}, {}".format(car_model, self.target_throttle))

    def run(self):
        """Run the controller."""
        # Start camera and the car engine:
        self.camera = self.getCamera(self.camera_device_name)
        if self.camera is not None:
            self.camera.enable(16)
        self.start_car()

        while True:
            self.step()

