"""Defines calibration_recorder controller
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import cv2
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
from Sim_ATAV.vehicle_control.base_controller.base_controller import BaseCarController
sys.path.append(FILE_PATH + "/../../../")
#LIBRARY_PATH = os.environ.get("WEBOTS_HOME") + "/projects/automobile/libraries/python"
try:
    LIBRARY_PATH = os.environ.get("WEBOTS_HOME") + "/lib/python36"
except:
    LIBRARY_PATH = 'C:/Program Files/Webots/lib/python36'
LIBRARY_PATH.replace('/', os.sep)
sys.path.append(LIBRARY_PATH)


# **********************************************************************************************
# This controller only takes a snapshot of the scene and saves to file for calibration purposes
# **********************************************************************************************
class CalibrationRecorder(BaseCarController):
    """CalibrationRecorder class is a car controller class for Webots.
    This controller is used to record camera images for distance calculation calibration."""
    def __init__(self, controller_parameters):
        (car_model, calibration_id) = controller_parameters
        BaseCarController.__init__(self, car_model)
        self.camera_name = "camera"
        self.camera = None
        self.calibration_id = calibration_id
        print("CalibrationRecorder Initialized: {}, id: {}".format(car_model, self.calibration_id))

    def run(self):
        """Runs the Controller.
        Only takes a snapshot of the scene and saves to file for calibration purposes.
        """
        # Start camera and the car engine:
        self.camera = self.getCamera(self.camera_name)
        if self.camera is not None:
            self.camera.enable(16)
        self.start_car()
        self.step()
        image_array_rgb = self.camera.getImageArray()
        image_array_rgb = np.array(image_array_rgb)
        image_array_rgb = np.rot90(image_array_rgb, -1)
        image_array_rgb = np.fliplr(image_array_rgb)
        image_array_bgr = image_array_rgb[..., ::-1]

        file_name = "..\\..\\calibration_images\\cal_" + \
            self.car_model + "_" + self.calibration_id + ".png"
        cv2.imwrite(file_name, image_array_bgr)
        print('Camera image saved to {}'.format(file_name))
        self.step()


def main():
    """For running the controller directly from Webots
    without passing it as a parameter to vehicle_controller"""
    controller = CalibrationRecorder(sys.argv)
    controller.run()


if __name__ == "__main__":
    main()
