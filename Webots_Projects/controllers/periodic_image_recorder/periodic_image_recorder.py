"""Defines periodic_image_recorder class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import datetime
import os
import sys
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
#LIBRARY_PATH = os.environ.get("WEBOTS_HOME") + "/projects/automobile/libraries/python"
try:
    LIBRARY_PATH = os.environ.get("WEBOTS_HOME") + "/lib/python36"
except:
    LIBRARY_PATH = 'C:/Program Files/Webots/lib/python36'
LIBRARY_PATH.replace('/', os.sep)
sys.path.append(LIBRARY_PATH)
from vehicle import Car


class PeriodicImageRecorder(Car):
    """PeriodicImageRecorder class is a car controller class for Webots.
    It is used for reading camera image periodically to check deterministic behavior in Webots."""

    def __init__(self):
        self.camera_device_name = "camera"
        self.camera = None
        Car.__init__(self)

    def run(self):
        """Runs the controller."""
        # Start camera and the car engine:
        self.camera = self.getCamera(self.camera_device_name)
        if self.camera is not None:
            self.camera.enable(10)
        self.setThrottle(0.0)
        self.setGear(1)
        gear = self.getGear()

        # Open file to save images
        now = datetime.datetime.now()
        record_file_name = "record_{}_{}_{}.txt".format(now.hour, now.minute, now.second)
        record_file = open(record_file_name, "wb", 0)

        print("Here we roll!")
        vehicle_alive = True
        step_counter = 0
        outer_counter = 0
        while vehicle_alive:
            self.step()
            step_counter += 1
            self.setSteeringAngle(0.0)

            if step_counter > 95:
                self.setThrottle(0.0)
                self.setBrakeIntensity(1.0)
            else:
                self.setThrottle(1.0)
                self.setBrakeIntensity(0.0)
            if step_counter == 100:
                step_counter = 0
                outer_counter += 1
                # Read image.
                image = self.camera.getImage()
                #print(type(image))
                record_file.write(b"--------IMAGE -----------\n")
                record_file.write(bytearray(image))
            if outer_counter == 10:
                vehicle_alive = False
        record_file.close()
        print("Now, Please revert the world, run the simulation once more, and compare the generated txt files which contain the images from camera.")
        sys.stdout.flush()


def main():
    """For running the controller directly from Webots."""
    controller = PeriodicImageRecorder()
    controller.run()


if __name__ == "__main__":
    main()
