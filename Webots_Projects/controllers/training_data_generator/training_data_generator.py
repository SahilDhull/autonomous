"""Defines training_data_generator controller
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
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_PATH + "/../../../")
#LIBRARY_PATH = os.environ.get("WEBOTS_HOME") + "/projects/automobile/libraries/python"
try:
    LIBRARY_PATH = os.environ.get("WEBOTS_HOME") + "/lib/python36"
except:
    LIBRARY_PATH = 'C:/Program Files/Webots/lib/python36'
LIBRARY_PATH.replace('/', os.sep)
sys.path.append(LIBRARY_PATH)
from Sim_ATAV.vehicle_control.training_data_generator.training_data_generator import TrainingDataGenerator


def main():
    """For running the controller directly from Webots
    without passing it as a parameter to vehicle_controller"""
    controller = TrainingDataGenerator(sys.argv)
    controller.run()


if __name__ == "__main__":
    main()
