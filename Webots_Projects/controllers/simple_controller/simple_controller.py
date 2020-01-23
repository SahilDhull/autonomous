"""Defines simple_controller controller
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
from Sim_ATAV.vehicle_control.simple_controller.simple_controller import SimpleController


def run(input_args):
    """For running controller from "vehicle_controller"."""
    controller = SimpleController(input_args)
    controller.run()


def main():
    """For running the controller directly from Webots
    without passing it as a parameter to vehicle_controller"""
    controller = SimpleController(sys.argv)
    controller.run()


if __name__ == "__main__":
    main()
