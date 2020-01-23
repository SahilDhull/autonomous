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

#import Robot

from controller import Supervisor


class Driver (Supervisor):
    def run(self):
        while True:
            if self.step(10) == -1:
                break



controller = Driver()
controller.run()

