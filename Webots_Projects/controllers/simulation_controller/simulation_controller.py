"""simulation_controller is the entry point for the supervisor controller in Webots.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import sys
import os
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_PATH + '/../../../')
from Sim_ATAV.simulation_control.sim_controller import SimController


if len(sys.argv) >= 1:
    try:
        PORT_NO = int(sys.argv[1])
    except ValueError:
        PORT_NO = 10020
else:
    PORT_NO = 10020

DEBUG_MODE = 0
SUPERVISOR = SimController()
SUPERVISOR.init(DEBUG_MODE, PORT_NO)
SUPERVISOR.run()
