"""Defines SimulationCommand class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class SimulationCommand(object):
    """SimulationCommand class is a structure
       keeping control command and the related object received."""
    def __init__(self, command, command_object):
        self.command = command
        self.object = command_object
