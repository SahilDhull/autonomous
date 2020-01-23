"""Defines GenericStanleyController class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import math


class GenericStanleyController:
    """GenericStanleyController is a simple implementation of the steering controller used in the
    Stanley autonomous car developed by Stanford for DARPA challenge."""
    def __init__(self):
        self.k = 0.15
        self.k2 = 0.3
        self.k3 = 1.1
        self.MAX_OUTPUT_VALUE = 0.5
        self.MIN_OUTPUT_VALUE = -0.5

    def set_parameters(self, param_k, param_k2, param_k3):
        """Set parameters of the controller."""
        self.k = param_k
        self.k2 = param_k2
        self.k3 = param_k3

    def set_output_range(self, min_output, max_output):
        """Set min and max control outputs."""
        self.MIN_OUTPUT_VALUE = min_output
        self.MAX_OUTPUT_VALUE = max_output

    def compute(self, angle_err, distance_err, speed):
        """Compute steering control."""
        control_output = self.k2 * angle_err + math.atan(self.k * distance_err / (speed + self.k3))
        control_output = max(min(control_output, self.MAX_OUTPUT_VALUE), self.MIN_OUTPUT_VALUE)
        return control_output
