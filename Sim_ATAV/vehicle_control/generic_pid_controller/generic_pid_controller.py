"""Defines GenericPIDController class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class GenericPIDController(object):
    """GenericPIDController is a simple PID controller implementation."""
    def __init__(self, P=0.1, I=0.0, D=0.0):
        self.P_GAIN = P
        self.I_GAIN = I
        self.D_GAIN = D
        self.prev_err = None
        self.prev_f_value = None
        self.i_state = 0.0
        self.I_MAX = float("inf")
        self.I_MIN = -float("inf")
        self.MAX_OUTPUT_VALUE = 1.0
        self.MIN_OUTPUT_VALUE = -1.0

    def set_parameters(self, param_p, param_i, param_d):
        """Set controller parameters (P, I, D)"""
        self.P_GAIN = param_p
        self.I_GAIN = param_i
        self.D_GAIN = param_d

    def set_output_range(self, min_output, max_output):
        """Set control output range."""
        self.MIN_OUTPUT_VALUE = min_output
        self.MAX_OUTPUT_VALUE = max_output

    def set_integrator_value_range(self, i_min, i_max):
        """Set integrator value range."""
        self.I_MIN = i_min
        self.I_MAX = i_max

    def compute(self, err_input):
        """Compute control output."""
        if self.prev_err is None:
            self.prev_err = err_input
        self.i_state += err_input
        self.i_state = max(min(self.i_state, self.I_MAX), self.I_MIN)

        p_term = err_input * self.P_GAIN
        i_term = self.i_state * self.I_GAIN
        d_term = (err_input - self.prev_err) * self.D_GAIN
        control_output = p_term + i_term + d_term
        control_output = max(min(control_output, self.MAX_OUTPUT_VALUE), self.MIN_OUTPUT_VALUE)
        self.prev_err = err_input

        return control_output

    def compute_no_derivative_kick(self, err_input, f_value):
        """Compute control output with "No derivative kick" approach."""
        if self.prev_f_value is None:
            self.prev_f_value = f_value
        self.i_state += err_input
        self.i_state = max(min(self.i_state, self.I_MAX), self.I_MIN)

        p_term = err_input * self.P_GAIN
        i_term = self.i_state * self.I_GAIN
        d_term = (f_value - self.prev_f_value) * self.D_GAIN
        control_output = p_term + i_term - d_term
        control_output = max(min(control_output, self.MAX_OUTPUT_VALUE), self.MIN_OUTPUT_VALUE)
        self.prev_err = err_input
        self.prev_f_value = f_value

        return control_output
