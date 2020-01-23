"""Defines STaliroSignal class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class STaliroSignal(object):
    """STaliroSignal class defines signal properties received from STaliro. 
       Control Point arrays of the signal etc.
       See STaliro documentation for a better understanding."""
    SIGNAL_TYPE_SPEED = 0
    SIGNAL_TYPE_X_POSITION = 1
    SIGNAL_TYPE_Y_POSITION = 2
    SIGNAL_TYPE_Z_POSITION = 3
    INTERPOLATION_TYPE_NONE = -1
    INTERPOLATION_TYPE_PIECEWISE_CONST = 0
    INTERPOLATION_TYPE_LINEAR = 1

    def __init__(self, signal_type, interpolation_type, 
                 ref_index, ref_field, signal_values, ref_values):
        self.signal_type = signal_type
        self.ref_index = ref_index
        self.ref_field = ref_field
        self.signal_values = signal_values
        self.ref_values = ref_values
        self.interpolation_type = interpolation_type

    def get_signal_value_corresponding_to_value_of_reference(self,
                                                             reference_value,
                                                             interpolation_type):
        """get_signal_value_corresponding_to_value_of_reference function returns signal value
         based on the current value of the reference signal and the interpolation type"""
        if interpolation_type == self.INTERPOLATION_TYPE_NONE:
            interpolation_type = self.interpolation_type
        # Return corresponding signal value to the given reference value
        smaller_ref_index = 0
        larger_ref_index = len(self.ref_values) - 1
        for i in range(len(self.ref_values)):
            if self.ref_values[i] <= reference_value:
                smaller_ref_index = i
            elif self.ref_values[i] > reference_value:
                larger_ref_index = i
                break
        if interpolation_type == self.INTERPOLATION_TYPE_PIECEWISE_CONST:
            ret_val = self.signal_values[smaller_ref_index]
        else:
            if self.ref_values[larger_ref_index] - self.ref_values[smaller_ref_index] == 0:
                ret_val = self.signal_values[smaller_ref_index]
            else:
                ratio = ((reference_value - self.ref_values[smaller_ref_index])
                         / (self.ref_values[larger_ref_index] - self.ref_values[smaller_ref_index]))
                ret_val = (self.signal_values[smaller_ref_index]
                           + ratio*(self.signal_values[larger_ref_index]
                                    - self.signal_values[smaller_ref_index]))
        return ret_val
