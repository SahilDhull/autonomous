"""Defines DataLogger class.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import numpy as np
from Sim_ATAV.simulation_control.item_description import ItemDescription


class DataLogger(object):
    """DataLogger class handles logs the requested data."""
    def __init__(self):
        self.log = []
        self.data_to_log = []
        self.vehicles_manager = None
        self.environment_manager = None
        self.pedestrians_manager = None
        self.current_log_size = 0
        self.simulation_time_ms = 1000
        self.simulation_step_size_ms = 10
        self.temp_current_log = None
        self.log_period_ms = 1

    def set_vehicles_manager(self, vehicles_manager):
        """Sets the VehiclesManager that this class will use to log vehicle data."""
        self.vehicles_manager = vehicles_manager

    def set_environment_manager(self, environment_manager):
        """Sets the EnvironmentManager that this class will use to log vehicle data."""
        self.environment_manager = environment_manager

    def set_pedestrians_manager(self, pedestrians_manager):
        """Sets the pedestrians_manager that this class will use to log vehicle data."""
        self.pedestrians_manager = pedestrians_manager

    def add_data_log_description(self, data_log_description):
        """Adds description of data to log."""
        self.data_to_log.append(data_log_description)

    def set_expected_simulation_time(self, simulation_time_ms):
        """Sets the expected total simulation time in ms.
        This is used to compute size of the pre-allocated log space."""
        self.simulation_time_ms = simulation_time_ms

    def set_simulation_step_size(self, simulation_step_size_ms):
        """Sets the simulation step size in ms
        This is used to compute size of the pre-allocated log space."""
        self.simulation_step_size_ms = simulation_step_size_ms

    def set_log_period(self, log_period_ms):
        """Sets the period to log new data."""
        self.log_period_ms = log_period_ms

    def log_data(self, current_time_ms):
        """Add the log for the current time into the data log."""
        if current_time_ms % self.log_period_ms == 0:
            if len(self.log) == 0:
                # This the first call to the log_data function. First allocate some log space
                expected_num_of_logs = int(self.simulation_time_ms / self.simulation_step_size_ms)
                size_of_a_log = len(self.data_to_log)
                self.log = np.zeros((expected_num_of_logs, size_of_a_log), dtype=float)
                self.temp_current_log = np.zeros((1, size_of_a_log), dtype=float)
                if self.log_period_ms == 1:
                    self.log_period_ms = self.simulation_step_size_ms
                print('log_data: log_count: {} single log size: {}'.format(expected_num_of_logs, size_of_a_log))
            item_index = 0
            det_eval_index = 0
            visibility_eval_index = 0
            if self.data_to_log is not None:
                for data_log_description in self.data_to_log:
                    if data_log_description.item_type == ItemDescription.ITEM_TYPE_TIME:
                        self.temp_current_log[0][item_index] = float(current_time_ms)
                    elif data_log_description.item_type == ItemDescription.ITEM_TYPE_VEHICLE:
                        vehicle_index = data_log_description.item_index
                        if len(self.vehicles_manager.vehicles) > vehicle_index:
                            state_index = data_log_description.item_state_index
                            self.temp_current_log[0][item_index] = \
                                self.vehicles_manager.vehicles[vehicle_index].get_vehicle_state_with_id(state_index)
                    elif data_log_description.item_type == ItemDescription.ITEM_TYPE_PEDESTRIAN:
                        pedestrian_index = data_log_description.item_index
                        if len(self.pedestrians_manager.pedestrians) > pedestrian_index:
                            state_index = data_log_description.item_state_index
                            self.temp_current_log[0][item_index] = \
                                self.pedestrians_manager.pedestrians[pedestrian_index].get_pedestrian_state_with_id(state_index)
                    elif data_log_description.item_type == ItemDescription.ITEM_TYPE_VEHICLE_DET_PERF:
                        vehicle_index = data_log_description.item_index
                        if len(self.vehicles_manager.vehicles) > vehicle_index:
                            state_index = data_log_description.item_state_index
                            self.temp_current_log[0][item_index] = \
                                self.vehicles_manager.get_det_perf(vehicle_index, state_index, 'Car')
                    elif data_log_description.item_type == ItemDescription.ITEM_TYPE_PED_DET_PERF:
                        vehicle_index = data_log_description.item_index
                        if len(self.vehicles_manager.vehicles) > vehicle_index:
                            state_index = data_log_description.item_state_index
                            self.temp_current_log[0][item_index] = \
                                self.vehicles_manager.get_det_perf(vehicle_index, state_index, 'Pedestrian')
                    elif data_log_description.item_type == ItemDescription.ITEM_TYPE_VEHICLE_CONTROL:
                        vehicle_index = data_log_description.item_index
                        if len(self.vehicles_manager.vehicles) > vehicle_index:
                            control_type = data_log_description.item_state_index
                            self.temp_current_log[0][item_index] = \
                                self.vehicles_manager.get_vehicle_control(vehicle_index, control_type)
                    elif data_log_description.item_type == ItemDescription.ITEM_TYPE_DET_EVAL:
                        if det_eval_index in self.vehicles_manager.detection_eval_dict:
                            self.temp_current_log[0][item_index] = \
                                self.vehicles_manager.detection_eval_dict[det_eval_index]
                        else:
                            self.temp_current_log[0][item_index] = 0.0
                        det_eval_index += 1
                    elif data_log_description.item_type == ItemDescription.VISIBILITY_EVAL:
                        if visibility_eval_index in self.vehicles_manager.visibility_eval_dict:
                            self.temp_current_log[0][item_index] = \
                                self.vehicles_manager.visibility_eval_dict[visibility_eval_index]
                        else:
                            self.temp_current_log[0][item_index] = 0.0
                            visibility_eval_index += 1
                    item_index += 1
                if self.current_log_size < len(self.log):
                    self.log[self.current_log_size] = self.temp_current_log
                else:
                    # We have run out of log space. Let's add one more space. (This is not expected to happen)
                    self.log = np.append(self.log, self.temp_current_log, axis=0)
                    print('Log size was unexpectedly small.')
                self.current_log_size += 1

    def get_log_info(self):
        """Returns number of log and """
        if len(self.log) > 0:
            log_info = (self.current_log_size, self.temp_current_log.shape[1])
        else:
            log_info = (0, 0)
        return log_info

    def get_log(self, start_index, end_index):
        """Returns the requested part of the data log."""
        if start_index == end_index:
            start_index = 0
            end_index = self.current_log_size
        elif end_index > self.current_log_size:
            end_index = self.current_log_size
        return self.log[start_index:end_index]
