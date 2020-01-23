"""Defines VehiclesManager class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import math
import time
import numpy as np
from Sim_ATAV.simulation_control.staliro_signal import STaliroSignal
from Sim_ATAV.simulation_control.item_description import ItemDescription
from Sim_ATAV.common.coordinate_system import CoordinateSystem


class VehiclesManager(object):
    """VehiclesManager keeps track of the vehicles in the simulation environment."""
    VHC_DUMMY = 0
    VHC_VUT = 1
    POSITION_REPORTING = 0
    ROTATION_REPORTING = 1
    CORNERS_REPORTING = 2

    def __init__(self, supervisor_controller, controller_comm_interface):
        self.emitter_name = "emitter"
        self.receiver_name = 'receiver'
        self.debug_mode = 0
        self.vehicles = []
        self.vehicle_dictionary = {}
        self.VUT_dictionary = {}
        self.dummy_vhc_dictionary = {}
        self.total_vhc_count = 0
        self.has_sent_controller_config = False
        self.supervisor_control = supervisor_controller
        self.time_step = 0.01
        self.current_sim_time = 0.0
        self.step_counter = 0
        self.num_of_steps_for_jerk_calc = 10
        self.acc_arr = [0.0]*(self.num_of_steps_for_jerk_calc + 3)
        self.cur_jerk_compute_index = 0
        self.controller_comm_interface = controller_comm_interface
        # Following are set to NOT SET instead of None in order to avoid getting device at every cycle
        self.supervisor_emitter = 'NOT SET'
        self.supervisor_receiver = 'NOT SET'
        self.reporting_dict = {}
        self.collect_detection_perf_from_vehicles = []
        self.detection_perf_dict = {}
        self.vehicles_to_collect_control = []
        self.vehicle_control_dict = {}
        self.stop_before_collision_list = []
        self.pedestrians_manager = None
        self.detection_eval_dict = {}
        self.visibility_eval_dict = {}
        self.vehicle_id_dictionary = {}

    def set_pedestrians_manager(self, pedestrians_manager):
        """Sets a reference to the pedestrians manager."""
        self.pedestrians_manager = pedestrians_manager

    def record_vehicle(self, vehicle_object, vehicle_type):
        """Add the vehicle in the records. vehicle_type can be VUT / Dummy"""
        self.vehicles.append(vehicle_object)
        self.vehicles[self.total_vhc_count].node = \
            self.supervisor_control.get_obj_node(self.vehicles[self.total_vhc_count])
        self.vehicles[self.total_vhc_count].translation = \
            self.supervisor_control.get_obj_field(self.vehicles[self.total_vhc_count], "translation")
        self.vehicles[self.total_vhc_count].rotation = \
            self.supervisor_control.get_obj_field(self.vehicles[self.total_vhc_count], "rotation")
        self.vehicles[self.total_vhc_count].name = \
            self.supervisor_control.get_obj_field(self.vehicles[self.total_vhc_count], "name")
        self.vehicles[self.total_vhc_count].front_right_wheel_angular_velocity = \
            self.supervisor_control.get_obj_field(self.vehicles[self.total_vhc_count],
                                                  "front_right_wheel_angular_velocity")
        self.vehicles[self.total_vhc_count].front_left_wheel_angular_velocity = \
            self.supervisor_control.get_obj_field(self.vehicles[self.total_vhc_count],
                                                  "front_left_wheel_angular_velocity")
        self.vehicles[self.total_vhc_count].rear_right_wheel_angular_velocity = \
            self.supervisor_control.get_obj_field(self.vehicles[self.total_vhc_count],
                                                  "rear_right_wheel_angular_velocity")
        self.vehicles[self.total_vhc_count].rear_left_wheel_angular_velocity = \
            self.supervisor_control.get_obj_field(self.vehicles[self.total_vhc_count],
                                                  "rear_left_wheel_angular_velocity")
        self.vehicles[self.total_vhc_count].current_position = \
            self.supervisor_control.get_obj_position_3D(self.vehicles[self.total_vhc_count])
        self.vehicles[self.total_vhc_count].current_orientation = \
            self.supervisor_control.get_obj_orientation(self.vehicles[self.total_vhc_count], 'y')
        # We use current vehicle index as its id as well:
        self.vehicle_dictionary[self.vehicles[self.total_vhc_count].def_name] = self.total_vhc_count
        if vehicle_type == self.VHC_VUT:
            self.VUT_dictionary[self.vehicles[self.total_vhc_count].def_name] = self.total_vhc_count
        if vehicle_type == self.VHC_DUMMY:
            self.dummy_vhc_dictionary[self.vehicles[self.total_vhc_count].def_name] = self.total_vhc_count
        self.vehicle_id_dictionary[vehicle_object.vhc_id] = self.total_vhc_count;
        self.total_vhc_count += 1

    def change_vehicle_pose(self, vehicle_object):
        """Change the vehicle pose. (Generally, due to an external command that changes vhc positions at each step.)"""
        if vehicle_object.vhc_id in self.vehicle_id_dictionary:
            vhc_ind = self.vehicle_id_dictionary[vehicle_object.vhc_id]
            vhc = self.vehicles[vhc_ind]
            pos = vehicle_object.current_position
            rot = vehicle_object.rotation
            self.supervisor_control.set_obj_position_3D(vhc, pos)
            self.supervisor_control.set_obj_rotation(vhc, rot)

    def update_vehicle_states(self, vhc):
        """Update the current states of the vehicle."""
        vhc.previous_velocity = vhc.current_velocity[:] if vhc.current_velocity is not None else None
        vhc.previous_acceleration_3d = vhc.acceleration_3d[:] if vhc.acceleration_3d is not None else None

        vhc.current_position = self.supervisor_control.get_obj_position_3D(vhc)
        vhc.current_orientation = self.supervisor_control.get_obj_orientation(vhc, 'y')
        velocity_6d = self.supervisor_control.get_obj_velocity(vhc)
        vhc.current_velocity = velocity_6d[:3]
        vhc.angular_velocity_3d = velocity_6d[3:]
        vhc.speed = math.sqrt(vhc.current_velocity[0]**2 + vhc.current_velocity[1]**2 + vhc.current_velocity[2]**2)

        if vhc.previous_orientation is None:
            vhc.yaw_rate = 0.0
        else:
            vhc.yaw_rate = ((vhc.current_orientation - vhc.previous_orientation) /
                            (self.current_sim_time - vhc.state_record_time))

        if vhc.previous_velocity is None:
            vhc.acceleration_3d = [0.0, 0.0, 0.0]
        else:
            vhc.acceleration_3d = [
                (vhc.current_velocity[0] - vhc.previous_velocity[0]) / (self.current_sim_time - vhc.state_record_time),
                (vhc.current_velocity[1] - vhc.previous_velocity[1]) / (self.current_sim_time - vhc.state_record_time),
                (vhc.current_velocity[2] - vhc.previous_velocity[2]) / (self.current_sim_time - vhc.state_record_time)]
        orient_vector = self.supervisor_control.get_obj_orientation_vector(vhc, [0.0, 0.0, 1.0])
        vhc.acceleration = np.dot(np.array(vhc.acceleration_3d), orient_vector)  # acceleration along vhc orientation

        if vhc.previous_acceleration_3d is None:
            vhc.jerk_3d = [0.0, 0.0, 0.0]
        else:
            vhc.jerk_3d = [((vhc.acceleration_3d[0] - vhc.previous_acceleration_3d[0]) /
                            (self.current_sim_time - vhc.state_record_time)),
                           ((vhc.acceleration_3d[1] - vhc.previous_acceleration_3d[1]) /
                            (self.current_sim_time - vhc.state_record_time)),
                           ((vhc.acceleration_3d[2] - vhc.previous_acceleration_3d[2]) /
                            (self.current_sim_time - vhc.state_record_time))]
        vhc.jerk = np.dot(np.array(vhc.jerk_3d), orient_vector)  # jerk along the vehicle orientation

        vhc.state_record_time = self.current_sim_time
        vhc.previous_orientation = vhc.current_orientation

    def update_all_vehicles_states(self):
        """Updates the state of the all vehicles."""
        for vhc in self.vehicles:
            self.update_vehicle_states(vhc)

    def get_reference_value(self, ref_index, ref_field, current_sim_time):
        """Get value of the reference field of the indexed vehicle at the given time."""
        if ref_index == 0:  # reference is time
            ret_val = current_sim_time
        else:
            vhc = self.vehicles[ref_index-1]
            if ref_field == 0:
                ret_val = vhc.speed
            elif ref_field == 1:
                pos = self.supervisor_control.get_obj_position_3D(vhc)
                ret_val = pos[0]
            elif ref_field == 2:
                pos = self.supervisor_control.get_obj_position_3D(vhc)
                ret_val = pos[1]
            elif ref_field == 3:
                pos = self.supervisor_control.get_obj_position_3D(vhc)
                ret_val = pos[2]
            else:
                ret_val = 0.0
        return ret_val

    def transmit_all_vhc_positions(self, emitter):
        """Transmit all vehicle positions through emitter."""
        for vhc in self.vehicles:
            # print('Vhc {} Position: {}'.format(vhc.vhc_id, vhc.current_position))
            self.controller_comm_interface.transmit_vehicle_position_message(emitter, vhc.vhc_id, vhc.current_position)

    def transmit_init_controller_params(self, emitter):
        """Transmit the neural network controller parameters."""
        if self.has_sent_controller_config is False:
            for vhc in self.vehicles:
                for c_param in vhc.controller_parameters:
                    self.controller_comm_interface.transmit_set_controller_parameters_message(
                        emitter=emitter,
                        vhc_id=c_param.vehicle_id,
                        parameter_name=c_param.parameter_name,
                        parameter_data=c_param.parameter_data)
                    time.sleep(0.1)
            self.has_sent_controller_config = True

    def apply_manual_position_control(self, vhc_id):
        """Manually control the position of the vehicle."""
        vhc = self.vehicles[self.dummy_vhc_dictionary[vhc_id]]
        pos = self.supervisor_control.get_obj_position_3D(vhc)
        for sig in vhc.signal:
            reference_value = self.get_reference_value(sig.ref_index, sig.ref_field, self.current_sim_time)
            signal_value = sig.get_signal_value_corresponding_to_value_of_reference(
                reference_value, STaliroSignal.INTERPOLATION_TYPE_NONE)
            if sig.signal_type == sig.SIGNAL_TYPE_SPEED:
                pos[0] = pos[0] + signal_value * self.time_step
                self.supervisor_control.set_obj_position_3D(vhc, pos)
            if sig.signal_type == sig.SIGNAL_TYPE_Y_POSITION:
                pos[2] = signal_value
                self.supervisor_control.set_obj_position_3D(vhc, pos)

    def set_time_step(self, time_step):
        """Set the time_step."""
        self.time_step = time_step
        self.num_of_steps_for_jerk_calc = int(math.ceil(0.05 / time_step))
        if self.debug_mode > 0:
            print('Num of steps for jerk calculation = {}'.format(self.num_of_steps_for_jerk_calc))
        self.acc_arr = [0.0]*(self.num_of_steps_for_jerk_calc+3)

    def get_emitter(self):
        """Returns the supervisor emitter"""
        supervisor_emitter = self.supervisor_control.get_emitter(self.emitter_name)
        return supervisor_emitter

    def get_receiver(self):
        """Returns the supervisor receiver"""
        supervisor_receiver = self.supervisor_control.get_receiver(self.receiver_name)
        if supervisor_receiver is not None:
            supervisor_receiver.enable(10)
        return supervisor_receiver

    def get_det_perf(self, vehicle_index, object_index, object_type_text):
        """Get detection performance for the requested object."""
        det_perf = 0.0
        if vehicle_index < len(self.vehicles):
            if (self.vehicles[vehicle_index].vhc_id, object_index, object_type_text) in self.detection_perf_dict:
                det_perf = self.detection_perf_dict[(self.vehicles[vehicle_index].vhc_id, object_index,
                                                     object_type_text)]
        return det_perf

    def get_vehicle_control(self, vehicle_index, control_type):
        """Get applied control actions for the given vehicle."""
        applied_control = 0.0
        if vehicle_index < len(self.vehicles):
            if (self.vehicles[vehicle_index].vhc_id, control_type) in self.vehicle_control_dict:
                applied_control = self.vehicle_control_dict[(self.vehicles[vehicle_index].vhc_id, control_type)]
        return applied_control

    def add_stop_before_collision_item(self, item_to_stop, item_not_to_collide):
        """Adds the item descriptions for the item to stop and item not to collide into the local list."""
        self.stop_before_collision_list.append((item_to_stop, item_not_to_collide))

    def set_initial_state(self, vehicle_index, state_index, initial_value):
        """Sets the given initial state value for the requested vehicle."""
        if len(self.vehicles) > vehicle_index:
            vhc = self.vehicles[vehicle_index]
            if state_index == vhc.STATE_ID_VELOCITY_X:
                obj_velocity = self.supervisor_control.get_obj_velocity(vhc)
                obj_velocity[CoordinateSystem.X_AXIS] = initial_value
                self.supervisor_control.set_obj_velocity(vhc, obj_velocity)
            elif state_index == vhc.STATE_ID_VELOCITY_Y:
                obj_velocity = self.supervisor_control.get_obj_velocity(vhc)
                obj_velocity[CoordinateSystem.Y_AXIS] = initial_value
                self.supervisor_control.set_obj_velocity(vhc, obj_velocity)
            elif state_index == vhc.STATE_ID_VELOCITY_Z:
                obj_velocity = self.supervisor_control.get_obj_velocity(vhc)
                obj_velocity[CoordinateSystem.Z_AXIS] = initial_value
                self.supervisor_control.set_obj_velocity(vhc, obj_velocity)
            else:
                print("WARNING! Requested initial state setting is not supported yet! {} {} {}".format(vehicle_index,
                                                                                                       state_index,
                                                                                                       initial_value))

    def simulate_vehicles(self, current_sim_time_s):
        """Simulation vehicles for one time step."""
        self.current_sim_time = current_sim_time_s
        control_type = 0
        if self.supervisor_emitter == 'NOT SET':
            self.supervisor_emitter = self.get_emitter()
        if self.supervisor_receiver == 'NOT SET':
            self.supervisor_receiver = self.get_receiver()
        self.update_all_vehicles_states()

        # Following is to stop vehicles before they collide into another vehicle or pedestrian (stops like DARTH VADER)
        for (item_to_stop, item_not_to_collide) in self.stop_before_collision_list:
            for (vhc_ind, vhc) in enumerate(self.vehicles):
                if item_to_stop.item_index in [ItemDescription.ITEM_INDEX_ALL, vhc_ind]:
                    if item_not_to_collide.item_type == ItemDescription.ITEM_TYPE_VEHICLE:
                        for (vhc2_ind, vhc2) in enumerate(self.vehicles):
                            if (item_not_to_collide.item_index in [ItemDescription.ITEM_INDEX_ALL, vhc2_ind] and
                                    vhc2_ind != vhc_ind):
                                not_to_collide_pos = vhc2.current_position
                                if (math.sqrt((vhc.current_position[0] - not_to_collide_pos[0])**2 +
                                              (vhc.current_position[2] - not_to_collide_pos[2])**2) < 15.0):
                                    vhc_towards_right = (math.pi/2.0 - math.pi/4 < vhc.current_orientation
                                                         < math.pi/2.0 + math.pi/4)
                                    vhc_towards_left = (-math.pi/2.0 - math.pi/4 < vhc.current_orientation
                                                        < -math.pi/2.0 + math.pi/4)
                                    vhc_on_right = ((not_to_collide_pos[0] - 8.0 < vhc.current_position[0]
                                                     < not_to_collide_pos[0] - 5.0)
                                                    and (not_to_collide_pos[2] - 2.5 < vhc.current_position[2]
                                                         < not_to_collide_pos[2] + 6.5))
                                    vhc_on_left = ((not_to_collide_pos[0] + 8.0 > vhc.current_position[0]
                                                    > not_to_collide_pos[0] + 5.0)
                                                   and (not_to_collide_pos[2] - 2.5 < vhc.current_position[2]
                                                        < not_to_collide_pos[2] + 6.5))
                                    if (vhc.speed > 0.0 and ((vhc_on_left and vhc_towards_left) or
                                                             (vhc_on_right and vhc_towards_right))):
                                        self.supervisor_control.set_obj_position_3D(vhc, vhc.current_position)
                                        self.supervisor_control.reset_obj_physics(vhc)
                    elif ((item_not_to_collide.item_type is ItemDescription.ITEM_TYPE_PEDESTRIAN) and
                          (self.pedestrians_manager is not None)):
                        for (ped_ind, ped) in enumerate(self.pedestrians_manager.pedestrians):
                            if item_not_to_collide.item_index in [ItemDescription.ITEM_INDEX_ALL, ped_ind]:
                                not_to_collide_pos = ped.current_position
                                if (math.sqrt((vhc.current_position[0] - not_to_collide_pos[0])**2 +
                                              (vhc.current_position[2] - not_to_collide_pos[2])**2) < 15.0):
                                    vhc_towards_left = (math.pi/2.0 - math.pi/4 < vhc.current_orientation
                                                        < math.pi/2.0 + math.pi/4)
                                    vhc_towards_right = (-math.pi/2.0 - math.pi/4 < vhc.current_orientation
                                                         < -math.pi/2.0 + math.pi/4)
                                    vhc_on_right = ((not_to_collide_pos[0] - 8.0 < vhc.current_position[0]
                                                     < not_to_collide_pos[0] - 5.0)
                                                    and (not_to_collide_pos[2] - 2.5 < vhc.current_position[2]
                                                         < not_to_collide_pos[2] + 6.5))
                                    vhc_on_left = ((not_to_collide_pos[0] + 8.0 > vhc.current_position[0]
                                                    > not_to_collide_pos[0] + 5.0)
                                                   and (not_to_collide_pos[2] - 2.5 < vhc.current_position[2]
                                                        < not_to_collide_pos[2] + 6.5))
                                    if (vhc.speed > 0.0 and ((vhc_on_left and vhc_towards_right) or
                                                             (vhc_on_right and vhc_towards_left))):
                                        self.supervisor_control.set_obj_position_3D(vhc, vhc.current_position)
                                        self.supervisor_control.reset_obj_physics(vhc)
        if self.supervisor_emitter is not None:
            self.controller_comm_interface.transmit_backlogged_messages(self.supervisor_emitter)
            for vhc in self.vehicles:
                report_vhc = False
                if (vhc.vhc_id, self.POSITION_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(vhc.vhc_id, self.POSITION_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(vhc.vhc_id, self.POSITION_REPORTING)] = -2  # Won't report next time
                    if period != -2:
                        report_vhc = True
                elif (0, self.POSITION_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(0, self.POSITION_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(vhc.vhc_id, self.POSITION_REPORTING)] = -2  # Won't report next time
                    if period != -2:
                        report_vhc = True
                else:
                    period = -2  # This won't happen. Defensive code.
                if report_vhc and (period in [0, -1] or self.current_sim_time % period == 0):
                    self.controller_comm_interface.transmit_vehicle_position_message(self.supervisor_emitter,
                                                                                     vhc.vhc_id,
                                                                                     vhc.current_position)
                report_vhc = False
                if (vhc.vhc_id, self.ROTATION_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(vhc.vhc_id, self.ROTATION_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(vhc.vhc_id, self.ROTATION_REPORTING)] = -2  # Won't report next time
                    if period != -2:
                        report_vhc = True
                elif (0, self.ROTATION_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(0, self.ROTATION_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(vhc.vhc_id, self.ROTATION_REPORTING)] = -2  # Won't report next time
                    if period != -2:
                        report_vhc = True
                else:
                    period = -2  # This won't happen. Defensive code.
                if report_vhc and (period in [0, -1] or self.current_sim_time % period == 0):
                    rotation_matrix = self.supervisor_control.get_obj_orientation(vhc)
                    self.controller_comm_interface.transmit_vehicle_rotation_message(self.supervisor_emitter,
                                                                                     vhc.vhc_id,
                                                                                     rotation_matrix)
                report_vhc = False
                if (vhc.vhc_id, self.CORNERS_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(vhc.vhc_id, self.CORNERS_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(vhc.vhc_id, self.CORNERS_REPORTING)] = -2  # Won't report next time
                    if period != -2:
                        report_vhc = True
                elif (0, self.CORNERS_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(0, self.CORNERS_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(vhc.vhc_id, self.CORNERS_REPORTING)] = -2  # Won't report next time
                    if period != -2:
                        report_vhc = True
                else:
                    period = -2  # This won't happen. Defensive code.
                if report_vhc and (period in [0, -1] or self.current_sim_time % period == 0):
                    self.controller_comm_interface.transmit_vehicle_box_corners_message(self.supervisor_emitter,
                                                                                        vhc.vhc_id,
                                                                                        vhc.get_vehicle_box_corners())
            if self.supervisor_receiver is not None:
                messages = self.controller_comm_interface.receive_all_communication(self.supervisor_receiver)
                command_list = self.controller_comm_interface.extract_all_commands_from_message(messages)
            else:
                command_list = []
            for vhc_ind in self.collect_detection_perf_from_vehicles:
                vhc = self.vehicles[vhc_ind]
                det_performances = self.controller_comm_interface.get_detection_performances(command_list, vhc.vhc_id)
                for (object_index, object_type_text, det_perf) in det_performances:
                    self.detection_perf_dict[(vhc.vhc_id, object_index, object_type_text)] = det_perf

            detection_evals = self.controller_comm_interface.get_detection_evaluations(command_list)
            for (idx, value) in detection_evals:
                self.detection_eval_dict[idx] = value
            visibility_evals = self.controller_comm_interface.get_visibility_evaluations(command_list)
            for (idx, value) in visibility_evals:
                self.visibility_eval_dict[idx] = value
            for vhc_ind in self.vehicles_to_collect_control:
                vhc = self.vehicles[vhc_ind]
                applied_vehicle_controls = self.controller_comm_interface.get_applied_vehicle_controls(command_list,
                                                                                                       vhc.vhc_id)
                for (control_type, control_action) in applied_vehicle_controls:
                    self.vehicle_control_dict[(vhc.vhc_id, control_type)] = control_action
            self.transmit_init_controller_params(self.supervisor_emitter)
        if control_type == 0:
            for vhc_id in self.dummy_vhc_dictionary:
                self.apply_manual_position_control(vhc_id)
        self.step_counter += 1

    def set_periodic_reporting(self, report_type, entity_id, period):
        """Enables periodic reporting for the given information type with given period."""
        self.reporting_dict[(entity_id, report_type)] = period

    def add_vehicle_to_collect_control_list(self, vhc_ind):
        """Add vehicle index to the list of the vehicles for which the control actions are recorded."""
        self.vehicles_to_collect_control.append(vhc_ind)
