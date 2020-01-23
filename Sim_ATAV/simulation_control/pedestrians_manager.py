"""Defines PedestriansManager class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import math
import numpy as np
from Sim_ATAV.common.coordinate_system import CoordinateSystem
from Sim_ATAV.simulation_control.staliro_signal import STaliroSignal
from Sim_ATAV.simulation_control.item_description import ItemDescription
from Sim_ATAV.simulation_control.pedestrian_walker import PedestrianWalker


class PedestriansManager(object):
    """PedestriansManager keeps track of the pedestrians in the simulation environment."""
    POSITION_REPORTING = 0
    ROTATION_REPORTING = 1
    CORNERS_REPORTING = 2

    def __init__(self, supervisor_controller, controller_comm_interface, vehicles_manager):
        self.emitter_name = "emitter"
        self.debug_mode = 0
        self.pedestrians = []
        self.pedestrian_dictionary = {}
        self.total_pedestrian_count = 0
        self.has_sent_controller_config = False
        self.supervisor_control = supervisor_controller
        self.time_step = 0.01
        self.current_sim_time = 0.0
        self.step_counter = 0
        self.controller_comm_interface = controller_comm_interface
        self.supervisor_emitter = 'NOT SET'
        self.reporting_dict = {}
        self.stop_before_collision_list = []
        self.vehicles_manager = vehicles_manager
        self.pedestrian_walker_list = []

    def record_pedestrian(self, pedestrian_object):
        """Add the pedestrian into the records."""
        self.pedestrians.append(pedestrian_object)
        self.pedestrians[-1].node = \
            self.supervisor_control.get_obj_node(pedestrian_object)
        self.pedestrians[-1].translation = \
            self.supervisor_control.get_obj_field(pedestrian_object, "translation")
        self.pedestrians[-1].rotation = \
            self.supervisor_control.get_obj_field(pedestrian_object, "rotation")
        self.pedestrians[-1].name = \
            self.supervisor_control.get_obj_field(pedestrian_object, "name")
        self.pedestrians[-1].current_position = \
            self.supervisor_control.get_obj_position_3D(pedestrian_object)
        # We use current pedestrian index as its id as well:
        self.pedestrian_dictionary[self.pedestrians[-1].def_name] = self.total_pedestrian_count
        self.total_pedestrian_count += 1
        if pedestrian_object.controller == 'void':
            self.pedestrian_walker_list.append(PedestrianWalker(self.pedestrians[-1], self.supervisor_control))
        else:
            self.pedestrian_walker_list.append(None)

    def update_pedestrian_states(self, pedestrian):
        """Update the current states of the pedestrian."""
        pedestrian.previous_position = pedestrian.current_position[:]
        pedestrian.current_position = self.supervisor_control.get_obj_position_3D(pedestrian)
        pedestrian.previous_velocity = pedestrian.current_velocity
        pedestrian.current_velocity = self.supervisor_control.get_obj_velocity(pedestrian)
        pedestrian.current_orientation = self.supervisor_control.get_obj_orientation(pedestrian, 'y')

        pedestrian.speed = math.sqrt(pedestrian.current_velocity[0]**2 + \
                                     pedestrian.current_velocity[1]**2 + \
                                     pedestrian.current_velocity[2]**2)
        pedestrian.state_record_time = self.current_sim_time

    def update_all_pedestrian_states(self):
        """Updates the state of the all pedestrians."""
        for pedestrian in self.pedestrians:
            self.update_pedestrian_states(pedestrian)

    def get_reference_value(self, ref_index, ref_field, current_sim_time):
        """Get value of the reference field of the indexed pedestrian at the given time."""
        if ref_index == 0:  # reference is time
            ret_val = current_sim_time
        else:
            pedestrian = self.pedestrians[ref_index-1]
            if ref_field == 0:
                ret_val = pedestrian.speed
            elif ref_field == 1:
                pos = self.supervisor_control.get_obj_position_3D(pedestrian)
                ret_val = pos[0]
            elif ref_field == 2:
                pos = self.supervisor_control.get_obj_position_3D(pedestrian)
                ret_val = pos[1]
            elif ref_field == 3:
                pos = self.supervisor_control.get_obj_position_3D(pedestrian)
                ret_val = pos[2]
            else:
                ret_val = 0.0
        return ret_val

    def transmit_all_pedestrian_positions(self, emitter):
        """Transmit all pedestrian positions through emitter."""
        for pedestrian in self.pedestrians:
            # print('Ped {} Position: {}'.format(pedestrian.ped_id, pedestrian.current_position))
            self.controller_comm_interface.transmit_pedestrian_position_message(emitter, \
                                                                                pedestrian.ped_id, \
                                                                                pedestrian.current_position)

    def apply_manual_position_control(self, pedestrian_id):
        """Manually control the position of the pedestrian."""
        pedestrian = self.pedestrians[self.pedestrian_dictionary[pedestrian_id]]
        pos = self.supervisor_control.get_obj_position_3D(pedestrian)
        for sig in pedestrian.signal:
            reference_value = self.get_reference_value(sig.ref_index, sig.ref_field, self.current_sim_time)
            signal_value = sig.get_signal_value_corresponding_to_value_of_reference(reference_value,
                                                                                    STaliroSignal.INTERPOLATION_TYPE_NONE)
            if sig.signal_type == sig.SIGNAL_TYPE_SPEED:
                # When I change speed aggressively, the vehicle rolls over. Look into this.
                pos[0] = pos[0] + signal_value * self.time_step
                self.supervisor_control.set_obj_position_3D(pedestrian, pos)
            if sig.signal_type == sig.SIGNAL_TYPE_Y_POSITION:
                pos[2] = signal_value
                self.supervisor_control.set_obj_position_3D(pedestrian, pos)

    def set_time_step(self, time_step):
        """Set the time_step."""
        self.time_step = time_step

    def get_emitter(self):
        """Returns the supervisor emitter"""
        supervisor_emitter = self.supervisor_control.get_emitter(self.emitter_name)
        return supervisor_emitter

    def add_stop_before_collision_item(self, item_to_stop, item_not_to_collide):
        """Adds the item descriptions for the item to stop and item not to collide into the local list."""
        self.stop_before_collision_list.append((item_to_stop, item_not_to_collide))

    def set_initial_state(self, pedestrian_index, state_index, initial_value):
        """Sets the given initial state value for the requested pedestrian."""
        if len(self.pedestrians) > pedestrian_index:
            pedestrian = self.pedestrians[pedestrian_index]
            if state_index == pedestrian.STATE_ID_VELOCITY_X:
                obj_velocity = self.supervisor_control.get_obj_velocity(pedestrian)
                obj_velocity[CoordinateSystem.X_AXIS] = initial_value
                self.supervisor_control.set_obj_velocity(pedestrian, obj_velocity)
            elif state_index == pedestrian.STATE_ID_VELOCITY_Y:
                obj_velocity = self.supervisor_control.get_obj_velocity(pedestrian)
                obj_velocity[CoordinateSystem.Y_AXIS] = initial_value
                self.supervisor_control.set_obj_velocity(pedestrian, obj_velocity)
            elif state_index == pedestrian.STATE_ID_VELOCITY_Z:
                obj_velocity = self.supervisor_control.get_obj_velocity(pedestrian)
                obj_velocity[CoordinateSystem.Z_AXIS] = initial_value
                self.supervisor_control.set_obj_velocity(pedestrian, obj_velocity)
            else:
                print("WARNING! Requested initial state setting is not supported yet! {} {} {}".format(pedestrian_index,
                                                                                                       state_index,
                                                                                                       initial_value))

    def simulate_pedestrians(self, current_sim_time_s):
        """Simulation pedestrians for one time step."""
        self.current_sim_time = current_sim_time_s
        control_type = 0
        if self.supervisor_emitter == 'NOT SET':
            self.supervisor_emitter = self.get_emitter()
        self.update_all_pedestrian_states()

        new_3D_positions = []
        new_rotations = []
        new_joint_angle_lists = []
        should_pedestrian_stop = [False]*len(self.pedestrians)

        for (ped_ind, pedestrian) in enumerate(self.pedestrians):
            if self.pedestrian_walker_list[ped_ind] is not None:
                (new_3D_position, new_rotation, new_joint_angles) = self.pedestrian_walker_list[ped_ind].compute_movement()
                new_3D_positions.append(new_3D_position)
                new_rotations.append(new_rotation)
                new_joint_angle_lists.append(new_joint_angles)
            else:
                new_3D_positions.append(pedestrian.current_position)
                new_rotations.append(pedestrian.rotation)
                new_joint_angle_lists.append(None)

        # Following is to stop pedestrians before they collide into another pedestrian or vehicle (stops like DARTH VADER)
        for (item_to_stop, item_not_to_collide) in self.stop_before_collision_list:
            for (ped_ind, pedestrian) in enumerate(self.pedestrians):
                if item_to_stop.item_index in [ItemDescription.ITEM_INDEX_ALL, ped_ind]:
                    if item_not_to_collide.item_type == ItemDescription.ITEM_TYPE_PEDESTRIAN:
                        for (ped2_ind, _ped2) in enumerate(self.pedestrians):
                            if item_not_to_collide.item_index in [ItemDescription.ITEM_INDEX_ALL, ped2_ind] and ped2_ind != ped_ind:
                                not_to_collide_pos = new_3D_positions[ped2_ind]
                                if np.linalg.norm(np.array(new_3D_positions[ped_ind]) - np.array(not_to_collide_pos)) < 10.0:
                                    ped_towards_left = (math.pi/2.0 - math.pi/4 < new_rotations[ped_ind][3] < math.pi/2.0 + math.pi/4)
                                    ped_towards_right = (-math.pi/2.0 - math.pi/4 < new_rotations[ped_ind][3] < -math.pi/2.0 + math.pi/4)
                                    ped_on_right = ((not_to_collide_pos[0] - 2.2 < new_3D_positions[ped_ind][0] < not_to_collide_pos[0] - 0.7)
                                                    and (not_to_collide_pos[2] - 1.0 < new_3D_positions[ped_ind][2] < not_to_collide_pos[2] + 4.0))
                                    ped_on_left = ((not_to_collide_pos[0] + 2.2 > new_3D_positions[ped_ind][0] > not_to_collide_pos[0] + 0.7)
                                                   and (not_to_collide_pos[2] - 1.0 < new_3D_positions[ped_ind][2] < not_to_collide_pos[2] + 4.0))
                                    if (ped_on_left and ped_towards_right) or (ped_on_right and ped_towards_left):
                                        should_pedestrian_stop[ped_ind] = True
                    elif (item_not_to_collide.item_type is ItemDescription.ITEM_TYPE_VEHICLE) and (self.vehicles_manager is not None):
                        for (vhc_ind, vhc) in enumerate(self.vehicles_manager.vehicles):
                            if item_not_to_collide.item_index in [ItemDescription.ITEM_INDEX_ALL, vhc_ind]:
                                not_to_collide_pos = vhc.current_position
                                if np.linalg.norm(np.array(new_3D_positions[ped_ind]) - np.array(not_to_collide_pos)) < 10.0:
                                    ped_towards_left = (math.pi/2.0 - math.pi/4 < new_rotations[ped_ind][3] < math.pi/2.0 + math.pi/4)
                                    ped_towards_right = (-math.pi/2.0 - math.pi/4 < new_rotations[ped_ind][3] < -math.pi/2.0 + math.pi/4)
                                    ped_on_right = ((not_to_collide_pos[0] - 2.2 < new_3D_positions[ped_ind][0] < not_to_collide_pos[0] - 0.7)
                                                    and (not_to_collide_pos[2] - 1.0 < new_3D_positions[ped_ind][2] < not_to_collide_pos[2] + 4.0))
                                    ped_on_left = ((not_to_collide_pos[0] + 2.2 > new_3D_positions[ped_ind][0] > not_to_collide_pos[0] + 0.7)
                                                   and (not_to_collide_pos[2] - 1.0 < new_3D_positions[ped_ind][2] < not_to_collide_pos[2] + 4.0))
                                    if (ped_on_left and ped_towards_right) or (ped_on_right and ped_towards_left):
                                        should_pedestrian_stop[ped_ind] = True

        for (ped_ind, pedestrian) in enumerate(self.pedestrians):
            if self.pedestrian_walker_list[ped_ind] is not None:
                if not should_pedestrian_stop[ped_ind]:
                    self.pedestrian_walker_list[ped_ind].apply_movement(new_3D_positions[ped_ind], \
                                                                        new_rotations[ped_ind], \
                                                                        new_joint_angle_lists[ped_ind])
            else:
                if should_pedestrian_stop[ped_ind]:  # This may not work when pedestrian is controlled by another supervisor.
                    temp_pos = pedestrian.previous_position[:]
                    self.supervisor_control.set_obj_position_3D(pedestrian, temp_pos)
                    self.supervisor_control.reset_obj_physics(pedestrian)

        if self.supervisor_emitter is not None:
            for pedestrian in self.pedestrians:
                report = False
                if (pedestrian.ped_id, self.POSITION_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(pedestrian.ped_id, self.POSITION_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(pedestrian.ped_id, self.POSITION_REPORTING)] = -2 # Won't report next time
                    if period != -2:
                        report = True
                elif (0, self.POSITION_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(0, self.POSITION_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(pedestrian.ped_id, self.POSITION_REPORTING)] = -2 # Won't report next time
                    if period != -2:
                        report = True
                if report and (period == 0 or self.current_sim_time % period == 0):
                    self.controller_comm_interface.transmit_pedestrian_position_message(self.supervisor_emitter, \
                                                                                        pedestrian.ped_id, \
                                                                                        pedestrian.current_position)
                report = False
                if (pedestrian.ped_id, self.ROTATION_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(pedestrian.ped_id, self.ROTATION_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(pedestrian.ped_id, self.ROTATION_REPORTING)] = -2 # Won't report next time
                    if period != -2:
                        report = True
                elif (0, self.ROTATION_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(0, self.ROTATION_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(pedestrian.ped_id, self.ROTATION_REPORTING)] = -2 # Won't report next time
                    if period != -2:
                        report = True
                if report and (period == 0 or self.current_sim_time % period == 0):
                    rotation_matrix = self.supervisor_control.get_obj_orientation(pedestrian)
                    self.controller_comm_interface.transmit_pedestrian_rotation_message(self.supervisor_emitter, \
                                                                                        pedestrian.ped_id, \
                                                                                        rotation_matrix)

                report = False
                if (pedestrian.ped_id, self.CORNERS_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(pedestrian.ped_id, self.CORNERS_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(pedestrian.ped_id, self.CORNERS_REPORTING)] = -2 # Won't report next time
                    if period != -2:
                        report = True
                elif (0, self.CORNERS_REPORTING) in self.reporting_dict:
                    period = self.reporting_dict[(0, self.CORNERS_REPORTING)]
                    if period == -1:  # Report only once
                        self.reporting_dict[(pedestrian.ped_id, self.CORNERS_REPORTING)] = -2 # Won't report next time
                    if period != -2:
                        report = True
                if report and (period == 0 or self.current_sim_time % period == 0):
                    self.controller_comm_interface.transmit_pedestrian_box_corners_message(self.supervisor_emitter, \
                                                                                           pedestrian.ped_id, \
                                                                                           pedestrian.get_pedestrian_box_corners())

        if control_type == 0:
            for pedestrian_id in self.pedestrian_dictionary:
                self.apply_manual_position_control(pedestrian_id)
        self.step_counter += 1

    def set_periodic_reporting(self, report_type, entity_id, period):
        """Enables periodic reporting for the given information type with given period."""
        self.reporting_dict[(entity_id, report_type)] = period
