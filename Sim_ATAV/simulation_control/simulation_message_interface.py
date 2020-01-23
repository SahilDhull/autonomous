"""Defines the SimulationMessageInterface class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


import struct
import sys
import numpy as np
from Sim_ATAV.simulation_control.webots_road import WebotsRoad
from Sim_ATAV.simulation_control.webots_road_disturbance import WebotsRoadDisturbance
from Sim_ATAV.simulation_control.webots_vehicle import WebotsVehicle
from Sim_ATAV.simulation_control.webots_fog import WebotsFog
from Sim_ATAV.simulation_control.webots_sensor import WebotsSensor
from Sim_ATAV.simulation_control.webots_pedestrian import WebotsPedestrian
from Sim_ATAV.simulation_control.webots_sim_object import WebotsSimObject
from Sim_ATAV.simulation_control.webots_video_recording import WebotsVideoRecording
from Sim_ATAV.simulation_control.staliro_signal import STaliroSignal
from Sim_ATAV.simulation_control.simulation_command import SimulationCommand
from Sim_ATAV.simulation_control.sim_data import SimData
from Sim_ATAV.simulation_control.heart_beat import HeartBeatConfig, HeartBeat
from Sim_ATAV.simulation_control.item_description import ItemDescription
from Sim_ATAV.common.controller_communication_interface import ControllerCommunicationInterface
from Sim_ATAV.vehicle_control.controller_commons.perf_evaluation.detection_evaluation_config \
    import DetectionEvaluationConfig


def get_gata_from_struct(type_identifier, msg, cur_index):
    if type_identifier == 'string':
        (length, ) = struct.unpack('I', msg[cur_index:cur_index + struct.calcsize('I')])
        cur_index += struct.calcsize('I')
        if length > 0:
            (data, ) = struct.unpack(str(length) + 's', msg[cur_index:cur_index + length])
            data = data.decode('ascii')
            data = data.rstrip(' \t\r\n\0')
        else:
            data = ''
        data = (data, )
        new_index = cur_index + length
    else:
        data = struct.unpack(type_identifier, msg[cur_index:cur_index + struct.calcsize(type_identifier)])
        new_index = cur_index + struct.calcsize(type_identifier)
    return data, new_index


class SimulationMessageInterface(object):
    """SimulationMessageInterface class understands the command types,
     and acts as an interpreter between the application and the communication server."""
    # Commands list:
    START_SIM = 1
    RELOAD_WORLD = 2
    SET_HEART_BEAT_CONFIG = 3
    CONTINUE_SIM = 4
    SET_ROBUSTNESS_TYPE = 5
    ADD_DATA_LOG_DESCRIPTION = 6
    SET_VIEW_FOLLOW_ITEM = 7
    GET_ROBUSTNESS = 8
    GET_DATA_LOG_INFO = 9
    GET_DATA_LOG = 10
    SET_CONTROLLER_PARAMETER = 11
    SET_DATA_LOG_PERIOD_MS = 12
    SET_VIEW_POINT_POSITION = 13
    SET_VIEW_POINT_ORIENTATION = 14
    SET_VIDEO_RECORDING = 15
    ADD_DET_EVAL_CONFIG = 16
    ADD_VISIBILITY_EVAL_CONFIG = 17
    CHANGE_VHC_POSITION = 20
    SURROUNDINGS_ADD = 100
    SURROUNDINGS_DEF = 101
    S_ROAD = 102  # Sub command (adds detail to a command)
    STRAIGHT_ROAD = 103  # Sub command (adds detail to a command)
    ROAD_DISTURBANCE_ADD = 104
    ROAD_DISTURBANCE_DEF = 105
    FOG_ADD = 106
    FOG_DEF = 107
    DUMMY_ACTORS_ADD = 120
    DUMMY_ACTORS_DEF = 121
    D_VHC = 122  # Sub command (adds detail to a command)
    VUT_ADD = 130
    VUT_DEF = 131
    VUT_VHC = 132  # Sub command (adds detail to a command)
    PEDESTRIAN_ADD = 140
    PEDESTRIAN_DEF = 141
    HUMAN = 142  # Sub command (adds detail to a command)
    ADD_OBJECT_TO_SIMULATION = 150
    SET_PERIODIC_REPORTING = 160
    PERIODIC_VHC_POSITIONS = 161  # Sub command (adds detail to a command)
    PERIODIC_VHC_ROTATIONS = 162  # Sub command (adds detail to a command)
    PERIODIC_PED_POSITIONS = 163  # Sub command (adds detail to a command)
    PERIODIC_PED_ROTATIONS = 164  # Sub command (adds detail to a command)
    PERIODIC_VHC_BOX_CORNERS = 165  # Sub command (adds detail to a command)
    PERIODIC_PED_BOX_CORNERS = 166  # Sub command (adds detail to a command)
    REPORT_ALL_ENTITIES = 0  # Sub command (adds detail to a command)
    SET_INITIAL_STATE = 180
    STOP_BEFORE_COLLISION = 190
    ROBUSTNESS = 201
    HEART_BEAT = 202
    DATA_LOG_INFO = 203
    DATA_LOG = 204
    ACK = 250

    def __init__(self):
        self.debug_mode = 0
        self.controller_comm_interface = ControllerCommunicationInterface()

    def interpret_message(self, msg):
        """Extracts the command and object information from the given raw msg."""
        obj = None
        (command, ) = struct.unpack('B', msg[0:struct.calcsize('B')])
        cur_msg_index = struct.calcsize('B')
        if self.debug_mode:
            print("SimulationMessageInterface : msg length:{}".format(len(msg)))
        if command == self.ACK:
            obj = None
        elif command == self.CONTINUE_SIM:
            obj = None
        elif command == self.HEART_BEAT:
            obj = self.interpret_heart_beat_message(msg[cur_msg_index:])
        elif command == self.SET_CONTROLLER_PARAMETER:
            obj = self.interpret_set_controller_parameter_command(msg[cur_msg_index:])
        elif command == self.SET_DATA_LOG_PERIOD_MS:
            obj = self.interpret_set_data_log_period_ms_command(msg[cur_msg_index:])
        elif command in (self.SURROUNDINGS_DEF, self.SURROUNDINGS_ADD):
            obj = self.interpret_add_road_to_simulation_command(msg[cur_msg_index:])
        elif command in (self.ROAD_DISTURBANCE_DEF, self.ROAD_DISTURBANCE_ADD):
            obj = self.interpret_add_road_disturbance_to_simulation_command(msg[cur_msg_index:])
        elif command in (self.FOG_DEF, self.FOG_ADD):
            obj = self.interpret_add_fog_to_simulation_command(msg[cur_msg_index:])
        elif command in (self.DUMMY_ACTORS_DEF, self.DUMMY_ACTORS_ADD, self.VUT_DEF, self.VUT_ADD):
            is_dummy = command in (self.DUMMY_ACTORS_DEF, self.DUMMY_ACTORS_ADD)
            obj = self.interpret_add_vehicle_to_simulation_command(msg[cur_msg_index:], is_dummy)
        elif command == self.CHANGE_VHC_POSITION:
            obj = self.interpret_change_vhc_position_command(msg[cur_msg_index:])
        elif command in (self.PEDESTRIAN_DEF, self.PEDESTRIAN_ADD):
            obj = self.interpret_add_pedestrian_command(msg[cur_msg_index:])
        elif command == self.ADD_OBJECT_TO_SIMULATION:
            obj = self.interpret_add_object_to_simulation_command(msg[cur_msg_index:])
        elif command == self.SET_ROBUSTNESS_TYPE:
            obj = self.interpret_set_robustness_type_command(msg[cur_msg_index:])
        elif command == self.ADD_DATA_LOG_DESCRIPTION:
            obj = self.interpret_add_data_log_description_command(msg[cur_msg_index:])
        elif command == self.SET_VIDEO_RECORDING:
            print('set video recording msg')
            obj = self.interpret_set_video_recording_command(msg[cur_msg_index:])
        elif command == self.START_SIM:
            obj = self.interpret_start_simulation_command(msg[cur_msg_index:])
        elif command == self.RELOAD_WORLD:
            obj = None
            print("SimulationMessageInterface: Revert world")
        elif command == self.GET_ROBUSTNESS:
            obj = None
        elif command == self.GET_DATA_LOG_INFO:
            obj = None
        elif command == self.GET_DATA_LOG:
            obj = self.interpret_get_data_log_command(msg[cur_msg_index:])
        elif command == self.DATA_LOG_INFO:
            obj = self.interpret_log_info_message(msg[cur_msg_index:])
        elif command == self.DATA_LOG:
            obj = self.interpret_data_log_message(msg[cur_msg_index:])
        elif command == self.SET_HEART_BEAT_CONFIG:
            obj = self.interpret_set_heart_beat_config_command(msg[cur_msg_index:])
        elif command == self.SET_VIEW_FOLLOW_ITEM:
            obj = self.interpret_set_view_follow_item_command(msg[cur_msg_index:])
        elif command == self.SET_VIEW_POINT_POSITION:
            obj = self.interpret_set_view_point_position_command(msg[cur_msg_index:])
        elif command == self.SET_VIEW_POINT_ORIENTATION:
            obj = self.interpret_set_view_point_orientation_command(msg[cur_msg_index:])
        elif command == self.SET_INITIAL_STATE:
            obj = self.interpret_set_initial_state_command(msg[cur_msg_index:])
        elif command == self.STOP_BEFORE_COLLISION:
            obj = self.interpret_stop_before_collision_command(msg[cur_msg_index:])
        elif command == self.SET_PERIODIC_REPORTING:
            obj = self.interpret_set_periodic_reporting_command(msg[cur_msg_index:])
        elif command == self.ROBUSTNESS:
            obj = self.interpret_robustness_msg(msg[cur_msg_index:])
        elif command == self.ADD_DET_EVAL_CONFIG:
            obj = self.interpret_add_detection_evaluation_config(msg[cur_msg_index:])
        elif command == self.ADD_VISIBILITY_EVAL_CONFIG:
            obj = self.interpret_add_visibility_evaluation_config(msg[cur_msg_index:])
        else:
            print("SimulationMessageInterface: Unknown COMMAND {}".format(command))

        ret_cmd = SimulationCommand(command, obj)
        return ret_cmd

    def generate_ack_message(self):
        """Creates ACKNOWLEDGEMENT message to be sent."""
        msg = struct.pack('B', self.ACK)
        return msg

    def generate_continue_sim_command(self):
        """Creates CONTINUE_SIM command."""
        command = struct.pack('B', self.CONTINUE_SIM)
        return command

    def generate_get_robustness_command(self):
        """Creates GET_ROBUSTNESS command."""
        command = struct.pack('B', self.GET_ROBUSTNESS)
        return command

    def generate_robustness_msg(self, robustness):
        """Creates robustness message with the given robustness value."""
        msg = struct.pack('B', self.ROBUSTNESS)
        msg += struct.pack('d', robustness)
        return msg

    def interpret_robustness_msg(self, msg):
        cur_msg_index = 0
        (obj, ) = struct.unpack('d', msg[cur_msg_index:cur_msg_index + struct.calcsize('d')])
        return obj

    def generate_set_heart_beat_config_command(self, sync_type, heart_beat_period_ms):
        """Creates SET_HEART_BEAT_CONFIG command."""
        command = struct.pack('B', self.SET_HEART_BEAT_CONFIG)
        command += struct.pack('II', sync_type, heart_beat_period_ms)
        return command

    def interpret_set_heart_beat_config_command(self, msg):
        cur_msg_index = 0
        obj = HeartBeatConfig()
        (obj.sync_type, obj.period_ms) = \
            struct.unpack('II', msg[cur_msg_index:cur_msg_index + struct.calcsize('II')])
        print("Heart Beat Type: {} Period: {}".format(obj.sync_type, obj.period_ms))
        return obj

    def generate_set_data_log_period_ms_command(self, data_log_period_ms):
        """Creates SET_DATA_LOG_PERIOD_MS command."""
        command = struct.pack('B', self.SET_DATA_LOG_PERIOD_MS)
        command += struct.pack('I', data_log_period_ms)
        return command

    def interpret_set_data_log_period_ms_command(self, msg):
        cur_msg_index = 0
        (obj, ) = struct.unpack('I', msg[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        return obj

    def generate_heart_beat_message(self, current_simulation_time_ms, simulation_status):
        """Creates heartbeat message with the given simulation time."""
        msg = struct.pack('B', self.HEART_BEAT)
        msg += struct.pack('B', simulation_status)
        msg += struct.pack('I', current_simulation_time_ms)
        return msg

    def interpret_heart_beat_message(self, msg):
        cur_msg_index = 0
        (sim_status,) = struct.unpack('B', msg[cur_msg_index:cur_msg_index + struct.calcsize('B')])
        cur_msg_index += struct.calcsize('B')
        (sim_time_ms, ) = struct.unpack('I', msg[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        obj = HeartBeat(simulation_status=sim_status, simulation_time_ms=sim_time_ms)
        return obj

    def generate_start_simulation_command(self, duration, step_size, sim_type):
        """Creates START_SIM command."""
        command = struct.pack('B', self.START_SIM)
        command += struct.pack('IIB', duration, step_size, sim_type)
        return command

    def interpret_start_simulation_command(self, msg):
        cur_msg_index = 0
        obj = SimData()
        (obj.simulation_duration_ms,
         obj.simulation_step_size_ms,
         obj.simulation_execution_mode) = \
            struct.unpack('IIB', msg[cur_msg_index:cur_msg_index + struct.calcsize('IIB')])
        print("SimulationMessageInterface: Simulation Duration: {} \
               step size: {} type: {}".format(obj.simulation_duration_ms,
                                              obj.simulation_step_size_ms,
                                              obj.simulation_execution_mode))
        return obj

    def generate_restart_simulation_command(self):
        """Creates RELOAD_WORLD command."""
        command = struct.pack('B', self.RELOAD_WORLD)
        return command

    def generate_set_robustness_type_command(self, robustness_type):
        """Creates SET_ROBUSTNESS_TYPE command."""
        command = struct.pack('B', self.SET_ROBUSTNESS_TYPE)
        command += struct.pack('I', robustness_type)
        return command

    def interpret_set_robustness_type_command(self, msg):
        cur_msg_index = 0
        (obj, ) = struct.unpack('I', msg[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        return obj

    def generate_set_view_follow_item_command(self, item_type, item_index):
        """Creates SET_VIEW_FOLLOW_ITEM command."""
        # The world must have DEF VIEWPOINT Viewpoint {...} for the viewpoint for this to work.
        command = struct.pack('B', self.SET_VIEW_FOLLOW_ITEM)
        command += struct.pack('BB', item_type, item_index)
        return command

    def interpret_set_view_follow_item_command(self, msg):
        cur_msg_index = 0
        obj = ItemDescription()
        (obj.item_type, obj.item_index) = \
            struct.unpack('BB', msg[cur_msg_index:cur_msg_index + struct.calcsize('BB')])
        return obj

    def generate_set_view_point_position_command(self, view_position):
        """Creates SET_VIEW_POINT_POSITION command."""
        # The world must have DEF VIEWPOINT Viewpoint {...} for the viewpoint for this to work.
        command = struct.pack('B', self.SET_VIEW_POINT_POSITION)
        command += struct.pack('ddd', view_position[0], view_position[1], view_position[2])
        return command

    def interpret_set_view_point_position_command(self, msg):
        cur_msg_index = 0
        obj = [0.0, 0.0, 0.0]
        (obj[0], obj[1], obj[2]) = \
            struct.unpack('ddd', msg[cur_msg_index:cur_msg_index + struct.calcsize('ddd')])
        return obj

    def generate_set_view_point_orientation_command(self, view_orientation):
        """Creates SET_VIEW_POINT_ORIENTATION command."""
        # The world must have DEF VIEWPOINT Viewpoint {...} for the viewpoint for this to work.
        command = struct.pack('B', self.SET_VIEW_POINT_ORIENTATION)
        command += struct.pack('dddd', view_orientation[0], view_orientation[1],
                               view_orientation[2], view_orientation[3])
        return command

    def generate_set_video_recording_command(self, video_recording_obj):
        """Creates SET_VIDEO_RECORDING command."""
        # The world must have DEF VIEWPOINT Viewpoint {...} for the viewpoint for this to work.
        command = struct.pack('B', self.SET_VIDEO_RECORDING)
        command += struct.pack('IIIII',
                               video_recording_obj.width,
                               video_recording_obj.height,
                               video_recording_obj.codec,
                               video_recording_obj.quality,
                               video_recording_obj.acceleration)
        command += struct.pack('?', video_recording_obj.is_caption)
        byte_data = video_recording_obj.caption_name.encode('ascii')
        command += struct.pack('I', len(byte_data))
        command += struct.pack(str(len(byte_data)) + 's', byte_data)
        byte_data = video_recording_obj.filename.encode('ascii')
        command += struct.pack('I', len(byte_data))
        command += struct.pack(str(len(byte_data)) + 's', byte_data)
        return command

    def interpret_set_video_recording_command(self, msg):
        """Interprets SET_VIDEO_RECORDING command."""
        video_recording_obj = WebotsVideoRecording()
        cur_msg_index = 0
        (video_recording_obj.width,
         video_recording_obj.height,
         video_recording_obj.codec,
         video_recording_obj.quality,
         video_recording_obj.acceleration) = struct.unpack('IIIII', msg[cur_msg_index:cur_msg_index + struct.calcsize('IIIII')])
        cur_msg_index += struct.calcsize('IIIII')
        (video_recording_obj.is_caption, ) = struct.unpack('?', msg[cur_msg_index:cur_msg_index + struct.calcsize('?')])
        cur_msg_index += struct.calcsize('?')
        ((video_recording_obj.caption_name, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
        ((video_recording_obj.filename, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
        return video_recording_obj

    def interpret_set_view_point_orientation_command(self, msg):
        cur_msg_index = 0
        obj = [0.0, 0.0, 0.0, 0.0]
        (obj[0], obj[1], obj[2], obj[3]) = \
            struct.unpack('dddd', msg[cur_msg_index:cur_msg_index + struct.calcsize('dddd')])
        return obj

    def generate_set_initial_state_command(self, item_type, item_index, state_index, initial_value):
        """Creates SET_INITIAL_STATE command."""
        # This message should be called after adding the objects into the world.
        command = struct.pack('B', self.SET_INITIAL_STATE)
        command += struct.pack('BBB', item_type, item_index, state_index)
        command += struct.pack('d', initial_value)
        return command

    def interpret_set_initial_state_command(self, msg):
        """Extracts information from SET_INITIAL_STATE command."""
        cur_msg_index = 0
        item = ItemDescription()
        (item.item_type, item.item_index, item.item_state_index) = \
            struct.unpack('BBB', msg[cur_msg_index:cur_msg_index + struct.calcsize('BBB')])
        cur_msg_index = cur_msg_index + struct.calcsize('BBB')
        (value, ) = struct.unpack('d', msg[cur_msg_index:cur_msg_index + struct.calcsize('d')])
        obj = (item, value)
        return obj

    def generate_stop_before_collision_command(self,
                                               item_to_stop_type,
                                               item_to_stop_ind,
                                               item_not_to_collide_type,
                                               item_not_to_collide_ind):
        """Creates STOP_BEFORE_COLLISION command."""
        command = struct.pack('B', self.STOP_BEFORE_COLLISION)
        command += struct.pack('BBBB', item_to_stop_type,
                               item_to_stop_ind,
                               item_not_to_collide_type,
                               item_not_to_collide_ind)
        return command

    def interpret_stop_before_collision_command(self, msg):
        """Extracts information from STOP_BEFORE_COLLISION command."""
        cur_msg_index = 0
        item_to_stop = ItemDescription()
        item_not_to_collide = ItemDescription()
        (item_to_stop.item_type,
         item_to_stop.item_index,
         item_not_to_collide.item_type,
         item_not_to_collide.item_index) = \
            struct.unpack('BBBB', msg[cur_msg_index:cur_msg_index + struct.calcsize('BBBB')])
        obj = (item_to_stop, item_not_to_collide)
        return obj

    def generate_add_data_log_description_command(self, item_type, item_index, item_state_index):
        """Creates ADD_DATA_LOG_DESCRIPTION command."""
        command = struct.pack('B', self.ADD_DATA_LOG_DESCRIPTION)
        command += struct.pack('BBB', item_type, item_index, item_state_index)
        return command

    def interpret_add_data_log_description_command(self, msg):
        cur_msg_index = 0
        obj = ItemDescription()
        (obj.item_type, obj.item_index, obj.item_state_index) = \
            struct.unpack('BBB', msg[cur_msg_index:cur_msg_index + struct.calcsize('BBB')])
        return obj

    def generate_get_log_info_command(self):
        """Creates GET_DATA_LOG_INFO command."""
        command = struct.pack('B', self.GET_DATA_LOG_INFO)
        return command

    def generate_log_info_message(self, log_info):
        """Creates DATA_LOG_INFO command."""
        command = struct.pack('B', self.DATA_LOG_INFO)
        command += struct.pack('II', log_info[0], log_info[1])
        return command

    def interpret_log_info_message(self, msg):
        cur_msg_index = 0
        (num_log, size_of_each_log) = \
            struct.unpack('II', msg[cur_msg_index:cur_msg_index + struct.calcsize('II')])
        obj = (num_log, size_of_each_log)
        return obj

    def generate_get_data_log_command(self, start_index, end_index):
        """Creates GET_DATA_LOG command."""
        command = struct.pack('B', self.GET_DATA_LOG)
        command += struct.pack('II', start_index, end_index)
        return command

    def interpret_get_data_log_command(self, msg):
        cur_msg_index = 0
        (log_start_index, log_end_index) = \
            struct.unpack('II', msg[cur_msg_index:cur_msg_index + struct.calcsize('II')])
        obj = (log_start_index, log_end_index)
        return obj

    def generate_data_log_message(self, data_log):
        """Creates DATA_LOG command."""
        command = struct.pack('B', self.DATA_LOG)
        command += struct.pack('I', data_log.size)
        command += data_log.astype('d').tostring()
        return command

    def interpret_data_log_message(self, msg):
        cur_msg_index = 0
        (num_data, ) = \
            struct.unpack('I', msg[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        obj = np.fromstring(msg[cur_msg_index:], dtype=float, count=num_data)
        return obj

    def generate_add_road_to_simulation_command(self, road_object, is_create):
        """Creates a command to add the given road to the simulation.
        is_create: True: the road object will be generated, False: it is already in the world."""
        if is_create:
            cmd = self.SURROUNDINGS_ADD
        else:
            cmd = self.SURROUNDINGS_DEF
        msg = struct.pack('B', cmd)
        msg += struct.pack('B', self.S_ROAD)
        msg += struct.pack('dddddddddB??',
                           road_object.position[0],
                           road_object.position[1],
                           road_object.position[2],
                           road_object.rotation[0],
                           road_object.rotation[1],
                           road_object.rotation[2],
                           road_object.rotation[3],
                           road_object.length,
                           road_object.width,
                           road_object.number_of_lanes,
                           road_object.right_border_bounding_object,
                           road_object.left_border_bounding_object)
        # Road type as string
        byte_data = road_object.road_type.encode('ascii')
        msg += struct.pack('I', len(byte_data))
        msg += struct.pack(str(len(byte_data)) + 's', byte_data)
        # Additional Road Parameters to be used in proto settings. As string.
        num_params = len(road_object.extra_road_parameters)
        msg += struct.pack('B', num_params)
        for (par_name, par_val) in road_object.extra_road_parameters:
            byte_data = par_name.encode('ascii')
            msg += struct.pack('I', len(byte_data))
            msg += struct.pack(str(len(byte_data)) + 's', byte_data)
            byte_data = par_val.encode('ascii')
            msg += struct.pack('I', len(byte_data))
            msg += struct.pack(str(len(byte_data)) + 's', byte_data)
        return msg

    def interpret_add_road_to_simulation_command(self, msg):
        cur_msg_index = 0
        (item_type, ) = struct.unpack('B', msg[cur_msg_index:cur_msg_index + struct.calcsize('B')])
        cur_msg_index += struct.calcsize('B')
        obj = None
        if item_type == self.S_ROAD:
            if self.debug_mode:
                print("Road")
            obj = WebotsRoad()
            (obj.position[0],
             obj.position[1],
             obj.position[2],
             obj.rotation[0],
             obj.rotation[1],
             obj.rotation[2],
             obj.rotation[3],
             obj.length,
             obj.width,
             obj.number_of_lanes,
             obj.right_border_bounding_object,
             obj.left_border_bounding_object) = \
                struct.unpack('dddddddddB??', msg[cur_msg_index:cur_msg_index + struct.calcsize('dddddddddB??')])
            cur_msg_index += struct.calcsize('dddddddddB??')
            if self.debug_mode:
                print("SimulationMessageInterface: road length: {}".format(obj.length))
            # Read road type as string
            ((obj.road_type, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
            # Read Road Parameters to be used in proto settings
            ((num_of_params, ), cur_msg_index) = get_gata_from_struct('B', msg, cur_msg_index)
            if self.debug_mode:
                print("SimulationMessageInterface: Adding Road: numOf Road Params {}".format(
                    num_of_params))
            for _ in range(num_of_params):
                ((param_name_str, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
                ((param_str, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
                obj.extra_road_parameters.append((param_name_str, param_str))

        else:
            print("SimulationMessageInterface: Unknown SURROUNDINGS DEF {}".format(item_type))
        return obj

    def generate_add_road_disturbance_to_simulation_command(self, road_disturbance, is_create):
        """Creates a command to add the given road disturbance to the simulation.
        is_create: True: the object will be generated, False: it is already in the world."""
        if is_create:
            cmd = self.ROAD_DISTURBANCE_ADD
        else:
            cmd = self.ROAD_DISTURBANCE_DEF
        msg = struct.pack('B', cmd)
        msg += struct.pack('dddddddddddd',
                           road_disturbance.position[0],
                           road_disturbance.position[1],
                           road_disturbance.position[2],
                           road_disturbance.rotation[0],
                           road_disturbance.rotation[1],
                           road_disturbance.rotation[2],
                           road_disturbance.rotation[3],
                           road_disturbance.length,
                           road_disturbance.width,
                           road_disturbance.height,
                           road_disturbance.surface_height,
                           road_disturbance.inter_object_spacing)
        msg += struct.pack('II',
                           road_disturbance.disturbance_id,
                           road_disturbance.disturbance_type)
        return msg

    def interpret_add_road_disturbance_to_simulation_command(self, msg):
        cur_msg_index = 0
        if self.debug_mode:
            print("Road Disturbance")
        obj = WebotsRoadDisturbance()
        (obj.position[0],
            obj.position[1],
            obj.position[2],
            obj.rotation[0],
            obj.rotation[1],
            obj.rotation[2],
            obj.rotation[3],
            obj.length,
            obj.width,
            obj.height,
            obj.surface_height,
            obj.inter_object_spacing) = \
            struct.unpack('dddddddddddd', msg[cur_msg_index:cur_msg_index + struct.calcsize('dddddddddddd')])
        cur_msg_index += struct.calcsize('dddddddddddd')
        (obj.disturbance_id, obj.disturbance_type) = \
            struct.unpack('II', msg[cur_msg_index:cur_msg_index+struct.calcsize('II')])
        cur_msg_index += struct.calcsize('II')
        if self.debug_mode:
            print("SimulationMessageInterface: road disturbance length: {}".format(obj.length))
        return obj

    def generate_add_fog_to_simulation_command(self, fog, is_create):
        """Creates a command to add the given fog to the simulation.
        is_create: True: the object will be generated, False: it is already in the world."""
        if is_create:
            cmd = self.FOG_ADD
        else:
            cmd = self.FOG_DEF
        msg = struct.pack('B', cmd)
        msg += struct.pack('dddd',
                           fog.color[0],
                           fog.color[1],
                           fog.color[2],
                           fog.visibility_range)
        byte_data = fog.fog_type.encode('ascii')
        msg += struct.pack('I', len(byte_data))
        msg += struct.pack(str(len(byte_data)) + 's', byte_data)
        return msg

    def interpret_add_fog_to_simulation_command(self, msg):
        cur_msg_index = 0
        if self.debug_mode:
            print("Road Disturbance")
        obj = WebotsFog()
        (obj.color[0],
            obj.color[1],
            obj.color[2],
            obj.visibility_range) = \
            struct.unpack('dddd', msg[cur_msg_index:cur_msg_index + struct.calcsize('dddd')])
        cur_msg_index += struct.calcsize('dddd')
        ((obj.fog_type, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
        if self.debug_mode:
            print("SimulationMessageInterface: Fog visibility range: {}".format(obj.visibility_range))
        return obj

    def generate_add_vehicle_to_simulation_command(self, vehicle_object, is_dummy, is_create):
        """Creates a command to add the given vehicle to the simulation.
        is_create: True: the vehicle object will be generated, False: it is already in the world."""
        if is_dummy and is_create:
            cmd = self.DUMMY_ACTORS_ADD
        elif is_dummy and not is_create:
            cmd = self.DUMMY_ACTORS_DEF
        elif not is_dummy and is_create:
            cmd = self.VUT_ADD
        else:
            cmd = self.VUT_DEF

        msg = struct.pack('B', cmd)
        # Vehicle main structure:
        msg += struct.pack('dddddddddd', vehicle_object.current_position[0],
                           vehicle_object.current_position[1],
                           vehicle_object.current_position[2],
                           vehicle_object.rotation[0],
                           vehicle_object.rotation[1],
                           vehicle_object.rotation[2],
                           vehicle_object.rotation[3],
                           vehicle_object.color[0],
                           vehicle_object.color[1],
                           vehicle_object.color[2])
        msg += struct.pack('I', vehicle_object.vhc_id)
        msg += struct.pack('?', vehicle_object.is_controller_name_absolute)
        byte_data = vehicle_object.vehicle_model.encode('ascii')
        msg += struct.pack('I', len(byte_data))
        msg += struct.pack(str(len(byte_data)) + 's', byte_data)
        byte_data = vehicle_object.controller.encode('ascii')
        msg += struct.pack('I', len(byte_data))
        msg += struct.pack(str(len(byte_data)) + 's', byte_data)
        # Vehicle Parameters to be used in proto settings. As string.
        num_params = len(vehicle_object.vehicle_parameters)
        msg += struct.pack('B', num_params)
        for (par_name, par_val) in vehicle_object.vehicle_parameters:
            byte_data = par_name.encode('ascii')
            msg += struct.pack('I', len(byte_data))
            msg += struct.pack(str(len(byte_data)) + 's', byte_data)
            byte_data = par_val.encode('ascii')
            msg += struct.pack('I', len(byte_data))
            msg += struct.pack(str(len(byte_data)) + 's', byte_data)
        # Controller Arguments other than vehicle type. As string.
        num_of_controller_arguments = len(vehicle_object.controller_arguments)
        msg += struct.pack('B', num_of_controller_arguments)
        for contr_arg in vehicle_object.controller_arguments:
            byte_data = contr_arg.encode('ascii')
            msg += struct.pack('I', len(byte_data))
            msg += struct.pack(str(len(byte_data)) + 's', byte_data)
        # Signals related to the vehicle:
        num_signals = len(vehicle_object.signal)
        msg += struct.pack('B', num_signals)
        for s in vehicle_object.signal:
            msg += struct.pack('BBBBh',
                               s.signal_type,
                               s.interpolation_type,
                               s.ref_index,
                               s.ref_field,
                               len(s.signal_values))
            for j in range(len(s.signal_values)):
                msg += struct.pack('d', s.signal_values[j])
            for j in range(len(s.ref_values)):
                msg += struct.pack('d', s.ref_values[j])
        # Sensors related to the vehicle:
        num_sensors = len(vehicle_object.sensor_array)
        msg += struct.pack('B', num_sensors)
        for sensor in vehicle_object.sensor_array:
            msg += struct.pack('B', sensor.sensor_location)
            byte_data = sensor.sensor_type.encode('ascii')
            msg += struct.pack('I', len(byte_data))
            msg += struct.pack(str(len(byte_data)) + 's', byte_data)
            for field in sensor.sensor_fields:
                byte_data = field.field_name.encode('ascii')
                msg += struct.pack('I', len(byte_data))
                msg += struct.pack(str(len(byte_data)) + 's', byte_data)
                byte_data = field.field_val.encode('ascii')
                msg += struct.pack('I', len(byte_data))
                msg += struct.pack(str(len(byte_data)) + 's', byte_data)
            # Length of next field name is 0. This finishes the sensor message:
            msg += struct.pack('I', 0)
        # Controller parameters related to the vehicle:
        num_of_controller_parameters = len(vehicle_object.controller_parameters)
        msg += struct.pack('B', num_of_controller_parameters)
        for c_param in vehicle_object.controller_parameters:
            msg += self.controller_comm_interface.generate_controller_parameter_message(
                c_param.parameter_name,
                c_param.parameter_data)
        return msg

    def interpret_add_vehicle_to_simulation_command(self, msg, is_dummy):
        cur_msg_index = 0
        obj = WebotsVehicle()
        (obj.current_position[0],
         obj.current_position[1],
         obj.current_position[2],
         obj.rotation[0],
         obj.rotation[1],
         obj.rotation[2],
         obj.rotation[3],
         obj.color[0],
         obj.color[1],
         obj.color[2]) = struct.unpack('dddddddddd', msg[cur_msg_index:cur_msg_index + struct.calcsize('dddddddddd')])
        cur_msg_index += struct.calcsize('dddddddddd')
        ((obj.vhc_id, ), cur_msg_index) = get_gata_from_struct('I', msg, cur_msg_index)
        if is_dummy:
            obj.def_name = "DVHC_" + str(obj.vhc_id)
        else:
            obj.def_name = "VUT_" + str(obj.vhc_id)

        ((obj.is_controller_name_absolute, ), cur_msg_index) = get_gata_from_struct('?', msg, cur_msg_index)
        ((vhc_model, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
        vhc_model = vhc_model.strip()  # Remove space characters
        obj.set_vehicle_model(vhc_model)
        if self.debug_mode:
            print("SimulationMessageInterface : Adding vhc: model: {}, \
                   length: {} ind: {}".format(obj.vehicle_model, len(obj.vehicle_model), cur_msg_index))

        ((obj.controller, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
        obj.controller = obj.controller.strip()  # Remove space characters
        if self.debug_mode:
            print("SimulationMessageInterface : Adding vhc: Controller: {}, \
                   length: {} ind: {}".format(obj.controller, len(obj.controller), cur_msg_index))

        # Read Vehicle Parameters to be used in proto settings
        ((num_of_params, ), cur_msg_index) = get_gata_from_struct('B', msg, cur_msg_index)
        if self.debug_mode:
            print("SimulationMessageInterface: Adding vhc: numOf vehicle Params {}".format(
                num_of_params))
        for _ in range(num_of_params):
            ((param_name_str, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
            ((param_str, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
            obj.vehicle_parameters.append((param_name_str, param_str))

        # Read Controller Arguments additional to vehicle type
        ((num_of_params, ), cur_msg_index) = get_gata_from_struct('B', msg, cur_msg_index)
        if self.debug_mode:
            print("SimulationMessageInterface: \
                   Adding vhc: num_of_contr_arguments {}".format(num_of_params))
        for i in range(num_of_params):
            ((param_str, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
            obj.controller_arguments.append(param_str)

        # Read signals
        ((num_of_signals, ), cur_msg_index) = get_gata_from_struct('B', msg, cur_msg_index)
        obj.signal = []
        if self.debug_mode:
            print("SimulationMessageInterface: \
                    Adding vhc: num_of_signals {}".format(num_of_signals))
        for i in range(num_of_signals):
            ((signal_type,
              interpolation_type,
              signal_ref_index,
              signal_ref_field,
              signal_val_count), cur_msg_index) = get_gata_from_struct('BBBBh', msg, cur_msg_index)
            signal_values = []
            reference_values = []
            for _ in range(0, signal_val_count):
                ((sig_val, ), cur_msg_index) = get_gata_from_struct('d', msg, cur_msg_index)
                signal_values.append(sig_val)
            for _ in range(0, signal_val_count):
                ((ref_val, ), cur_msg_index) = get_gata_from_struct('d', msg, cur_msg_index)
                reference_values.append(ref_val)
            obj.signal.append(STaliroSignal(signal_type,
                                            interpolation_type,
                                            signal_ref_index,
                                            signal_ref_field,
                                            signal_values,
                                            reference_values))
            if self.debug_mode:
                print("SimulationMessageInterface: Added Signal")

        # Read Sensors
        ((num_of_sensors, ), cur_msg_index) = get_gata_from_struct('B', msg, cur_msg_index)
        if self.debug_mode:
            print("SimulationMessageInterface: \
                   Adding vhc: num_of_sensors {}".format(num_of_sensors))
        obj.sensor_array = [WebotsSensor() for sens_ind in range(num_of_sensors)]
        for sens_ind in range(num_of_sensors):
            ((obj.sensor_array[sens_ind].sensor_location, ), cur_msg_index) = \
                get_gata_from_struct('B', msg, cur_msg_index)
            ((obj.sensor_array[sens_ind].sensor_type, ), cur_msg_index) = \
                get_gata_from_struct('string', msg, cur_msg_index)
            ((temp_field_name, ), cur_msg_index) = \
                get_gata_from_struct('string', msg, cur_msg_index)
            while temp_field_name:
                ((temp_field_val, ), cur_msg_index) = \
                    get_gata_from_struct('string', msg, cur_msg_index)
                obj.sensor_array[sens_ind].add_sensor_field(temp_field_name, temp_field_val)
                ((temp_field_name, ), cur_msg_index) = \
                    get_gata_from_struct('string', msg, cur_msg_index)

        # Read Controller Parameters (NOT arguments!)
        ((num_of_control_params, ), cur_msg_index) = \
            get_gata_from_struct('B', msg, cur_msg_index)
        obj.controller_parameters = []
        if self.debug_mode:
            print("SimulationMessageInterface: \
                   Adding vhc : num_of_control_params {}".format(num_of_control_params))
        for i in range(num_of_control_params):
            (controller_param, param_msg_size) = \
                self.controller_comm_interface.interpret_controller_parameter_message(
                    msg[cur_msg_index:])
            cur_msg_index += param_msg_size
            controller_param.set_vehicle_id(obj.vhc_id)
            obj.controller_parameters.append(controller_param)
            if self.debug_mode:
                print("SimulationMessageInterface: Added Controller Parameter.")
        print('obj.controller : {}'.format(obj.controller))
        return obj

    def generate_change_vhc_position_command(self, vehicle_object):
        """Creates a command to change the given vehicle's pose."""

        cmd = self.CHANGE_VHC_POSITION

        msg = struct.pack('B', cmd)
        # Vehicle main structure:
        msg += struct.pack('ddddddd', vehicle_object.current_position[0],
                           vehicle_object.current_position[1],
                           vehicle_object.current_position[2],
                           vehicle_object.rotation[0],
                           vehicle_object.rotation[1],
                           vehicle_object.rotation[2],
                           vehicle_object.rotation[3])
        msg += struct.pack('I', vehicle_object.vhc_id)

        return msg

    def interpret_change_vhc_position_command(self, msg):
        cur_msg_index = 0
        obj = WebotsVehicle()
        (obj.current_position[0],
         obj.current_position[1],
         obj.current_position[2],
         obj.rotation[0],
         obj.rotation[1],
         obj.rotation[2],
         obj.rotation[3]) = struct.unpack('ddddddd', msg[cur_msg_index:cur_msg_index + struct.calcsize('ddddddd')])
        cur_msg_index += struct.calcsize('ddddddd')
        ((obj.vhc_id,), cur_msg_index) = get_gata_from_struct('I', msg, cur_msg_index)

        return obj

    def generate_add_pedestrian_to_simulation_command(self, pedestrian_object, is_create):
        """Creates a command to add the given pedestrian to the simulation.
        is_create: True: the pedestrian will be created, False: it is already in the world."""
        if is_create:
            cmd = self.PEDESTRIAN_ADD
        else:
            cmd = self.PEDESTRIAN_DEF
        msg = struct.pack('B', cmd)
        msg += struct.pack('ddddddddddddddddd',
                           pedestrian_object.current_position[0],
                           pedestrian_object.current_position[1],
                           pedestrian_object.current_position[2],
                           pedestrian_object.rotation[0],
                           pedestrian_object.rotation[1],
                           pedestrian_object.rotation[2],
                           pedestrian_object.rotation[3],
                           pedestrian_object.shirt_color[0],
                           pedestrian_object.shirt_color[1],
                           pedestrian_object.shirt_color[2],
                           pedestrian_object.pants_color[0],
                           pedestrian_object.pants_color[1],
                           pedestrian_object.pants_color[2],
                           pedestrian_object.shoes_color[0],
                           pedestrian_object.shoes_color[1],
                           pedestrian_object.shoes_color[2],
                           pedestrian_object.target_speed)
        msg += struct.pack('I', pedestrian_object.ped_id)
        msg += struct.pack('I', len(pedestrian_object.trajectory))
        for traj_point in pedestrian_object.trajectory:
            msg += struct.pack('d', traj_point)
        msg += struct.pack('I', len(pedestrian_object.controller))
        msg += struct.pack(str(len(pedestrian_object.controller)+1)+'s', pedestrian_object.controller.encode('ascii'))
        return msg

    def interpret_add_pedestrian_command(self, msg):
        cur_msg_index = 0
        obj = WebotsPedestrian()
        (obj.current_position[0],
         obj.current_position[1],
         obj.current_position[2],
         obj.rotation[0],
         obj.rotation[1],
         obj.rotation[2],
         obj.rotation[3],
         obj.shirt_color[0],
         obj.shirt_color[1],
         obj.shirt_color[2],
         obj.pants_color[0],
         obj.pants_color[1],
         obj.pants_color[2],
         obj.shoes_color[0],
         obj.shoes_color[1],
         obj.shoes_color[2],
         obj.target_speed) = \
            struct.unpack('ddddddddddddddddd',
                          msg[cur_msg_index:cur_msg_index + struct.calcsize('ddddddddddddddddd')])
        cur_msg_index += struct.calcsize('ddddddddddddddddd')
        ((obj.ped_id, ), cur_msg_index) = get_gata_from_struct('I', msg, cur_msg_index)
        obj.def_name = "PED_" + str(obj.ped_id)
        (trajectory_len, ) = struct.unpack('I', msg[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        for _ in range(trajectory_len):
            (traj_point, ) = struct.unpack('d', msg[cur_msg_index:cur_msg_index + struct.calcsize('d')])
            cur_msg_index += struct.calcsize('d')
            obj.trajectory.append(traj_point)
        (controller_len, ) = struct.unpack('I', msg[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        obj.controller = msg[cur_msg_index:cur_msg_index + controller_len + 1]
        obj.controller = obj.controller.decode('ascii')
        obj.controller = obj.controller.rstrip(' \t\r\n\0')  # Remove null characters at the end
        obj.controller = obj.controller.strip()  # Remove space characters
        return obj

    def generate_add_object_to_simulation_command(self, simulation_object):
        """Creates a command to add the object description to the simulation."""
        cmd = self.ADD_OBJECT_TO_SIMULATION
        msg = struct.pack('B', cmd)
        # Object def name as string
        byte_data = simulation_object.def_name.encode('ascii')
        msg += struct.pack('I', len(byte_data))
        msg += struct.pack(str(len(byte_data)) + 's', byte_data)
        # Object name as string
        byte_data = simulation_object.object_name.encode('ascii')
        msg += struct.pack('I', len(byte_data))
        msg += struct.pack(str(len(byte_data)) + 's', byte_data)
        # Object Parameters to be used in proto settings. As string.
        num_params = len(simulation_object.object_parameters)
        msg += struct.pack('B', num_params)
        for (par_name, par_val) in simulation_object.object_parameters:
            byte_data = par_name.encode('ascii')
            msg += struct.pack('I', len(byte_data))
            msg += struct.pack(str(len(byte_data)) + 's', byte_data)
            byte_data = par_val.encode('ascii')
            msg += struct.pack('I', len(byte_data))
            msg += struct.pack(str(len(byte_data)) + 's', byte_data)
        return msg

    def interpret_add_object_to_simulation_command(self, msg):
        """Interprets ADD_OBJECT_TO_SIMULATION command to add a generic object."""
        obj = WebotsSimObject()
        cur_msg_index = 0
        ((obj.def_name, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
        ((obj.object_name, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
        ((num_of_params, ), cur_msg_index) = get_gata_from_struct('B', msg, cur_msg_index)
        if self.debug_mode:
            print("SimulationMessageInterface: \
                   Adding generic object: {} num_of_parameters {}".format(obj.object_name, num_of_params))
        for i in range(num_of_params):
            ((param_name_str, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
            ((param_str, ), cur_msg_index) = get_gata_from_struct('string', msg, cur_msg_index)
            obj.object_parameters.append((param_name_str, param_str))
        return obj

    def generate_set_controller_parameter_command(self,
                                                  vhc_id=0,
                                                  parameter_name='N/A',
                                                  parameter_data=None):
        """Generates SET_CONTROLLER_PARAMETER command
        to set a robot controller parameter in webots."""
        command = struct.pack('B', self.SET_CONTROLLER_PARAMETER)
        command += struct.pack('I', vhc_id)
        command += \
            self.controller_comm_interface.generate_controller_parameter_message(
                parameter_name=parameter_name,
                parameter_data=parameter_data)
        return command

    def interpret_set_controller_parameter_command(self, message):
        cur_msg_index = 0
        (vhc_id, ) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (data, _data_length) = \
            self.controller_comm_interface.interpret_controller_parameter_message(message[cur_msg_index:])
        data.vehicle_id = vhc_id
        return data

    def generate_add_detection_evaluation_config(self, det_eval_config):
        command = struct.pack('B', self.ADD_DET_EVAL_CONFIG)
        command += \
            self.controller_comm_interface.generate_set_detection_monitor_message(detection_monitor=det_eval_config)
        return command

    def interpret_add_detection_evaluation_config(self, message):
        cur_msg_index = 0
        (data, _data_length) = \
            self.controller_comm_interface.interpret_set_detection_monitor_message(message[cur_msg_index:])
        return data

    def generate_add_visibility_evaluation_config(self, visibility_eval_config):
        command = struct.pack('B', self.ADD_VISIBILITY_EVAL_CONFIG)
        command += \
            self.controller_comm_interface.generate_set_visibility_monitor_message(
                visibility_monitor=visibility_eval_config)
        return command

    def interpret_add_visibility_evaluation_config(self, message):
        cur_msg_index = 0
        (data, _data_length) = \
            self.controller_comm_interface.interpret_set_visibility_monitor_message(message[cur_msg_index:])
        return data

    def generate_set_periodic_reporting_command(self, report_type, entity_id, period):
        """Generates SET_PERIODIC_REPORTING command.
        It is used for supervisor to transmit related data over its transmitter to the vehicles."""
        command = struct.pack('B', self.SET_PERIODIC_REPORTING)
        command += struct.pack('B', report_type)
        command += struct.pack('I', entity_id)
        command += struct.pack('i', period)
        return command

    def interpret_set_periodic_reporting_command(self, message):
        """Interprets SET_PERIODIC_REPORTING command.
        It is used for supervisor to transmit related data over its transmitter to the vehicles."""
        cur_msg_index = 0
        (report_type, ) = struct.unpack('B', message[cur_msg_index:cur_msg_index+struct.calcsize('B')])
        cur_msg_index += struct.calcsize('B')
        (entity_id, ) = struct.unpack('I', message[cur_msg_index:cur_msg_index+struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (period, ) = struct.unpack('i', message[cur_msg_index:cur_msg_index+struct.calcsize('i')])
        cur_msg_index += struct.calcsize('i')
        return report_type, entity_id, period
