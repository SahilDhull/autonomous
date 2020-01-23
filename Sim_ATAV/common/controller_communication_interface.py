"""Defines ControllerCommunicationInterface and VehiclePosition classes
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import struct
from Sim_ATAV.simulation_control.webots_controller_parameter import WebotsControllerParameter
from Sim_ATAV.vehicle_control.controller_commons.perf_evaluation.detection_evaluation_config \
    import DetectionEvaluationConfig
from Sim_ATAV.vehicle_control.controller_commons.perf_evaluation.visibility_evaluator \
    import VisibilityConfig, VisibilitySensor


class ControllerCommunicationInterface(object):
    """ControllerCommunicationInterface class
    handles the messaging between supervisor and the vehicle controllers."""
    VHC_POSITION_MESSAGE = 1
    SET_CONTROLLER_PARAMETERS_MESSAGE = 2
    PEDESTRIAN_POSITION_MESSAGE = 3
    VHC_ROTATION_MESSAGE = 4
    PED_ROTATION_MESSAGE = 5
    VHC_BOX_CORNERS_MESSAGE = 6
    PED_BOX_CORNERS_MESSAGE = 7
    DET_PERF_MESSAGE = 8
    VHC_CONTROL_ACTION_MESSAGE = 9
    SET_DETECTION_MONITOR = 10
    DETECTION_EVALUATION_MESSAGE = 11
    SET_VISIBILITY_MONITOR = 12
    VISIBILITY_EVALUATION_MESSAGE = 13

    def __init__(self):
        self.message_backlog = []

    def interpret_message(self, message):
        """Interpret received controller message."""
        data = None
        data_size = 0
        (command, ) = struct.unpack('B', message[0:struct.calcsize('B')])
        if command == self.VHC_POSITION_MESSAGE:
            (data, data_size) = self.interpret_vehicle_position_message(message)
        elif command == self.PEDESTRIAN_POSITION_MESSAGE:
            (data, data_size) = self.interpret_pedestrian_position_message(message)
        elif command == self.SET_CONTROLLER_PARAMETERS_MESSAGE:
            (data, data_size) = self.interpret_set_controller_parameters_message(message)
        elif command == self.VHC_ROTATION_MESSAGE:
            (data, data_size) = self.interpret_vehicle_rotation_message(message)
        elif command == self.PED_ROTATION_MESSAGE:
            (data, data_size) = self.interpret_pedestrian_rotation_message(message)
        elif command == self.VHC_BOX_CORNERS_MESSAGE:
            (data, data_size) = self.interpret_vehicle_box_corners_message(message)
        elif command == self.PED_BOX_CORNERS_MESSAGE:
            (data, data_size) = self.interpret_pedestrian_box_corners_message(message)
        elif command == self.DET_PERF_MESSAGE:
            (data, data_size) = self.interpret_detection_perf_message(message)
        elif command == self.VHC_CONTROL_ACTION_MESSAGE:
            (data, data_size) = self.interpret_vhc_control_action_message(message)
        elif command == self.DETECTION_EVALUATION_MESSAGE:
            (data, data_size) = self.interpret_detection_evaluation_message(message)
        elif command == self.VISIBILITY_EVALUATION_MESSAGE:
            (data, data_size) = self.interpret_visibility_evaluation_message(message)
        elif command == self.SET_DETECTION_MONITOR:
            (data, data_size) = self.interpret_set_detection_monitor_message(message)
        elif command == self.SET_VISIBILITY_MONITOR:
            (data, data_size) = self.interpret_set_visibility_monitor_message(message)
        return command, data, data_size

    def receive_command(self, receiver):
        """Receive controller message from the receiver device."""
        command = None
        data = None
        data_size = 0
        message = []
        if receiver.getQueueLength() > 0:
            message = receiver.getData()
            if message:
                (command, data, data_size) = self.interpret_message(message)
            receiver.nextPacket()
        return command, data, len(message), data_size

    def receive_all_communication(self, receiver):
        """Receive all messages from the receiver device."""
        message = []
        while receiver.getQueueLength() > 0:
            message += receiver.getData()
            receiver.nextPacket()
        return bytes(message)

    def extract_all_commands_from_message(self, message):
        command_list = []
        cur_msg_index = 0
        msg_len = len(message)
        while cur_msg_index < msg_len - 1:
            (command, data, data_size) = self.interpret_message(message[cur_msg_index:])
            command_list.append((command, data))
            cur_msg_index += data_size
        return command_list

    def get_all_vehicle_positions(self, command_list):
        vhc_pos_dict = {}
        for ind in range(len(command_list)):
            (command, data) = command_list[ind]
            if command == self.VHC_POSITION_MESSAGE:
                vhc_pos_dict[data.vehicle_id] = data.position
        return vhc_pos_dict

    def get_all_pedestrian_positions(self, command_list):
        pos_dict = {}
        for ind in range(len(command_list)):
            (command, data) = command_list[ind]
            if command == self.PEDESTRIAN_POSITION_MESSAGE:
                pos_dict[data.pedestrian_id] = data.position
        return pos_dict

    def get_all_vehicle_rotations(self, command_list):
        rot_dict = {}
        for ind in range(len(command_list)):
            (command, data) = command_list[ind]
            if command == self.VHC_ROTATION_MESSAGE:
                rot_dict[data[0]] = data[1]
        return rot_dict

    def get_all_vehicle_box_corners(self, command_list):
        rot_dict = {}
        for ind in range(len(command_list)):
            (command, data) = command_list[ind]
            if command == self.VHC_BOX_CORNERS_MESSAGE:
                rot_dict[data[0]] = data[1]
        return rot_dict

    def get_all_pedestrian_rotations(self, command_list):
        rot_dict = {}
        for ind in range(len(command_list)):
            (command, data) = command_list[ind]
            if command == self.PED_ROTATION_MESSAGE:
                rot_dict[data[0]] = data[1]
        return rot_dict

    def get_all_pedestrian_box_corners(self, command_list):
        rot_dict = {}
        for ind in range(len(command_list)):
            (command, data) = command_list[ind]
            if command == self.PED_BOX_CORNERS_MESSAGE:
                rot_dict[data[0]] = data[1]
        return rot_dict

    def get_detection_performances(self, command_list, vhc_id):
        det_performances = []
        for ind in range(len(command_list)):
            (command, data) = command_list[ind]
            if command == self.DET_PERF_MESSAGE:
                object_index = data[0]
                object_type_text = data[1]
                det_perf = data[2]
                det_performances.append((object_index, object_type_text, det_perf))
        return det_performances

    def get_applied_vehicle_controls(self, command_list, vhc_id):
        control_actions = []
        for ind in range(len(command_list)):
            (command, data) = command_list[ind]
            if command == self.VHC_CONTROL_ACTION_MESSAGE:
                rcv_vhc_id = data[0]
                if rcv_vhc_id == vhc_id:
                    control_type = data[1]
                    control_action_value = data[2]
                    control_actions.append((control_type, control_action_value))
        return control_actions

    def get_detection_evaluations(self, commmand_list):
        detection_evals = []
        for ind in range(len(commmand_list)):
            (command, data) = commmand_list[ind]
            if command == self.DETECTION_EVALUATION_MESSAGE:
                detection_evals.append(data)
        return detection_evals

    def get_visibility_evaluations(self, commmand_list):
        visibility_evals = []
        for ind in range(len(commmand_list)):
            (command, data) = commmand_list[ind]
            if command == self.VISIBILITY_EVALUATION_MESSAGE:
                visibility_evals.append(data)
        return visibility_evals

    def generate_vehicle_position_message(self, vhc_id, vhc_position):
        """Generate a controller message with vehicle position info."""
        message = struct.pack('B', self.VHC_POSITION_MESSAGE)
        message += struct.pack('I', vhc_id)
        message += struct.pack('ddd', vhc_position[0], vhc_position[1], vhc_position[2])
        return message

    def transmit_vehicle_position_message(self, emitter, vhc_id, vhc_position):
        """Generate and send vehicle position message through emitter device."""
        if emitter is not None:
            message = self.generate_vehicle_position_message(vhc_id, vhc_position)
            emitter.send(message)

    def interpret_vehicle_position_message(self, message):
        """Extract vehicle position information from the received vehicle position message."""
        cur_msg_index = struct.calcsize('B')  # Command is already read
        vhc_position = [0.0, 0.0, 0.0]
        (vhc_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (vhc_position[0], vhc_position[1], vhc_position[2]) = \
            struct.unpack('ddd', message[cur_msg_index:cur_msg_index + struct.calcsize('ddd')])
        cur_msg_index += struct.calcsize('ddd')
        data = VehiclePosition(vhc_id=vhc_id, vhc_position=vhc_position)
        return data, cur_msg_index

    def generate_vehicle_rotation_message(self, vhc_id, vhc_rotation):
        """Generate a controller message with vehicle rotation info."""
        message = struct.pack('B', self.VHC_ROTATION_MESSAGE)
        message += struct.pack('I', vhc_id)
        message += struct.pack('ddddddddd',
                                vhc_rotation[0], vhc_rotation[1], vhc_rotation[2],
                                vhc_rotation[3], vhc_rotation[4], vhc_rotation[5],
                                vhc_rotation[6], vhc_rotation[7], vhc_rotation[8])
        return message

    def transmit_vehicle_rotation_message(self, emitter, vhc_id, vhc_rotation):
        """Generate and send vehicle rotation message through emitter device."""
        if emitter is not None:
            message = self.generate_vehicle_rotation_message(vhc_id, vhc_rotation)
            emitter.send(message)

    def interpret_vehicle_rotation_message(self, message):
        """Extract vehicle rotation information from the received message."""
        cur_msg_index = struct.calcsize('B')  # Command is already read
        vhc_rotation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        (vhc_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (vhc_rotation[0], vhc_rotation[1], vhc_rotation[2],
         vhc_rotation[3], vhc_rotation[4], vhc_rotation[5],
         vhc_rotation[6], vhc_rotation[7], vhc_rotation[8]) = \
            struct.unpack('ddddddddd', message[cur_msg_index:cur_msg_index + struct.calcsize('ddddddddd')])
        cur_msg_index += struct.calcsize('ddddddddd')
        data = (vhc_id, vhc_rotation)
        return data, cur_msg_index

    def generate_vehicle_box_corners_message(self, vhc_id, vhc_corners):
        """Generate a controller message with vehicle box_corners info."""
        message = struct.pack('B', self.VHC_BOX_CORNERS_MESSAGE)
        message += struct.pack('I', vhc_id)
        for c_ind in range(8):
            message += struct.pack('ddd', vhc_corners[c_ind][0], vhc_corners[c_ind][1], vhc_corners[c_ind][2])
        return message

    def transmit_vehicle_box_corners_message(self, emitter, vhc_id, vhc_corners):
        """Generate and send vehicle box_corners message through emitter device."""
        if emitter is not None:
            message = self.generate_vehicle_box_corners_message(vhc_id, vhc_corners)
            emitter.send(message)

    def interpret_vehicle_box_corners_message(self, message):
        """Extract vehicle box_corners information from the received message."""
        cur_msg_index = struct.calcsize('B')  # Command is already read
        (vhc_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        corners_dict = {}
        for c_ind in range(8):
            corners_dict[c_ind] = [0.0, 0.0, 0.0]
            (corners_dict[c_ind][0], corners_dict[c_ind][1], corners_dict[c_ind][2]) = \
                struct.unpack('ddd', message[cur_msg_index:cur_msg_index + struct.calcsize('ddd')])
            cur_msg_index += struct.calcsize('ddd')
        data = (vhc_id, corners_dict)
        return data, cur_msg_index

    def generate_pedestrian_position_message(self, ped_id, ped_position):
        """Generate a controller message with pedestrian position info."""
        message = struct.pack('B', self.PEDESTRIAN_POSITION_MESSAGE)
        message += struct.pack('I', ped_id)
        message += struct.pack('ddd', ped_position[0], ped_position[1], ped_position[2])
        return message

    def transmit_pedestrian_position_message(self, emitter, ped_id, ped_position):
        """Generate and send pedestrian position message through emitter device."""
        if emitter is not None:
            message = self.generate_pedestrian_position_message(ped_id, ped_position)
            emitter.send(message)

    def interpret_pedestrian_position_message(self, message):
        """Extract pedestrian position information from the received pedestrian position message."""
        cur_msg_index = struct.calcsize('B')  # Command is already read
        ped_position = [0.0, 0.0, 0.0]
        (ped_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (ped_position[0], ped_position[1], ped_position[2]) = \
            struct.unpack('ddd', message[cur_msg_index:cur_msg_index + struct.calcsize('ddd')])
        cur_msg_index += struct.calcsize('ddd')
        data = PedestrianPosition(ped_id=ped_id, ped_position=ped_position)
        return data, cur_msg_index

    def generate_pedestrian_rotation_message(self, pedestrian_id, rotation):
        """Generate a controller message with pedestrian rotation info."""
        message = struct.pack('B', self.PED_ROTATION_MESSAGE)
        message += struct.pack('I', pedestrian_id)
        message += struct.pack('ddddddddd',
                                rotation[0], rotation[1], rotation[2],
                                rotation[3], rotation[4], rotation[5],
                                rotation[6], rotation[7], rotation[8])
        return message

    def transmit_pedestrian_rotation_message(self, emitter, pedestrian_id, rotation):
        """Generate and send pedestrian rotation message through emitter device."""
        if emitter is not None:
            message = self.generate_pedestrian_rotation_message(pedestrian_id, rotation)
            emitter.send(message)

    def interpret_pedestrian_rotation_message(self, message):
        """Extract pedestrian rotation information from the received message."""
        cur_msg_index = struct.calcsize('B')  # Command is already read
        rotation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        (pedestrian_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (rotation[0], rotation[1], rotation[2],
         rotation[3], rotation[4], rotation[5],
         rotation[6], rotation[7], rotation[8]) = \
            struct.unpack('ddddddddd', message[cur_msg_index:cur_msg_index + struct.calcsize('ddddddddd')])
        cur_msg_index += struct.calcsize('ddddddddd')
        data = (pedestrian_id, rotation)
        return data, cur_msg_index

    def generate_pedestrian_box_corners_message(self, ped_id, ped_corners):
        """Generate a controller message with pedestrian box_corners info."""
        message = struct.pack('B', self.PED_BOX_CORNERS_MESSAGE)
        message += struct.pack('I', ped_id)
        for c_ind in range(8):
            message += struct.pack('ddd', ped_corners[c_ind][0], ped_corners[c_ind][1], ped_corners[c_ind][2])
        return message

    def transmit_pedestrian_box_corners_message(self, emitter, ped_id, ped_corners):
        """Generate and send pedestrian box_corners message through emitter device."""
        if emitter is not None:
            message = self.generate_pedestrian_box_corners_message(ped_id, ped_corners)
            emitter.send(message)

    def interpret_pedestrian_box_corners_message(self, message):
        """Extract pedestrian box_corners information from the received message."""
        cur_msg_index = struct.calcsize('B')  # Command is already read
        (ped_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        corners_dict = {}
        for c_ind in range(8):
            corners_dict[c_ind] = [0.0, 0.0, 0.0]
            (corners_dict[c_ind][0], corners_dict[c_ind][1], corners_dict[c_ind][2]) = \
                struct.unpack('ddd', message[cur_msg_index:cur_msg_index + struct.calcsize('ddd')])
            cur_msg_index += struct.calcsize('ddd')
        data = (ped_id, corners_dict)
        return data, cur_msg_index

    def interpret_detection_perf_message(self, message):
        cur_msg_index = struct.calcsize('B')  # Command is already read
        (obj_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')

        (length, ) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        if length > 0:
            (obj_type, ) = struct.unpack(str(length) + 's', message[cur_msg_index:cur_msg_index + length])
            obj_type = obj_type.decode('ascii')
            obj_type = obj_type.rstrip(' \t\r\n\0')
        else:
            obj_type = ''
        cur_msg_index += length

        (obj_det_perf,) = struct.unpack('d', message[cur_msg_index:cur_msg_index + struct.calcsize('d')])
        cur_msg_index += struct.calcsize('d')

        data = (obj_id, obj_type, obj_det_perf)
        return data, cur_msg_index

    def interpret_vhc_control_action_message(self, message):
        cur_msg_index = struct.calcsize('B')  # Command is already read
        (vhc_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')

        (control_type, ) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')

        (control_value,) = struct.unpack('d', message[cur_msg_index:cur_msg_index + struct.calcsize('d')])
        cur_msg_index += struct.calcsize('d')

        data = (vhc_id, control_type, control_value)
        return data, cur_msg_index

    def generate_detection_box_perf_message(self, obj_id, object_type_text, det_perf):
        message = struct.pack('B', self.DET_PERF_MESSAGE)
        message += struct.pack('I', obj_id)
        byte_data = object_type_text.encode('ascii')
        message += struct.pack('I', len(byte_data))
        message += struct.pack(str(len(byte_data)) + 's', byte_data)
        message += struct.pack('d', det_perf)
        return message

    def generate_control_action_message(self, vhc_id, control_type, control_value):
        message = struct.pack('B', self.VHC_CONTROL_ACTION_MESSAGE)
        message += struct.pack('I', vhc_id)
        message += struct.pack('I', control_type)
        message += struct.pack('d', control_value)
        return message

    def generate_detection_evaluation_message(self, idx, value):
        message = struct.pack('B', self.DETECTION_EVALUATION_MESSAGE)
        message += struct.pack('I', idx)
        message += struct.pack('d', value)
        return message

    def interpret_detection_evaluation_message(self, message):
        cur_msg_index = struct.calcsize('B')  # Command is already read
        (index, ) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (value, ) = struct.unpack('d', message[cur_msg_index:cur_msg_index + struct.calcsize('d')])
        cur_msg_index += struct.calcsize('d')
        data = (index, value)
        return data, cur_msg_index

    def generate_visibility_evaluation_message(self, idx, value):
        message = struct.pack('B', self.VISIBILITY_EVALUATION_MESSAGE)
        message += struct.pack('I', idx)
        message += struct.pack('d', value)
        return message

    def interpret_visibility_evaluation_message(self, message):
        cur_msg_index = struct.calcsize('B')  # Command is already read
        (index, ) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (value, ) = struct.unpack('d', message[cur_msg_index:cur_msg_index + struct.calcsize('d')])
        cur_msg_index += struct.calcsize('d')
        data = (index, value)
        return data, cur_msg_index

    def generate_controller_parameter_message(self, parameter_name='N/A', parameter_data=None):
        """Generates controller parameter message to be used inside add vehicle or
        set controller params messages.
        Message structure:
        Length of parameter name string(1),
        parameter name string(?),
        Parameter data type character(1),
        Length of parameter data(4),
        parameter data(?)"""
        message = struct.pack('I', len(parameter_name))
        message += struct.pack(str(len(parameter_name))+'s', parameter_name.encode('ascii'))
        data_type_name = 'x'
        data_length = 0
        if type(parameter_data) == list:
            data_length = len(parameter_data)
            if parameter_data:
                if type(parameter_data[0]) is bool:
                    data_type_name = '?'
                elif type(parameter_data[0]) is int:
                    data_type_name = 'I'
                elif type(parameter_data[0]) is float:
                    data_type_name = 'd'
        elif type(parameter_data) == str:
            data_length = len(parameter_data)
            data_type_name = 's'
        elif type(parameter_data) is bool:
            data_length = 1
            data_type_name = '?'
        elif type(parameter_data) is int:
            data_length = 1
            data_type_name = 'I'
        elif type(parameter_data) is float:
            data_length = 1
            data_type_name = 'd'
        message += struct.pack(str(len(data_type_name))+'s', data_type_name.encode('ascii'))
        message += struct.pack('I', data_length)
        pack_str = '%s{}'.format(data_type_name)
        message += struct.pack(pack_str % data_length, *parameter_data)
        return message

    def interpret_controller_parameter_message(self, message):
        """Extract controller parameter info from the received message."""
        cur_msg_index = 0
        (param_name_length, ) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        parameter_name = message[cur_msg_index:cur_msg_index + param_name_length]
        parameter_name = parameter_name.decode('ascii')
        parameter_name = parameter_name.rstrip(' \t\r\n\0')  # Remove null characters at the end
        cur_msg_index += param_name_length
        data_type_name = message[cur_msg_index:cur_msg_index + 1]
        data_type_name = data_type_name.decode('ascii')
        data_type_name = data_type_name.rstrip(' \t\r\n\0')  # Remove null characters at the end
        cur_msg_index += 1
        (data_length, ) = \
            struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        unpack_str = '%s{}'.format(data_type_name)
        parameter_data = list(struct.unpack(unpack_str % data_length,
                                            message[cur_msg_index:cur_msg_index +
                                                                  struct.calcsize(unpack_str % data_length)]))
        cur_msg_index += struct.calcsize(unpack_str % data_length)
        data = WebotsControllerParameter(parameter_name=parameter_name,
                                         parameter_data=parameter_data)
        return data, cur_msg_index

    def generate_set_controller_parameters_message(self,
                                                   vhc_id=0,
                                                   parameter_name='N/A',
                                                   parameter_data=None):
        """Generates SET_CONTROLLER_PARAMETERS message.
        Message structure:
        Command(1),
        Applicable vehicle id(4),
        Length of parameter name string(1),
        parameter name string(?),
        Parameter data type character(1),
        Length of parameter data(4),
        parameter data(?)"""
        message = struct.pack('B', self.SET_CONTROLLER_PARAMETERS_MESSAGE)
        message += struct.pack('I', int(vhc_id))
        message += self.generate_controller_parameter_message(parameter_name, parameter_data)
        return message

    def transmit_set_controller_parameters_message(self,
                                                   emitter,
                                                   vhc_id=0,
                                                   parameter_name='N/A',
                                                   parameter_data=None):
        """Generate and transmit SET_CONTROLLER_PARAMETERS message through the emitter device."""
        message = self.generate_set_controller_parameters_message(vhc_id=vhc_id,
                                                                  parameter_name=parameter_name,
                                                                  parameter_data=parameter_data)
        if emitter is not None:
            emitter.send(message)
        else:
            self.message_backlog.append(message)

    def interpret_set_controller_parameters_message(self, message):
        """Extract the SET_CONTROLLER_PARAMETERS message from the received message."""
        cur_msg_index = struct.calcsize('B')
        (vhc_id, ) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (data, data_size) = self.interpret_controller_parameter_message(message[cur_msg_index:])
        data.set_vehicle_id(vhc_id)
        return data, data_size + struct.calcsize('I') + struct.calcsize('B')

    def generate_set_detection_monitor_message(self, detection_monitor):
        message = struct.pack('B', self.SET_DETECTION_MONITOR)
        message += struct.pack('I', detection_monitor.vehicle_id)
        message += struct.pack('I', len(detection_monitor.sensor_type))
        message += struct.pack(str(len(detection_monitor.sensor_type)) + 's',
                               detection_monitor.sensor_type.encode('ascii'))
        message += struct.pack('I', detection_monitor.sensor_id)
        message += struct.pack('I', len(detection_monitor.eval_type))
        message += struct.pack(str(len(detection_monitor.eval_type)) + 's', detection_monitor.eval_type.encode('ascii'))
        message += struct.pack('I', len(detection_monitor.eval_alg))
        message += struct.pack(str(len(detection_monitor.eval_alg)) + 's', detection_monitor.eval_alg.encode('ascii'))
        message += struct.pack('I', len(detection_monitor.target_objs))
        for target_obj in detection_monitor.target_objs:
            message += struct.pack('I', len(target_obj[0]))
            message += struct.pack(str(len(target_obj[0])) + 's', target_obj[0].encode('ascii'))
            message += struct.pack('I', target_obj[1])
        return message

    def interpret_set_detection_monitor_message(self, message):
        cur_msg_index = struct.calcsize('B')
        (vhc_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (param_name_length,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        sensor_type = message[cur_msg_index:cur_msg_index + param_name_length]
        sensor_type = sensor_type.decode('ascii')
        sensor_type = sensor_type.rstrip(' \t\r\n\0')  # Remove null characters at the end
        cur_msg_index += param_name_length
        (sensor_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (param_name_length,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        eval_type = message[cur_msg_index:cur_msg_index + param_name_length]
        eval_type = eval_type.decode('ascii')
        eval_type = eval_type.rstrip(' \t\r\n\0')  # Remove null characters at the end
        cur_msg_index += param_name_length
        (param_name_length,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        eval_alg = message[cur_msg_index:cur_msg_index + param_name_length]
        eval_alg = eval_alg.decode('ascii')
        eval_alg = eval_alg.rstrip(' \t\r\n\0')  # Remove null characters at the end
        cur_msg_index += param_name_length
        (num_of_objects,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        object_list = []
        for i in range(num_of_objects):
            (param_name_length,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
            cur_msg_index += struct.calcsize('I')
            object_type = message[cur_msg_index:cur_msg_index + param_name_length]
            object_type = object_type.decode('ascii')
            object_type = object_type.rstrip(' \t\r\n\0')  # Remove null characters at the end
            cur_msg_index += param_name_length
            (object_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
            cur_msg_index += struct.calcsize('I')
            object_list.append((object_type, object_id))
        data = DetectionEvaluationConfig(vehicle_id=vhc_id,
                                         sensor_type=sensor_type,
                                         sensor_id=sensor_id,
                                         target_objs=object_list,
                                         eval_type=eval_type,
                                         eval_alg=eval_alg)
        return data, cur_msg_index

    def transmit_set_detection_monitor_message(self, emitter, detection_monitor):
        """Generate and transmit SET_DET_EVAL_CONFIG message through the emitter device."""
        message = self.generate_set_detection_monitor_message(detection_monitor)
        if emitter is not None:
            emitter.send(message)
        else:
            self.message_backlog.append(message)

    def generate_set_visibility_monitor_message(self, visibility_monitor):
        message = struct.pack('B', self.SET_VISIBILITY_MONITOR)
        message += struct.pack('I', visibility_monitor.vehicle_id)
        message += struct.pack('d', visibility_monitor.sensor.local_position[0])
        message += struct.pack('d', visibility_monitor.sensor.local_position[1])
        message += struct.pack('d', visibility_monitor.sensor.local_position[2])
        message += struct.pack('d', visibility_monitor.sensor.x_rotation)
        message += struct.pack('d', visibility_monitor.sensor.max_range)
        message += struct.pack('d', visibility_monitor.sensor.hor_fov)
        message += struct.pack('I', len(visibility_monitor.object_list))
        for obj in visibility_monitor.object_list:
            obj_type = obj[VisibilityConfig.OBJ_TYPE_IND]
            message += struct.pack('I', len(obj_type))
            message += struct.pack(str(len(obj_type)) + 's', obj_type.encode('ascii'))
            message += struct.pack('I', obj[VisibilityConfig.OBJ_ID_IND])
        return message

    def interpret_set_visibility_monitor_message(self, message):
        obj_list = []
        cur_msg_index = struct.calcsize('B')
        (vhc_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        (sensor_pos_x, sensor_pos_y, sensor_pos_z) = struct.unpack('ddd', message[cur_msg_index:cur_msg_index + struct.calcsize('ddd')])
        cur_msg_index += struct.calcsize('ddd')
        (sensor_rotation_x,) = struct.unpack('d', message[cur_msg_index:cur_msg_index + struct.calcsize('d')])
        cur_msg_index += struct.calcsize('d')
        (sensor_range,) = struct.unpack('d', message[cur_msg_index:cur_msg_index + struct.calcsize('d')])
        cur_msg_index += struct.calcsize('d')
        (sensor_fov,) = struct.unpack('d', message[cur_msg_index:cur_msg_index + struct.calcsize('d')])
        cur_msg_index += struct.calcsize('d')

        (obj_count,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
        cur_msg_index += struct.calcsize('I')
        for i in range(obj_count):
            (param_name_length,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
            cur_msg_index += struct.calcsize('I')
            obj_type = message[cur_msg_index:cur_msg_index + param_name_length]
            obj_type = obj_type.decode('ascii')
            obj_type = obj_type.rstrip(' \t\r\n\0')  # Remove null characters at the end
            cur_msg_index += param_name_length
            (obj_id,) = struct.unpack('I', message[cur_msg_index:cur_msg_index + struct.calcsize('I')])
            cur_msg_index += struct.calcsize('I')
            obj_list.append([None, None])  # We first append two None type and fix them using correct indices.
            obj_list[-1][VisibilityConfig.OBJ_ID_IND] = obj_id
            obj_list[-1][VisibilityConfig.OBJ_TYPE_IND] = obj_type
        data = VisibilityConfig(sensor=VisibilitySensor(hor_fov=sensor_fov,
                                                        max_range=sensor_range,
                                                        position=(sensor_pos_x, sensor_pos_y, sensor_pos_z),
                                                        x_rotation=sensor_rotation_x),
                                object_list=obj_list,
                                vehicle_id=vhc_id)
        return data, cur_msg_index

    def transmit_set_visibility_monitor_message(self, emitter, visibility_monitor):
        """Generate and transmit SET_VISIBILITY_MONITOR message through the emitter device."""
        message = self.generate_set_visibility_monitor_message(visibility_monitor)
        if emitter is not None:
            emitter.send(message)
        else:
            self.message_backlog.append(message)

    def transmit_backlogged_messages(self, emitter):
        """Transmit backlogged messages through the emitter device."""
        if emitter is not None:
            for message in self.message_backlog:
                emitter.send(message)
            self.message_backlog = []


class VehiclePosition(object):
    """VehiclePosition class is a data structure to relate vehicle id to vehicle position."""
    def __init__(self, vhc_id=None, vhc_position=None):
        if vhc_id is None:
            self.vehicle_id = 0
        else:
            self.vehicle_id = vhc_id
        if vhc_position is None:
            self.position = [0.0, 0.0, 0.0]
        else:
            self.position = vhc_position

    def set_vehicle_id(self, vhc_id):
        """Set vehicle id field."""
        self.vehicle_id = vhc_id

    def get_vehicle_id(self):
        """Get vehicle id field."""
        return self.vehicle_id

    def set_vehicle_position(self, position):
        """Set vehicle position field."""
        self.position = position

    def get_vehicle_position(self):
        """Get vehicle position field."""
        return self.position


class PedestrianPosition(object):
    """PedestrianPosition class is a data structure to relate pedestrian id to its position."""
    def __init__(self, ped_id=None, ped_position=None):
        if ped_id is None:
            self.pedestrian_id = 0
        else:
            self.pedestrian_id = ped_id
        if ped_position is None:
            self.position = [0.0, 0.0, 0.0]
        else:
            self.position = ped_position

    def set_pedestrian_id(self, vhc_id):
        """Set pedestrian_id field."""
        self.pedestrian_id = vhc_id

    def get_pedestrian_id(self):
        """Get pedestrian_id field."""
        return self.pedestrian_id

    def set_pedestrian_position(self, position):
        """Set pedestrian position field."""
        self.position = position

    def get_pedestrian_position(self):
        """Get pedestrian position field."""
        return self.position
