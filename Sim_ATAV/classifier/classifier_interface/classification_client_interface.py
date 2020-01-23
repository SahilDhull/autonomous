"""Defines the ClassificationClientInterface class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------

"""

import struct
import mmap
from Sim_ATAV.classifier.classifier_interface.communication_client import CommunicationClient
from Sim_ATAV.classifier.classifier_interface.classification_command import ClassificationCommand


class ClassificationClientInterface(object):
    """ClassificationClientInterface provides the methods for communicating with a
    classification server."""

    def __init__(self, is_debug_mode=False):
        self.is_debug_mode = is_debug_mode
        self.shared_file_name = "Local\\WebotsCameraImage"
        self.shared_file_size = 1437696
        self.server_ip = "localhost"
        self.server_port = 10101
        self.max_connection_retry_count = 10
        self.shared_mem = None
        self.comm_client = None
        self.current_frame_id = 0
        self.max_frame_id = 2**31 - 1

    def connect_to_classification_server(self):
        """Connect to the classification server."""
        self.comm_client = CommunicationClient(self.server_ip, self.server_port)
        self.comm_client.connect_to_server(self.max_connection_retry_count)

    def setup_shared_memory(self, file_name=None, file_size=None):
        """Create a shared memory file and send the file details to the server."""
        if file_name is not None:
            self.shared_file_name = file_name
        if file_size is not None:
            self.shared_file_size = file_size
        self.shared_mem = mmap.mmap(0,
                                    self.shared_file_size,
                                    self.shared_file_name,
                                    mmap.ACCESS_WRITE)
        self.send_shared_memory_settings()

    def send_shared_memory_settings(self):
        """ Send Shared Memory Settings to the server."""
        command = struct.pack('B', ClassificationCommand.CMD_START_SHARED_MEMORY)
        command += struct.pack('II', self.shared_file_size, len(self.shared_file_name)+1)
        command += struct.pack(str(len(self.shared_file_name)+1)+'s', self.shared_file_name.encode('utf-8'))
        self.comm_client.send_to_server(command)
        recv_msg = self.comm_client.receive_blocking()
        (response, ) = struct.unpack('B', recv_msg[0:struct.calcsize('B')])
        if response != ClassificationCommand.ACK:
            print("Shared Memory Could not be set!")

    def classify_data(self, data, width, height):
        """Put the given image to the shared memory and ask server to classify it.
        Return classification results."""
        self.write_to_shared_memory(data)
        return self.get_classification_from_server(width, height)

    def write_to_shared_memory(self, data):
        """Write given data to the shared memory."""
        self.shared_mem.seek(0)
        self.shared_mem.write(data)
        self.shared_mem.flush()

    def get_classification_from_server(self, width, height):
        """Send CLASSIFY command to the server and return the received response."""
        if self.current_frame_id >= self.max_frame_id:
            self.current_frame_id = 1
        else:
            self.current_frame_id += 1
        command = struct.pack('B', ClassificationCommand.CMD_CLASSIFY)
        command += struct.pack('III', self.current_frame_id, width, height)
        self.comm_client.send_to_server(command)
        recv_msg = self.comm_client.receive_blocking()
        (detection_boxes, detection_probs, detection_classes) = \
            self.interpret_classification_response(recv_msg)
        return detection_boxes, detection_probs, detection_classes

    def close_shared_memory(self):
        """Close the shared memory."""
        self.shared_mem.close()
        self.shared_mem = None
        command = struct.pack('B', ClassificationCommand.CMD_CLOSE_SHARED_MEMORY)
        self.comm_client.send_to_server(command)
        recv_msg = self.comm_client.receive_blocking()
        (response, ) = struct.unpack('B', recv_msg[0:struct.calcsize('B')])
        if response != ClassificationCommand.ACK:
            print("Shared Memory Could not be closed on server!")

    def end_communication_with_server(self):
        """Ask server to end communications at its side and close communication at this side."""
        command = struct.pack('B', ClassificationCommand.CMD_END_COMMUNICATION)
        self.comm_client.send_to_server(command)
        recv_msg = self.comm_client.receive_blocking()
        (response, ) = struct.unpack('B', recv_msg[0:struct.calcsize('B')])
        if response != ClassificationCommand.ACK:
            print("Communication could not be ended!")
        self.comm_client.disconnect_from_server()
        self.comm_client = None

    def interpret_classification_response(self, recv_msg):
        """Interpret the returned classification response.
        Return the success status, number of detected objects,
        detection boxes, detection probabilities, detection classes"""
        detection_boxes = []
        detection_probs = []
        detection_classes = []
        (success, ) = struct.unpack('B', recv_msg[0:struct.calcsize('B')])
        cur_msg_index = struct.calcsize('B')
        if success == ClassificationCommand.CMD_CLASSIFICATION_RESULT:
            (recv_frame_id, ) = \
                struct.unpack('I', recv_msg[cur_msg_index:cur_msg_index+struct.calcsize('I')])
            cur_msg_index += struct.calcsize('I')
            (num_of_detected_objects, ) = \
                struct.unpack('I', recv_msg[cur_msg_index:cur_msg_index+struct.calcsize('I')])
            cur_msg_index += struct.calcsize('I')
            for obj_ind in range(num_of_detected_objects):
                det_box = [0, 0, 0, 0]
                (det_box[0],
                 det_box[1],
                 det_box[2],
                 det_box[3]) = \
                    struct.unpack('ffff',
                                  recv_msg[cur_msg_index:cur_msg_index+struct.calcsize('ffff')])
                cur_msg_index += struct.calcsize('ffff')
                detection_boxes.append(det_box)
                (det_prob, ) = \
                    struct.unpack('f', recv_msg[cur_msg_index:cur_msg_index+struct.calcsize('f')])
                cur_msg_index += struct.calcsize('f')
                detection_probs.append(det_prob)
                (det_class, ) = \
                    struct.unpack('I', recv_msg[cur_msg_index:cur_msg_index+struct.calcsize('I')])
                cur_msg_index += struct.calcsize('I')
                detection_classes.append(det_class)
        else:
            print("Classification was not successful")
        return detection_boxes, detection_probs, detection_classes
