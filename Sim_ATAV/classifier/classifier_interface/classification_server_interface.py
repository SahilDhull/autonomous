"""Defines the ClassificationServerInterface class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import struct
import mmap
import time
from Sim_ATAV.classifier.classifier_interface.communication_server import CommunicationServer
from Sim_ATAV.classifier.classifier_interface.classification_command import ClassificationCommand


class ClassificationServerInterface(object):
    """ClassificationServerInterface provides the methods for setting up a classification server."""

    def __init__(self, port=10101, is_debug_mode=False, classification_engine=None):
        # Constants and configuration
        self.server_port = port
        self.is_debug_mode = is_debug_mode
        self.shared_mem = None
        self.shared_file_size = 1437696
        self.shared_file_name = "Local\\WebotsCameraImage"
        self.communication_up = False
        self.comm_server = None
        self.client_socket = None
        self.classification_engine = classification_engine

    def setup_connection(self):
        """Setup connection."""
        self.comm_server = CommunicationServer(True, self.server_port, self.is_debug_mode)
        self.client_socket = self.comm_server.get_connection()

    def end_communication(self):
        """Close socket communication."""
        time.sleep(0.5)
        if self.comm_server is not None:
            self.comm_server.close_connection()
            self.comm_server = None
            self.client_socket = None

    def start_service(self):
        """Starts classification server. Will run indefinitely."""
        self.communication_up = True
        while self.communication_up:
            self.receive_and_execute_commands()
        self.end_communication()

    def receive_and_execute_commands(self):
        """Receive the commands from the client and execute them."""
        rcv_msg = self.comm_server.receive_blocking(self.client_socket)
        (command, ) = struct.unpack('B', rcv_msg[0:struct.calcsize('B')])
        cur_msg_index = struct.calcsize('B')
        response = None

        if command == ClassificationCommand.CMD_CLASSIFY:
            # Classify the image which was put in the shared memory.
            (frame_id, width, height) = struct.unpack('III',
                                                      rcv_msg[cur_msg_index:cur_msg_index + \
                                                              struct.calcsize('III')])
            cur_msg_index += struct.calcsize('III')
            if self.is_debug_mode:
                print('Classification Requested. frame id: {} w: {} h: {}'.format(frame_id,
                                                                                  width,
                                                                                  height))
            # Do classification
            if self.shared_mem is not None and self.classification_engine is not None:
                if self.classification_engine.sess is None:
                    self.classification_engine.start_classification_engine()
                self.shared_mem.seek(0)
                shared_bytes = self.shared_mem.read(width*height*3)
                (detection_boxes, detection_probs, detection_classes, detection_result_image, _original_image) = \
                    self.classification_engine.do_object_detection_on_raw_data(shared_bytes,
                                                                               width,
                                                                               height,
                                                                               is_return_det_image=False,
                                                                               is_return_org_image=False)
                response = self.generate_classification_response(frame_id,
                                                                 detection_boxes,
                                                                 detection_probs,
                                                                 detection_classes)
            else:
                response = struct.pack('B', ClassificationCommand.NACK)
        elif command == ClassificationCommand.CMD_START_SHARED_MEMORY:
            # Record the shared memory file information.
            (shared_file_size, filename_length) = \
                struct.unpack('II', rcv_msg[cur_msg_index:cur_msg_index + struct.calcsize('II')])
            cur_msg_index += struct.calcsize('II')
            (shared_file_name, ) = struct.unpack(str(filename_length)+'s',
                                                 rcv_msg[cur_msg_index:cur_msg_index + \
                                                    struct.calcsize(str(filename_length)+'s')])
            shared_file_name = shared_file_name.rstrip(b'\0')
            shared_file_name = shared_file_name.decode("utf-8")
            self.shared_mem = mmap.mmap(0,
                                        shared_file_size,
                                        shared_file_name,
                                        mmap.ACCESS_READ)
            response = struct.pack('B', ClassificationCommand.ACK)
        elif command == ClassificationCommand.CMD_CLOSE_SHARED_MEMORY:
            # Close the shared memory file.
            self.shared_mem.close()
            self.shared_mem = None
            response = struct.pack('B', ClassificationCommand.ACK)
        elif command == ClassificationCommand.CMD_END_COMMUNICATION:
            # End communication. Close the shared memory file if it was not closed.
            if self.shared_mem is not None:
                self.shared_mem.close()
                self.shared_mem = None
            response = struct.pack('B', ClassificationCommand.ACK)
            self.communication_up = False

        if response is not None:
            self.comm_server.send_blocking(self.client_socket, response)

    def generate_classification_response(self,
                                         frame_id,
                                         detection_boxes,
                                         detection_probs,
                                         detection_classes):
        """Generates the classification results response to the client."""
        response = struct.pack('B', ClassificationCommand.CMD_CLASSIFICATION_RESULT)
        response += struct.pack('I', frame_id)
        num_of_detected_objects = len(detection_classes)
        response += struct.pack('I', num_of_detected_objects)
        for obj_ind in range(num_of_detected_objects):
            response += struct.pack('ffff',
                                    float(detection_boxes[obj_ind][0]),
                                    float(detection_boxes[obj_ind][1]),
                                    float(detection_boxes[obj_ind][2]),
                                    float(detection_boxes[obj_ind][3]))
            response += struct.pack('f', float(detection_probs[obj_ind]))
            response += struct.pack('I', int(detection_classes[obj_ind]))
        return response
