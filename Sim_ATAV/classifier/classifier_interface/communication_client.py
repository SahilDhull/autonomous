"""Defines CommunicationClient class.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import socket
import struct
import time


class CommunicationClient(object):
    """CommunicationClient class handles the communication with the Webots.
    There must be a corresponding server initiated at the Webots.
    It connects to the server running at the Webots side."""
    def __init__(self, server_ip, server_port):
        self.debug_mode = 0
        self.server_port = server_port
        self.server_ip = server_ip
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server_address = (self.server_ip, self.server_port)

    def connect_to_server(self, max_connection_retry):
        """Connects to the server running at the Webots side."""
        if self.debug_mode:
            print("CommunicationClient: Connecting to {}".format(self.server_address))
        connected = False
        try_count = 0
        while not connected and try_count < max_connection_retry:
            try:
                self.client_socket.connect(self.server_address)
                connected = True
            except:  # TODO: Handle communication errors based on the exception err message.
                try_count += 1
                time.sleep(1.0)
                print("CommunicationClient: Retrying Connecting to {}".format(self.server_address))
                self.disconnect_from_server()
                time.sleep(0.5)
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                time.sleep(0.5)
        if not connected:
            print("CommunicationClient: Could not connect!")

        return connected

    def disconnect_from_server(self):
        """Disconnects from the server at the Webots side."""
        if self.debug_mode:
            print("CommunicationClient: Closing socket")
        self.client_socket.close()

    def send_to_server(self, message):
        """Sends the message to the server at the Webots side."""
        if self.debug_mode:
            print("CommunicationClient: Sending {} bytes".format(len(message)))
        send_msg = struct.pack('I', len(message))
        send_msg += message
        self.client_socket.sendall(send_msg)

    def receive_blocking(self):
        """Receives message in blocking mode.
        First 4 bytes of the message have to be the message length as uint32.
        The first byte is not returned from this function."""
        try:
            chunks = []
            bytes_recd = 0
            required_len = 64000
            while bytes_recd < required_len:
                chunk = self.client_socket.recv(min(required_len, 4096))
                chunks.append(chunk)
                if required_len == 64000 and len(chunk) >= struct.calcsize('I'):
                    msg_len = struct.unpack('I', chunk[0:struct.calcsize('I')])[0]
                    required_len = msg_len + struct.calcsize('I')
                bytes_recd = bytes_recd + len(chunk)
            ret_msg = b''.join(chunks)
            return ret_msg[struct.calcsize('I'):]
        except:
            raise

    def receive_from_server(self, expected_size):
        """Receive message from Webots side."""
        received_size = 0
        received_data = []
        while received_size < expected_size:
            new_received = self.client_socket.recv(expected_size - received_size)
            received_data.append(new_received)
            received_size += len(new_received)

        ret_msg = ''.join(received_data)
        if self.debug_mode:
            print("CommunicationClient: Received {} bytes of data".format(received_size))
        return ret_msg

    def set_socket_timeout(self, time_out):
        """Sets the socket time out duration."""
        self.client_socket.settimeout(time_out)
