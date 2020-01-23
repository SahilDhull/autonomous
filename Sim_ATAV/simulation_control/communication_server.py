"""Defines the CommunicationServer class.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import socket
import select
import struct


class CommunicationServer(object):
    """CommunicationServer class is used to create a TCP/IP server,
    and to handle communications in the Webots supervisor."""
    def __init__(self, is_local, port_no, is_debug_mode):
        self.is_debug_mode = is_debug_mode
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if is_local:
            hostname = 'localhost'
        else:
            hostname = socket.gethostname()
        self.server_socket.bind((hostname, port_no))

    def get_connection(self):
        """Accepts connection from a client."""
        self.server_socket.listen(0)
        (client_socket, address) = self.server_socket.accept()
        if self.is_debug_mode:
            print("Communication Server: Comm accepted from {}".format(address))
        self.empty_socket(client_socket)
        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return client_socket

    def receive_blocking(self, client_socket):
        """Receives message in blocking mode.
        First 4 bytes of the message have to be the message length as uint32.
        The first byte is not returned from this function."""
        try:
            chunks = []
            bytes_recd = 0
            required_len = 64000
            while bytes_recd < required_len:
                chunk = client_socket.recv(min(required_len, 4096))
                chunks.append(chunk)
                if required_len == 64000 and len(chunk) >= struct.calcsize('I'):
                    msg_len = struct.unpack('I', chunk[0:struct.calcsize('I')])[0]
                    if self.is_debug_mode:
                        print('Communication Server: Received msg length: {}'.format(msg_len))
                    required_len = msg_len + struct.calcsize('I')
                bytes_recd = bytes_recd + len(chunk)
            ret_msg = b''.join(chunks)
            return ret_msg[struct.calcsize('I'):]
        except:
            raise

    def send_blocking(self, client_socket, data):
        """Sends the given data. Automatically adds data length in front of the sent message!"""
        try:
            len_data = len(data)
            send_msg = struct.pack('I', len_data)
            send_msg += data
            total_sent = 0
            if self.is_debug_mode:
                print("Communication Server: Length of msg to send: {}".format(len_data))
            while total_sent < len_data:
                sent = client_socket.send(send_msg[total_sent:])
                total_sent = total_sent + sent
        except:
            raise

    def empty_socket(self, recv_socket):
        """Empties garbage data on the receive buffer of the socket"""
        try:
            socket_input = [recv_socket]
            while True:
                input_ready, o, e = select.select(socket_input, [], [], 0.0)
                if input_ready:
                    break
                for s in input_ready:
                    s.recv(1)
        except:
            raise

    def close_connection(self):
        """Closes the server socket."""
        # self.server_socket.shutdown(socket.SHUT_RDWR)
        print("SUPERVISOR Server socket closed!!!")
        self.server_socket.close()
