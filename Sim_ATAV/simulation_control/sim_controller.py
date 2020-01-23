"""Defines SimController class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import time
import sys
import numpy as np
from Sim_ATAV.simulation_control.simulation_message_interface import SimulationMessageInterface
from Sim_ATAV.simulation_control.communication_server import CommunicationServer
from Sim_ATAV.simulation_control.robustness_computation import RobustnessComputation
from Sim_ATAV.simulation_control.sim_object_generator import SimObjectGenerator
from Sim_ATAV.simulation_control.supervisor_controls import SupervisorControls
from Sim_ATAV.simulation_control.vehicles_manager import VehiclesManager
from Sim_ATAV.simulation_control.pedestrians_manager import PedestriansManager
from Sim_ATAV.simulation_control.environment_manager import EnvironmentManager
from Sim_ATAV.simulation_control.heart_beat import HeartBeatConfig, HeartBeat
from Sim_ATAV.simulation_control.data_logger import DataLogger
from Sim_ATAV.simulation_control.item_description import ItemDescription
from Sim_ATAV.common.controller_communication_interface import ControllerCommunicationInterface
# For Data Logging and Plotting:
# from DataLogger import *


class SimController(object):
    """SimController class controls the flow of the simulation."""
    def __init__(self):
        self.heart_beat_config = HeartBeatConfig()
        self.sim_data = None
        self.sim_obj_generator = SimObjectGenerator()
        self.message_interface = SimulationMessageInterface()
        self.comm_server = None
        self.client_socket = None
        self.comm_port_number = 10021
        self.supervisor_control = SupervisorControls()
        self.vehicles_manager = None
        self.pedestrians_manager = None
        self.environment_manager = EnvironmentManager()
        self.current_sim_time_ms = 0
        self.robustness_function = None
        self.debug_mode = 1
        self.data_logger = None
        self.view_follow_item = None
        self.video_recording_obj = None
        self.controller_comm_interface = ControllerCommunicationInterface()

    def init(self, debug_mode, supervisor_params):
        """Initialize the simulation controller."""
        if self.debug_mode:
            print("Starting supervisor initialization")
            sys.stdout.flush()
        self.comm_port_number = supervisor_params
        self.supervisor_control.init(supervisor_params)
        self.controller_comm_interface = ControllerCommunicationInterface()
        self.vehicles_manager = VehiclesManager(self.supervisor_control, self.controller_comm_interface)
        self.pedestrians_manager = PedestriansManager(supervisor_controller=self.supervisor_control,
                                                      controller_comm_interface=self.controller_comm_interface,
                                                      vehicles_manager=self.vehicles_manager)
        self.vehicles_manager.set_pedestrians_manager(self.pedestrians_manager)
        self.debug_mode = debug_mode
        self.vehicles_manager.debug_mode = self.debug_mode
        self.pedestrians_manager.debug_mode = self.debug_mode
        if self.debug_mode:
            print("supervisor Initialization: OK")

    def set_debug_mode(self, mode):
        """Set debug mode."""
        self.debug_mode = mode

    def generate_road_network(self, road_list, sim_obj_generator, road_network_id):
        """Generate and add the road network to the simulator."""
        road_network_string = sim_obj_generator.generate_road_network_string(road_list, road_network_id)
        if self.debug_mode == 2:
            print(road_network_string)
            sys.stdout.flush()
        self.supervisor_control.add_obj_to_sim_from_string(road_network_string)

    def generate_road_disturbance(self, road_disturbance, sim_obj_generator):
        """Generate and add the road disturbance to the simulator."""
        obj_string = sim_obj_generator.generate_road_disturbance_string(road_disturbance)
        if self.debug_mode == 2:
            print(obj_string)
            sys.stdout.flush()
        self.supervisor_control.add_obj_to_sim_from_string(obj_string)

    def generate_fog(self, fog, sim_obj_generator):
        """Generate and add the road disturbance to the simulator."""
        obj_string = sim_obj_generator.generate_fog_string(fog)
        if self.debug_mode == 2:
            print(obj_string)
            sys.stdout.flush()
        self.supervisor_control.add_obj_to_sim_from_string(obj_string)

    def generate_vehicle(self, vhc, sim_obj_generator):
        """Generate and add the vehicle to the simulator."""
        vehicle_string = sim_obj_generator.generate_vehicle_string(vhc)
        if self.debug_mode == 2:
            print(vehicle_string)
            sys.stdout.flush()
        self.supervisor_control.add_obj_to_sim_from_string(vehicle_string)

    def generate_pedestrian(self, pedestrian, sim_obj_generator):
        """Generate and add the pedestrian to the simulator."""
        pedestrian_string = sim_obj_generator.generate_pedestrian_string(pedestrian)
        if self.debug_mode == 2:
            print(pedestrian_string)
            sys.stdout.flush()
        self.supervisor_control.add_obj_to_sim_from_string(pedestrian_string)

    def generate_sim_object(self, sim_object, sim_obj_generator):
        """Generate the given generic object and add to the simulator."""
        object_string = sim_obj_generator.generate_object_string(sim_object)
        if self.debug_mode == 2:
            print(object_string)
            sys.stdout.flush()
        self.supervisor_control.add_obj_to_sim_from_string(object_string)

    def receive_and_execute_commands(self):
        """Read all incoming commands until START SIMULATION Command."""
        if self.debug_mode:
            print("Waiting Commands")
            sys.stdout.flush()
        road_segments_to_add = []
        add_road_network_to_world = False
        v_u_t_to_add = []
        vhc_to_change = []
        add_v_u_t_to_world = []
        dummy_vhc_to_add = []
        add_dummy_vhc_to_world = []
        pedestrians_to_add = []
        add_pedestrian_to_world = []
        road_disturbances_to_add = []
        add_road_disturbance_to_world = []
        fog_to_add = []
        add_fog_to_world = []
        sim_objects_to_add = []
        initial_state_settings_list = []
        continue_simulation = False
        remote_command = None
        while continue_simulation is False or self.sim_data is None:
            rcv_msg = self.comm_server.receive_blocking(self.client_socket)
            remote_command = self.message_interface.interpret_message(rcv_msg)
            remote_response = []
            if self.debug_mode:
                print("Received command : {}".format(remote_command.command))
                sys.stdout.flush()
            if remote_command.command == self.message_interface.START_SIM:
                self.sim_data = remote_command.object
                continue_simulation = True
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.RELOAD_WORLD:
                continue_simulation = True
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.CONTINUE_SIM:
                continue_simulation = True
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.SET_VIDEO_RECORDING:
                print('video recording...')
                self.video_recording_obj = remote_command.object
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.SET_DATA_LOG_PERIOD_MS:
                if self.data_logger is None:
                    self.data_logger = DataLogger()
                    self.data_logger.set_environment_manager(self.environment_manager)
                    self.data_logger.set_vehicles_manager(self.vehicles_manager)
                    self.data_logger.set_pedestrians_manager(self.pedestrians_manager)
                self.data_logger.set_log_period(remote_command.object)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.SET_CONTROLLER_PARAMETER:
                if self.sim_data is not None and self.vehicles_manager is not None:
                    emitter = self.vehicles_manager.get_emitter()
                else:
                    emitter = None
                self.controller_comm_interface.transmit_set_controller_parameters_message(
                    emitter,
                    remote_command.object.vehicle_id,
                    remote_command.object.parameter_name,
                    remote_command.object.parameter_data)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command in (self.message_interface.SURROUNDINGS_DEF,
                                            self.message_interface.SURROUNDINGS_ADD):
                road_segments_to_add.append(remote_command.object)
                add_road_network_to_world = (remote_command.command == self.message_interface.SURROUNDINGS_ADD)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command in (self.message_interface.DUMMY_ACTORS_DEF,
                                            self.message_interface.DUMMY_ACTORS_ADD):
                dummy_vhc_to_add.append(remote_command.object)
                if remote_command.command == self.message_interface.DUMMY_ACTORS_ADD:
                    add_dummy_vhc_to_world.append(True)
                else:
                    add_dummy_vhc_to_world.append(False)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command in (self.message_interface.VUT_DEF, self.message_interface.VUT_ADD):
                v_u_t_to_add.append(remote_command.object)
                if remote_command.command == self.message_interface.VUT_ADD:
                    add_v_u_t_to_world.append(True)
                else:
                    add_v_u_t_to_world.append(False)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.CHANGE_VHC_POSITION:
                vhc_to_change.append(remote_command.object)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command in (self.message_interface.ROAD_DISTURBANCE_DEF,
                                            self.message_interface.ROAD_DISTURBANCE_ADD):
                road_disturbances_to_add.append(remote_command.object)
                if remote_command.command == self.message_interface.ROAD_DISTURBANCE_ADD:
                    add_road_disturbance_to_world.append(True)
                else:
                    add_road_disturbance_to_world.append(False)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command in (self.message_interface.FOG_DEF,
                                            self.message_interface.FOG_ADD):
                fog_to_add.append(remote_command.object)
                if remote_command.command == self.message_interface.FOG_ADD:
                    add_fog_to_world.append(True)
                else:
                    add_fog_to_world.append(False)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command in (self.message_interface.PEDESTRIAN_DEF,
                                            self.message_interface.PEDESTRIAN_ADD):
                pedestrians_to_add.append(remote_command.object)
                add_pedestrian_to_world.append(remote_command.command == self.message_interface.PEDESTRIAN_ADD)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.ADD_OBJECT_TO_SIMULATION:
                sim_objects_to_add.append(remote_command.object)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.SET_HEART_BEAT_CONFIG:
                self.heart_beat_config = remote_command.object
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.SET_ROBUSTNESS_TYPE:
                robustness_type = remote_command.object
                self.robustness_function = RobustnessComputation(robustness_type,
                                                                 self.supervisor_control,
                                                                 self.vehicles_manager,
                                                                 self.environment_manager,
                                                                 self.pedestrians_manager)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.SET_VIEW_FOLLOW_ITEM:
                self.view_follow_item = remote_command.object
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.SET_VIEW_POINT_POSITION:
                view_point_position = remote_command.object
                viewpoint = self.supervisor_control.getFromDef('VIEWPOINT')
                if viewpoint is not None:
                    view_position_obj = viewpoint.getField('position')
                    view_position_obj.setSFVec3f(view_point_position)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.SET_VIEW_POINT_ORIENTATION:
                view_point_orientation = remote_command.object
                viewpoint = self.supervisor_control.getFromDef('VIEWPOINT')
                if viewpoint is not None:
                    view_orientation_obj = viewpoint.getField('orientation')
                    view_orientation_obj.setSFRotation(view_point_orientation)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.SET_INITIAL_STATE:
                initial_state_settings_list.append(remote_command.object)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.STOP_BEFORE_COLLISION:
                (item_to_stop, item_not_to_collide) = remote_command.object
                if item_to_stop.item_type == ItemDescription.ITEM_TYPE_VEHICLE:
                    self.vehicles_manager.add_stop_before_collision_item(item_to_stop, item_not_to_collide)
                elif item_to_stop.item_type == ItemDescription.ITEM_TYPE_PEDESTRIAN:
                    self.pedestrians_manager.add_stop_before_collision_item(item_to_stop, item_not_to_collide)
                else:
                    print("WARNING! NOT Recognized Item Type for STOP_BEFORE_COLLISION!")
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.ADD_DATA_LOG_DESCRIPTION:
                if self.data_logger is None:
                    self.data_logger = DataLogger()
                    self.data_logger.set_environment_manager(self.environment_manager)
                    self.data_logger.set_vehicles_manager(self.vehicles_manager)
                    self.data_logger.set_pedestrians_manager(self.pedestrians_manager)
                self.data_logger.add_data_log_description(remote_command.object)
                if remote_command.object.item_type in [ItemDescription.ITEM_TYPE_VEHICLE_DET_PERF,
                                                       ItemDescription.ITEM_TYPE_PED_DET_PERF]:
                    self.vehicles_manager.collect_detection_perf_from_vehicles.append(remote_command.object.item_index)
                if remote_command.object.item_type == ItemDescription.ITEM_TYPE_VEHICLE_CONTROL:
                    self.vehicles_manager.add_vehicle_to_collect_control_list(remote_command.object.item_index)
                if self.sim_data is not None:
                    self.data_logger.set_expected_simulation_time(self.sim_data.simulation_duration_ms)
                    self.data_logger.set_simulation_step_size(self.sim_data.simulation_step_size_ms)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.GET_ROBUSTNESS:
                if self.robustness_function is not None:
                    rob = self.robustness_function.get_robustness()
                else:
                    rob = 0.0
                remote_response = self.message_interface.generate_robustness_msg(rob)
            elif remote_command.command == self.message_interface.ADD_DET_EVAL_CONFIG:
                if self.sim_data is not None and self.vehicles_manager is not None:
                    emitter = self.vehicles_manager.get_emitter()
                else:
                    emitter = None
                det_eval_config = remote_command.object
                for det_eval_obj in det_eval_config.target_objs:
                    self.data_logger.add_data_log_description(ItemDescription(
                        item_type=ItemDescription.ITEM_TYPE_DET_EVAL))
                self.controller_comm_interface.transmit_set_detection_monitor_message(emitter, det_eval_config)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.ADD_VISIBILITY_EVAL_CONFIG:
                if self.sim_data is not None and self.vehicles_manager is not None:
                    emitter = self.vehicles_manager.get_emitter()
                else:
                    emitter = None
                visibility_eval_config = remote_command.object
                for vis_eval_obj in visibility_eval_config.object_list:
                    self.data_logger.add_data_log_description(ItemDescription(
                        item_type=ItemDescription.VISIBILITY_EVAL))
                self.controller_comm_interface.transmit_set_visibility_monitor_message(emitter, visibility_eval_config)
                remote_response = self.message_interface.generate_ack_message()
            elif remote_command.command == self.message_interface.GET_DATA_LOG_INFO:
                if self.data_logger is not None:
                    log_info = self.data_logger.get_log_info()
                else:
                    log_info = (0, 0)
                remote_response = self.message_interface.generate_log_info_message(log_info)
            elif remote_command.command == self.message_interface.GET_DATA_LOG:
                requested_log_start_index = remote_command.object[0]
                requested_log_end_index = remote_command.object[1]
                if self.data_logger is not None:
                    data_log = self.data_logger.get_log(requested_log_start_index, requested_log_end_index)
                else:
                    data_log = np.empty(0)
                remote_response = self.message_interface.generate_data_log_message(data_log)
            elif remote_command.command == self.message_interface.SET_PERIODIC_REPORTING:
                if remote_command.object[0] == self.message_interface.PERIODIC_VHC_POSITIONS:
                    self.vehicles_manager.set_periodic_reporting(self.vehicles_manager.POSITION_REPORTING,
                                                                 remote_command.object[1],
                                                                 remote_command.object[2])
                elif remote_command.object[0] == self.message_interface.PERIODIC_VHC_ROTATIONS:
                    self.vehicles_manager.set_periodic_reporting(self.vehicles_manager.ROTATION_REPORTING,
                                                                 remote_command.object[1],
                                                                 remote_command.object[2])
                elif remote_command.object[0] == self.message_interface.PERIODIC_VHC_BOX_CORNERS:
                    self.vehicles_manager.set_periodic_reporting(self.vehicles_manager.CORNERS_REPORTING,
                                                                 remote_command.object[1],
                                                                 remote_command.object[2])
                elif remote_command.object[0] == self.message_interface.PERIODIC_PED_POSITIONS:
                    self.pedestrians_manager.set_periodic_reporting(self.pedestrians_manager.POSITION_REPORTING,
                                                                    remote_command.object[1],
                                                                    remote_command.object[2])
                elif remote_command.object[0] == self.message_interface.PERIODIC_PED_ROTATIONS:
                    self.pedestrians_manager.set_periodic_reporting(self.pedestrians_manager.ROTATION_REPORTING,
                                                                    remote_command.object[1],
                                                                    remote_command.object[2])
                elif remote_command.object[0] == self.message_interface.PERIODIC_PED_BOX_CORNERS:
                    self.pedestrians_manager.set_periodic_reporting(self.pedestrians_manager.CORNERS_REPORTING,
                                                                    remote_command.object[1],
                                                                    remote_command.object[2])
                remote_response = self.message_interface.generate_ack_message()
            if remote_response:
                self.comm_server.send_blocking(self.client_socket, remote_response)

        # Generate simulation environment (add VUT, Dummy Actors and Surroundings)
        if road_segments_to_add and self.debug_mode:
            print("Number of road segments: {}".format(len(road_segments_to_add)))
        if add_road_network_to_world:
            self.generate_road_network(road_segments_to_add,
                                       self.sim_obj_generator,
                                       self.environment_manager.get_num_of_road_networks() + 1)
        if road_segments_to_add:
            self.environment_manager.record_road_network(road_segments_to_add)

        for (dist_ind, disturbance) in enumerate(road_disturbances_to_add):
            if add_road_disturbance_to_world[dist_ind]:
                self.generate_road_disturbance(disturbance, self.sim_obj_generator)
            self.environment_manager.record_road_disturbance(disturbance)

        for (fog_ind, fog) in enumerate(fog_to_add):
            if add_fog_to_world[fog_ind]:
                self.generate_fog(fog, self.sim_obj_generator)
            else:
                world_fog = self.supervisor_control.getFromDef('FOG')
                if world_fog is not None:
                    world_fog_color = world_fog.getField('color')
                    world_fog_color.setSFVec3f(fog.color)
                    world_fog_visibility = world_fog.getField('visibilityRange')
                    world_fog_visibility.setSFFloat(fog.visibility_range)

        if v_u_t_to_add and self.debug_mode:
            print("Number of VUT: {}".format(len(v_u_t_to_add)))
        for (vhc_ind, vhc) in enumerate(v_u_t_to_add):
            if add_v_u_t_to_world[vhc_ind]:
                self.generate_vehicle(vhc, self.sim_obj_generator)
            self.vehicles_manager.record_vehicle(vhc, self.vehicles_manager.VHC_VUT)

        if vhc_to_change and self.debug_mode:
            print("Changing the position of {} vehicles".format(len(vhc_to_change)))
        for (vhc_ind, vhc) in enumerate(vhc_to_change):
            self.vehicles_manager.change_vehicle_pose(vhc)

        if dummy_vhc_to_add and self.debug_mode:
            print("Number of Dummy vehicles: {}".format(len(dummy_vhc_to_add)))
        for (vhc_ind, vhc) in enumerate(dummy_vhc_to_add):
            if add_dummy_vhc_to_world[vhc_ind]:
                self.generate_vehicle(vhc, self.sim_obj_generator)
            self.vehicles_manager.record_vehicle(vhc, self.vehicles_manager.VHC_DUMMY)

        if pedestrians_to_add and self.debug_mode:
            print("Number of Pedestrians: {}".format(len(pedestrians_to_add)))
        for (ped_ind, pedestrian) in enumerate(pedestrians_to_add):
            if add_pedestrian_to_world[ped_ind]:
                self.generate_pedestrian(pedestrian, self.sim_obj_generator)
            self.pedestrians_manager.record_pedestrian(pedestrian)

        if sim_objects_to_add and self.debug_mode:
            print("Number of simulation objects: {}".format(len(sim_objects_to_add)))
        for sim_object in sim_objects_to_add:
            self.generate_sim_object(sim_object, self.sim_obj_generator)

        for initial_state_setting in initial_state_settings_list:
            item = initial_state_setting[0]
            initial_value = initial_state_setting[1]
            if item.item_type == ItemDescription.ITEM_TYPE_PEDESTRIAN:
                if self.debug_mode:
                    print("Setting initial state {} of Pedestrian {} to {}".format(item.item_state_index,
                                                                                   item.item_index,
                                                                                   initial_value))
                self.pedestrians_manager.set_initial_state(item.item_index,
                                                           item.item_state_index,
                                                           initial_value)
            elif item.item_type == ItemDescription.ITEM_TYPE_VEHICLE:
                if self.debug_mode:
                    print("Setting initial state {} of Vehicle {} to {}".format(item.item_state_index,
                                                                                item.item_index,
                                                                                initial_value))
                self.vehicles_manager.set_initial_state(item.item_index,
                                                        item.item_state_index,
                                                        initial_value)
            else:
                print("WARNING! Initial State Setting NOT recognized: {}, {}, {}, {}".format(item.item_type,
                                                                                             item.item_index,
                                                                                             item.item_state_index,
                                                                                             initial_value))

        if remote_command is not None and remote_command.command == self.message_interface.RELOAD_WORLD:
            time.sleep(1.0)
            try:
                print('Closing connection!')
                self.comm_server.close_connection()
            except Exception as ex:
                print('Could not close connection!')
                print(repr(ex))
            time.sleep(0.5)
            self.comm_server = None
            print('Reverting Simulation!')
            self.supervisor_control.revert_simulation()
            time.sleep(1)
        sys.stdout.flush()

    def prepare_sim_environment(self):
        """Prepares the simulation environment based on the communication with simulation controller."""
        if self.debug_mode:
            print("Will Prepare Sim Environment")
        self.supervisor_control.initialize_creating_simulation_environment()
        self.comm_server = CommunicationServer(True, self.comm_port_number, self.debug_mode)
        self.client_socket = self.comm_server.get_connection()

        # Read all incoming commands until START SIMULATION Command
        self.receive_and_execute_commands()

        # Set viewpoint
        view_point_vehicle_index = 0
        if self.view_follow_item is not None:
            if self.view_follow_item.item_type == ItemDescription.ITEM_TYPE_VEHICLE:
                view_point_vehicle_index = self.view_follow_item.item_index
            if self.vehicles_manager.vehicles:
                if len(self.vehicles_manager.vehicles) <= view_point_vehicle_index:
                    view_point_vehicle_index = 0
                viewpoint = self.supervisor_control.getFromDef('VIEWPOINT')
                # !!! The world must have DEF VIEWPOINT Viewpoint {...} for the viewpoint for this to work. !!!
                if viewpoint is not None:
                    follow_point = viewpoint.getField('follow')
                    if follow_point is not None:
                        follow_point.setSFString(
                            self.vehicles_manager.vehicles[view_point_vehicle_index].name.getSFString())

        # Set time parameters for the objects where necessary.
        if self.data_logger is not None:
            self.data_logger.set_expected_simulation_time(self.sim_data.simulation_duration_ms)
            self.data_logger.set_simulation_step_size(self.sim_data.simulation_step_size_ms)

        self.vehicles_manager.set_time_step(self.sim_data.simulation_step_size_ms / 1000.0)
        self.pedestrians_manager.set_time_step(self.sim_data.simulation_step_size_ms / 1000.0)

        # Reflect changes to the simulation environment
        self.supervisor_control.finalize_creating_simulation_environment()
        if self.video_recording_obj is not None:
            if self.video_recording_obj.is_caption:
                self.supervisor_control.set_label(65535, self.video_recording_obj.caption_name, 0, 0, 0.1, 0xff0000, 0,
                                                  "Arial")
            self.supervisor_control.start_movie_recording(self.video_recording_obj.filename,
                                                          self.video_recording_obj.width,
                                                          self.video_recording_obj.height,
                                                          self.video_recording_obj.codec,
                                                          self.video_recording_obj.quality,
                                                          self.video_recording_obj.acceleration,
                                                          self.video_recording_obj.is_caption)

    def run(self):
        """The overall execution of the simulation."""
        # Prepare Simulation Environment
        self.prepare_sim_environment()

        # Start simulation
        print('Simulation Duration = {}, step size = {}'.format(self.sim_data.simulation_duration_ms,
                                                                self.sim_data.simulation_step_size_ms))
        sys.stdout.flush()
        if self.sim_data.simulation_execution_mode == self.sim_data.SIM_TYPE_RUN:
            self.supervisor_control.set_simulation_mode(self.supervisor_control.SIMULATION_MODE_RUN)
        elif self.sim_data.simulation_execution_mode == self.sim_data.SIM_TYPE_FAST_NO_GRAPHICS:
            self.supervisor_control.set_simulation_mode(self.supervisor_control.SIMULATION_MODE_FAST)
        else:
            self.supervisor_control.set_simulation_mode(self.supervisor_control.SIMULATION_MODE_REAL_TIME)

        # Execute simulation
        while self.current_sim_time_ms < self.sim_data.simulation_duration_ms:
            cur_sim_time_s = self.current_sim_time_ms / 1000.0
            self.vehicles_manager.simulate_vehicles(cur_sim_time_s)
            self.pedestrians_manager.simulate_pedestrians(cur_sim_time_s)
            if self.robustness_function is not None:
                self.robustness_function.compute_robustness(cur_sim_time_s)
            if self.data_logger is not None:
                self.data_logger.log_data(self.current_sim_time_ms)
            if (self.heart_beat_config is not None
                    and self.heart_beat_config.sync_type in (HeartBeatConfig.WITH_SYNC, HeartBeatConfig.WITHOUT_SYNC)
                    and self.current_sim_time_ms % self.heart_beat_config.period_ms == 0):
                heart_beat_message = self.message_interface.generate_heart_beat_message(self.current_sim_time_ms,
                                                                                        HeartBeat.SIMULATION_RUNNING)
                self.comm_server.send_blocking(self.client_socket, heart_beat_message)
                if self.heart_beat_config.sync_type == HeartBeatConfig.WITH_SYNC:
                    # This means it has to wait for a new command.
                    self.receive_and_execute_commands()
            self.supervisor_control.step_simulation(self.sim_data.simulation_step_size_ms)
            self.current_sim_time_ms = self.current_sim_time_ms + self.sim_data.simulation_step_size_ms
            if self.current_sim_time_ms >= (self.sim_data.simulation_duration_ms -
                                            self.sim_data.simulation_step_size_ms - 1):
                if self.video_recording_obj is not None:
                    self.supervisor_control.stop_movie_recording()
                    self.supervisor_control.wait_until_movie_is_ready()

        # End Simulation
        self.supervisor_control.set_simulation_mode(self.supervisor_control.SIMULATION_MODE_PAUSE)
        heart_beat_message = self.message_interface.generate_heart_beat_message(self.current_sim_time_ms,
                                                                                HeartBeat.SIMULATION_STOPPED)
        self.comm_server.send_blocking(self.client_socket, heart_beat_message)
        for _ in range(100):  # Maximum 100 messages are accepted after the simulation. Against lock / memory grow etc.
            if self.comm_server is not None:
                self.receive_and_execute_commands()
