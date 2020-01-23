""" Defines SimEnvironmentConfigurator class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import time
import sys
from Sim_ATAV.simulation_control.heart_beat import HeartBeat
from Sim_ATAV.simulation_configurator.simulation_communication_interface import SimulationCommunicationInterface
from Sim_ATAV.simulation_configurator import sim_config_tools


class SimEnvironmentConfigurator(object):
    INTER_MESSAGE_WAIT_TIME = 0.01  # To give some time to Webots to modify simulation environment

    def __init__(self, sim_config):
        self.sim_config = sim_config
        self.comm_interface = None

    def print_error(self, error_str):
        print('ERROR: SimEnvironmentConfigurator: ' + error_str)
        sys.stdout.flush()

    def connect(self, max_connection_retry=100):
        # Try connecting. If fails, restart webots and try again.
        if self.comm_interface is None:
            self.comm_interface = SimulationCommunicationInterface(server_address=self.sim_config.server_ip,
                                                                   server_port=self.sim_config.server_port,
                                                                   max_connection_retry=max_connection_retry)
            time.sleep(0.1)
        simulator_instance = None  # simulator_instance will keep the pid of the Webots if we need to start it here.
        if self.comm_interface.comm_module is None:
            self.comm_interface = None
            try:
                # This is something which generally pops up when Webots crashes in Windows
                sim_config_tools.kill_process_by_name('WerFault.exe')
            except Exception as ex:
                print('ERROR: ' + repr(ex))
            try:
                sim_config_tools.kill_webots_by_name()
            except Exception as ex:
                print('ERROR: ' + repr(ex))
            time.sleep(0.5)
            simulator_instance = sim_config_tools.start_webots(world_file=self.sim_config.world_file,
                                                               minimized=False)
            time.sleep(2.5)

        if self.comm_interface is None:
            self.comm_interface = SimulationCommunicationInterface(server_address=self.sim_config.server_ip,
                                                                   server_port=self.sim_config.server_port,
                                                                   max_connection_retry=max_connection_retry)
            time.sleep(0.1)

        if self.comm_interface is not None:
            success = True
        else:
            success = False
            try:
                # This is something which generally pops up when Webots crashes in Windows
                sim_config_tools.kill_process_by_name('WerFault.exe')
            except Exception as ex:
                print('ERROR: ' + repr(ex))
            try:
                sim_config_tools.kill_webots_by_name()
            except Exception as ex:
                print('ERROR: ' + repr(ex))
            time.sleep(2.0)
            sim_config_tools.start_webots(world_file=self.sim_config.world_file, minimized=False)
            time.sleep(2.0)
        return success, simulator_instance

    def start_simulation(self):
        if not self.comm_interface.start_simulation(self.sim_config.sim_duration_ms,
                                                    self.sim_config.sim_step_size,
                                                    self.sim_config.run_config_arr[0].simulation_run_mode):
            self.print_error('START SIMULATION error')

    def wait_until_simulation_ends(self):
        simulation_continues = True
        while simulation_continues:
            received_heart_beat = self.comm_interface.receive_heart_beat()
            if received_heart_beat is not None:
                if received_heart_beat.simulation_status == HeartBeat.SIMULATION_STOPPED:
                    simulation_continues = False
        return

    def receive_simulation_trace(self):
        return self.comm_interface.get_data_log()

    def end_simulation(self):
        if not self.comm_interface.restart_simulation():
            self.print_error('RESTART SIMULATION error')
        time.sleep(0.2)
        self.comm_interface.disconnect_from_simulator()
        del self.comm_interface
        self.comm_interface = None

    def run_simulation_get_trace(self):
        self.start_simulation()
        self.wait_until_simulation_ends()
        sim_trace = self.receive_simulation_trace()
        self.end_simulation()
        return sim_trace

    def setup_sim_environment(self, sim_environment):
        if hasattr(sim_environment, 'fog') and sim_environment.fog is not None:
            if not self.comm_interface.add_fog_to_simulation(sim_environment.fog, is_create=True):
                self.print_error('ADD FOG error')
            time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'heart_beat_config') and sim_environment.heart_beat_config is not None:
            if not self.comm_interface.set_heart_beat_config(
                    sync_type=sim_environment.heart_beat_config.sync_type,
                    heart_beat_period_ms=sim_environment.heart_beat_config.period_ms):
                self.print_error('SET HEART BEAT error')
            time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'ego_vehicles_list'):
            for vhc_obj in sim_environment.ego_vehicles_list:
                if not self.comm_interface.add_vehicle_to_simulation(vehicle_object=vhc_obj,
                                                                     is_dummy=False,
                                                                     is_create=True):
                    self.print_error('ADD EGO VEHICLE error')
                time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'agent_vehicles_list'):
            for vhc_obj in sim_environment.agent_vehicles_list:
                if not self.comm_interface.add_vehicle_to_simulation(vehicle_object=vhc_obj,
                                                                     is_dummy=True,
                                                                     is_create=True):
                    self.print_error('ADD AGENT VEHICLE error')
                time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'pedestrians_list'):
            for pedestrian in sim_environment.pedestrians_list:
                if not self.comm_interface.add_pedestrian_to_simulation(pedestrian_object=pedestrian,
                                                                        is_create=True):
                    self.print_error('ADD PEDESTRIAN error')
                time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'road_list'):
            for road in sim_environment.road_list:
                if not self.comm_interface.add_road_to_simulation(road_object=road,
                                                                  is_create=True):
                    self.print_error('ADD ROAD error')
                time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'road_disturbances_list'):
            for road_disturbance in sim_environment.road_disturbances_list:
                if not self.comm_interface.add_road_disturbance_to_simulation(road_disturbance_object=road_disturbance,
                                                                              is_create=True):
                    self.print_error('ADD ROAD DISTURBANCE error')
                time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'controller_params_list'):
            for controller_param in sim_environment.controller_params_list:
                if not self.comm_interface.send_controller_parameter(controller_param):
                    self.print_error('ADD controller param error')
            time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'data_log_description_list'):
            for data_log_description in sim_environment.data_log_description_list:
                if not self.comm_interface.add_data_log_description(
                        item_type=data_log_description.item_type,
                        item_index=data_log_description.item_index,
                        item_state_index=data_log_description.item_state_index):
                    self.print_error('ADD DATA LOG DESCRIPTION error')
            time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'data_log_period_ms') and sim_environment.data_log_period_ms is not None:
            if not self.comm_interface.set_data_log_period_ms(sim_environment.data_log_period_ms):
                self.print_error('SET DATA LOG PERIOD error')
            time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'detection_evaluation_config_list'):
            for detection_evaluation_config in sim_environment.detection_evaluation_config_list:
                if not self.comm_interface.add_detection_evaluation_config(detection_evaluation_config):
                    self.print_error('ADD DETECTION EVAL CONFIG error')
            time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'visibility_evaluation_config_list'):
            for visibility_evaluation_config in sim_environment.visibility_evaluation_config_list:
                if not self.comm_interface.add_visibility_evaluation_config(visibility_evaluation_config):
                    self.print_error('ADD VISIBILITY EVAL CONFIG error')
            time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'initial_state_config_list'):
            for initial_state_config in sim_environment.initial_state_config_list:
                if not self.comm_interface.set_initial_state(item_type=initial_state_config.item.item_type,
                                                             item_index=initial_state_config.item.item_index,
                                                             item_state_index=initial_state_config.item.item_state_index,
                                                             initial_value=initial_state_config.value):
                    self.print_error("SET INITIAL STATE error")
                time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'stop_before_collision_config_list'):
            for stop_before_collision_config in sim_environment.stop_before_collision_config_list:
                if not self.comm_interface.set_stop_before_collision_item(
                        item_to_stop_type=stop_before_collision_config.item_to_stop_type,
                        item_to_stop_ind=stop_before_collision_config.item_to_stop_ind,
                        item_not_to_collide_type=stop_before_collision_config.item_not_to_collide_type,
                        item_not_to_collide_ind=stop_before_collision_config.item_not_to_collide_ind):
                    self.print_error("SET STOP BEFORE COLLISION error")
                time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'periodic_reporting_config_list'):
            for periodic_reporting_config in sim_environment.periodic_reporting_config_list:
                if not self.comm_interface.set_periodic_reporting(
                        entity_type=periodic_reporting_config.item_type,
                        report_type=periodic_reporting_config.report_type,
                        entity_id=periodic_reporting_config.item_id,
                        period=periodic_reporting_config.reporting_period):
                    self.print_error("SET PERIODIC REPORTING error")
                time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'view_follow_config') and sim_environment.view_follow_config is not None:
            if (sim_environment.view_follow_config.item_type is not None and
                    sim_environment.view_follow_config.item_index is not None):
                if not self.comm_interface.set_view_follow_point(
                        item_type=sim_environment.view_follow_config.item_type,
                        item_index=sim_environment.view_follow_config.item_index):
                    self.print_error('SET VIEW FOLLOW POINT error')
            if sim_environment.view_follow_config.position is not None:
                if not self.comm_interface.set_view_point_position(sim_environment.view_follow_config.position):
                    self.print_error('SET VIEW POINT POSITION error')
            if sim_environment.view_follow_config.rotation is not None:
                if not self.comm_interface.set_view_point_orientation(sim_environment.view_follow_config.rotation):
                    self.print_error('SET VIEW POINT ORIENTATION error')
            time.sleep(self.INTER_MESSAGE_WAIT_TIME)
        if hasattr(sim_environment, 'generic_sim_objects_list'):
            for sim_obj in sim_environment.generic_sim_objects_list:
                if not self.comm_interface.add_generic_object_to_simulation(sim_obj):
                    self.print_error("ADD GENERIC SIM OBJECT error")
                time.sleep(self.INTER_MESSAGE_WAIT_TIME)
