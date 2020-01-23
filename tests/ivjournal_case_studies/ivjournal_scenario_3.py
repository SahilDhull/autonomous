"""Case Study 3 for IV Journal Work
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import time
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_PATH + '/../../')  # This is needed to call from Matlab
from Sim_ATAV.simulation_control.sim_data import SimData
from Sim_ATAV.simulation_control.webots_sensor import WebotsSensor
from Sim_ATAV.simulation_control.webots_fog import WebotsFog
from Sim_ATAV.simulation_control.webots_vehicle import WebotsVehicle
from Sim_ATAV.simulation_control.heart_beat import HeartBeatConfig
from Sim_ATAV.simulation_control.item_description import ItemDescription
from Sim_ATAV.simulation_control.webots_controller_parameter import WebotsControllerParameter
from Sim_ATAV.simulation_configurator import sim_config_tools
from Sim_ATAV.simulation_configurator import covering_array_utilities
from Sim_ATAV.simulation_configurator import trajectory_tools
from Sim_ATAV.simulation_configurator import experiment_tools
from Sim_ATAV.vehicle_control.controller_commons.perf_evaluation.detection_evaluation_config \
    import DetectionEvaluationConfig
from Sim_ATAV.vehicle_control.controller_commons.perf_evaluation.visibility_evaluator \
    import VisibilitySensor, VisibilityConfig
from Sim_ATAV.simulation_configurator.sim_environment import SimEnvironment
from Sim_ATAV.simulation_control.initial_state_config import InitialStateConfig
from Sim_ATAV.simulation_configurator.stop_before_collision_config import StopBeforeCollisionConfig
from Sim_ATAV.simulation_configurator.view_follow_config import ViewFollowConfig
from Sim_ATAV.simulation_control.periodic_reporting_config import PeriodicReportingConfig
from Sim_ATAV.simulation_configurator.sim_environment_configurator import SimEnvironmentConfigurator
from Sim_ATAV.classifier.classifier_interface.gpu_check import check_system_gpu


(HAS_GPU, GPU_CPU_ID) = check_system_gpu()
SIMULATION_DURATION_MS = 12000
EXPERIMENT_WORLD_TIME_STEP_MS = 10
LANE_WIDTH = 3.5

PED_COLOR_DICT = {1: [1.0, 0.0, 0.0],
                  2: [0.0, 1.0, 0.0],
                  3: [0.0, 0.0, 1.0],
                  4: [1.0, 1.0, 1.0],
                  5: [0.0, 0.0, 0.0]}

CAR_COLOR_DICT = {1: [1.0, 0.0, 0.0],
                  2: [0.0, 1.0, 0.0],
                  3: [0.0, 0.0, 1.0],
                  4: [1.0, 1.0, 1.0],
                  5: [0.0, 0.0, 0.0]}

PARAMETER_NAME_TYPE_DICT = {'ego_init_speed_m_s': float,
                            'ego_enter_lateral_offset': float,
                            'ego_enter_longitudinal_offset': float,
                            'ego_exit_lateral_offset': float,
                            'ego_exit_longitudinal_offset': float,
                            'agent_1_enter_lateral_offset': float,
                            'agent_1_enter_longitudinal_offset': float,
                            'agent_1_exit_lateral_offset': float,
                            'agent_1_exit_longitudinal_offset': float,
                            'agent_1_init_longitudinal_position': float,
                            'agent_1_speed': float}

DEFAULT_PARAMETERS_DICT = {'ego_init_speed_m_s': 20.0,
                           'ego_enter_lateral_offset': 0.0,
                           'ego_enter_longitudinal_offset': 0.0,
                           'ego_exit_lateral_offset': 0.0,
                           'ego_exit_longitudinal_offset': 0.0,
                           'agent_1_enter_lateral_offset': 0.0,
                           'agent_1_enter_longitudinal_offset': 0.0,
                           'agent_1_exit_lateral_offset': 0.0,
                           'agent_1_exit_longitudinal_offset': 0.0,
                           'agent_1_init_longitudinal_position': 70.0,
                           'agent_1_speed': 20.0}

FALSIFICATION_PARAMETER_INDEX_MAP = ['ego_init_speed_m_s',
                                     'ego_enter_lateral_offset',
                                     'ego_enter_longitudinal_offset',
                                     'ego_exit_lateral_offset',
                                     'ego_exit_longitudinal_offset',
                                     'agent_1_enter_lateral_offset',
                                     'agent_1_enter_longitudinal_offset',
                                     'agent_1_exit_lateral_offset',
                                     'agent_1_exit_longitudinal_offset',
                                     'agent_1_init_longitudinal_position',
                                     'agent_1_speed']

SIM_CONFIG = sim_config_tools.SimulationConfig(1)
SIM_CONFIG.run_config_arr.append(sim_config_tools.RunConfig())
SIM_CONFIG.run_config_arr[0].simulation_run_mode = SimData.SIM_TYPE_RUN
SIM_CONFIG.sim_duration_ms = SIMULATION_DURATION_MS
SIM_CONFIG.sim_step_size = EXPERIMENT_WORLD_TIME_STEP_MS


def lane_number_to_lat_pos(lane_number, direction='N'):
    """Returns the lateral position fo the center of the lane.
    Valid only for the corresponding world file."""
    if direction == 'N':
        lat_pos = float(lane_number - 2) * LANE_WIDTH
    elif direction == 'S':
        lat_pos = 13.0 - float(lane_number - 2) * LANE_WIDTH
    elif direction == 'E':
        lat_pos = 12.0 + float(lane_number - 2) * LANE_WIDTH
    else:  # direction == 'W'
        lat_pos = 29.0 - float(lane_number - 2) * LANE_WIDTH
    return lat_pos


def lane_start_long_pos(direction):
    if direction == 'N':
        long_pos = 45.0
    elif direction == 'S':
        long_pos = -4.0
    elif direction == 'E':
        long_pos = -18.0
    else:
        long_pos = 31.0
    return long_pos


def define_sim_environment(parameters_dict):

    ego_init_speed_m_s = parameters_dict['ego_init_speed_m_s']
    ego_enter_lateral_offset = parameters_dict['ego_enter_lateral_offset']
    ego_enter_longitudinal_offset = parameters_dict['ego_enter_longitudinal_offset']
    ego_exit_lateral_offset = parameters_dict['ego_exit_lateral_offset']
    ego_exit_longitudinal_offset = parameters_dict['ego_exit_longitudinal_offset']
    agent_1_enter_lateral_offset = parameters_dict['agent_1_enter_lateral_offset']
    agent_1_enter_longitudinal_offset = parameters_dict['agent_1_enter_longitudinal_offset']
    agent_1_exit_lateral_offset = parameters_dict['agent_1_exit_lateral_offset']
    agent_1_exit_longitudinal_offset = parameters_dict['agent_1_exit_longitudinal_offset']
    agent_1_init_longitudinal_position = parameters_dict['agent_1_init_longitudinal_position']
    agent_1_speed = parameters_dict['agent_1_speed']

    agent_1_enter_lateral_position = lane_number_to_lat_pos(3, 'S') + agent_1_enter_lateral_offset
    ego_enter_lateral_position = float(lane_number_to_lat_pos(3, 'N')) + ego_enter_lateral_offset
    agent_1_model = 'CitroenCZero'
    agent_1_color = [1, 0, 0]

    sim_environment = SimEnvironment()
    # ----- Define VEHICLES:
    cur_vhc_id = 0
    # Ego vehicle
    vhc_obj = sim_config_tools.create_vehicle_object()
    vhc_obj.current_position = [ego_enter_lateral_position, 0.35, -100.0]
    vhc_obj.current_orientation = 0.0
    vhc_obj.rotation = [0.0, 1.0, 0.0, vhc_obj.current_orientation]
    cur_vhc_id += 1
    vhc_obj.vhc_id = cur_vhc_id
    vhc_obj.color = [1.0, 1.0, 0.0]
    vhc_obj.set_vehicle_model('ToyotaPrius')
    vhc_obj.controller = 'automated_driving_with_fusion2'
    vhc_obj.is_controller_name_absolute = True
    vhc_obj.controller_arguments.append('Toyota')
    vhc_obj.controller_arguments.append(str(ego_init_speed_m_s*3.6))
    vhc_obj.controller_arguments.append(str(ego_enter_lateral_position))
    vhc_obj.controller_arguments.append(str(cur_vhc_id))
    vhc_obj.controller_arguments.append('True')
    vhc_obj.controller_arguments.append(str(HAS_GPU))
    vhc_obj.controller_arguments.append(str(GPU_CPU_ID))
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[0].sensor_location = WebotsSensor.TOP
    vhc_obj.sensor_array[0].sensor_type = 'Camera'
    vhc_obj.sensor_array[0].add_sensor_field('rotation', '1 0 0 0.01')
    vhc_obj.sensor_array[0].add_sensor_field('width', '1248')
    vhc_obj.sensor_array[0].add_sensor_field('height', '384')
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
    vhc_obj.sensor_array[-1].sensor_type = 'Display'
    vhc_obj.sensor_array[-1].add_sensor_field('width', '1248')
    vhc_obj.sensor_array[-1].add_sensor_field('height', '384')
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
    vhc_obj.sensor_array[-1].sensor_type = 'Display'
    vhc_obj.sensor_array[-1].add_sensor_field('name', '"sensor_display"')
    vhc_obj.sensor_array[-1].add_sensor_field('width', '512')
    vhc_obj.sensor_array[-1].add_sensor_field('height', '512')
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
    vhc_obj.sensor_array[-1].sensor_type = 'Receiver'
    vhc_obj.sensor_array[-1].add_sensor_field('name', '"receiver"')
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
    vhc_obj.sensor_array[-1].sensor_type = 'Compass'
    vhc_obj.sensor_array[-1].add_sensor_field('name', '"compass"')
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
    vhc_obj.sensor_array[-1].sensor_type = 'Emitter'
    vhc_obj.sensor_array[-1].add_sensor_field('name', '"emitter"')
    vhc_obj.sensor_array[-1].add_sensor_field('channel', '2')
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
    vhc_obj.sensor_array[-1].sensor_type = 'GPS'
    # vhc_obj.sensor_array[-1].add_sensor_field('accuracy', '0.1')
    # vhc_obj.sensor_array[-1].add_sensor_field('speedNoise', '0.1')
    # vhc_obj.sensor_array.append(WebotsSensor())
    # vhc_obj.sensor_array[-1].sensor_type = 'VelodyneVLP-16'  # 'VelodyneVLP-16' # 'IbeoLux'  # 'VelodyneHDL-64E'
    # vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.TOP
    # vhc_obj.sensor_array[-1].add_sensor_field('rotation', '0 1 0 3.14157')
    # vhc_obj.sensor_array[-1].add_sensor_field('name', '"velodyne"')
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_type = 'Radar' # 'Radar' #'DelphiESR'
    vhc_obj.sensor_array[-1].add_sensor_field('name', '"radar"')
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.FRONT
    sim_environment.ego_vehicles_list.append(vhc_obj)

    # ----- Agent vehicles
    # Agent:
    cur_vhc_id += 1
    vhc_obj = sim_config_tools.create_vehicle_object()
    vhc_obj.current_position = [agent_1_enter_lateral_position, 0.35, agent_1_init_longitudinal_position]
    vhc_obj.current_orientation = math.pi
    vhc_obj.controller = 'path_and_speed_follower'
    vhc_obj.controller_arguments.append(str(str(agent_1_speed)))
    vhc_obj.controller_arguments.append('True')
    vhc_obj.controller_arguments.append(str(vhc_obj.current_position[0]))
    vhc_obj.controller_arguments.append(str(cur_vhc_id))
    vhc_obj.controller_arguments.append('False')
    vhc_obj.controller_arguments.append('False')
    vhc_obj.rotation = [0.0, 1.0, 0.0, vhc_obj.current_orientation]
    vhc_obj.vhc_id = cur_vhc_id
    vhc_obj.set_vehicle_model(agent_1_model)
    vhc_obj.color = agent_1_color
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
    vhc_obj.sensor_array[-1].sensor_type = 'Receiver'
    vhc_obj.sensor_array[-1].add_sensor_field('name', '"receiver"')
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
    vhc_obj.sensor_array[-1].sensor_type = 'Compass'
    vhc_obj.sensor_array[-1].add_sensor_field('name', '"compass"')
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
    vhc_obj.sensor_array[-1].sensor_type = 'Emitter'
    vhc_obj.sensor_array[-1].add_sensor_field('name', '"emitter"')
    vhc_obj.sensor_array[-1].add_sensor_field('channel', '2')
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
    vhc_obj.sensor_array[-1].sensor_type = 'GPS'
    sim_environment.agent_vehicles_list.append(vhc_obj)

    # ----- Define PEDESTRIANS:
    # No pedestrians in this test.

    # ----- Fog:
    sim_environment.fog = WebotsFog()
    sim_environment.fog.visibility_range = 700.0

    # ----- Road Disturbances:
    # No road disturbance

    # ----- Initial State Configurations:
    sim_environment.initial_state_config_list.append(
        InitialStateConfig(item=ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                                                item_index=0,
                                                item_state_index=WebotsVehicle.STATE_ID_VELOCITY_Y),
                           value=ego_init_speed_m_s))

    # ----- Controller Parameters:
    # Ego Target Path:
    target_pos_list = [[lane_number_to_lat_pos(3, 'N') + ego_enter_lateral_offset, -1000.0],
                       [lane_number_to_lat_pos(3, 'N') + ego_enter_lateral_offset,
                        lane_start_long_pos('S') + ego_enter_longitudinal_offset],
                       [lane_start_long_pos('W') - ego_exit_longitudinal_offset,
                        lane_number_to_lat_pos(3, 'W') + ego_exit_lateral_offset],
                       [1000.0, lane_number_to_lat_pos(3, 'W') + ego_exit_lateral_offset]]
    sim_environment.ego_target_path = target_pos_list[:]
    for target_pos in target_pos_list:
        sim_environment.controller_params_list.append(
            WebotsControllerParameter(vehicle_id=1,
                                      parameter_name='target_position',
                                      parameter_data=target_pos))

    # Agent Target Path:
    target_pos_list = [[lane_number_to_lat_pos(3, 'S') + agent_1_enter_lateral_offset, 1000.0],
                       [lane_number_to_lat_pos(3, 'S') + agent_1_enter_lateral_offset,
                        lane_start_long_pos('N') - agent_1_enter_longitudinal_offset],
                       [lane_start_long_pos('E') + agent_1_exit_longitudinal_offset,
                        lane_number_to_lat_pos(3, 'E') + agent_1_exit_lateral_offset],
                       [-1000.0, lane_number_to_lat_pos(3, 'E') + agent_1_exit_lateral_offset]]
    for target_pos in target_pos_list:
        sim_environment.controller_params_list.append(
            WebotsControllerParameter(vehicle_id=2,
                                      parameter_name='target_position',
                                      parameter_data=target_pos))

    # ----- Stop before collision config:
    # N/A

    # ----- Heart Beat Configuration:
    sim_environment.heart_beat_config = HeartBeatConfig(sync_type=HeartBeatConfig.WITHOUT_SYNC,
                                                        period_ms=2000)

    # ----- View Follow configuration:
    sim_environment.view_follow_config = \
        ViewFollowConfig(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                         item_index=0,
                         position=[sim_environment.ego_vehicles_list[0].current_position[0],
                                   sim_environment.ego_vehicles_list[0].current_position[1] + 3.0,
                                   sim_environment.ego_vehicles_list[0].current_position[2] - 15.0],
                         rotation=[0, -1, 0, math.pi])

    # ----- Data Log Configurations:
    sim_environment.data_log_description_list.append(
        ItemDescription(item_type=ItemDescription.ITEM_TYPE_TIME, item_index=0, item_state_index=0))
    for vhc_ind in range(len(sim_environment.ego_vehicles_list) + len(sim_environment.agent_vehicles_list)):
        sim_environment.data_log_description_list.append(
            ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                            item_index=vhc_ind,
                            item_state_index=WebotsVehicle.STATE_ID_POSITION_X))
        sim_environment.data_log_description_list.append(
            ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                            item_index=vhc_ind,
                            item_state_index=WebotsVehicle.STATE_ID_POSITION_Y))
        sim_environment.data_log_description_list.append(
            ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                            item_index=vhc_ind,
                            item_state_index=WebotsVehicle.STATE_ID_ORIENTATION))
        sim_environment.data_log_description_list.append(
            ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                            item_index=vhc_ind,
                            item_state_index=WebotsVehicle.STATE_ID_SPEED))
        sim_environment.data_log_description_list.append(
            ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                            item_index=vhc_ind,
                            item_state_index=WebotsVehicle.STATE_ID_ACCELERATION))
        sim_environment.data_log_description_list.append(
            ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                            item_index=vhc_ind,
                            item_state_index=WebotsVehicle.STATE_ID_YAW_RATE))

    sim_environment.data_log_description_list.append(
        ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE_CONTROL,
                        item_index=0,
                        item_state_index=ItemDescription.VEHICLE_CONTROL_THROTTLE))
    for ped_ind in range(len(sim_environment.pedestrians_list)):
        sim_environment.data_log_description_list.append(
            ItemDescription(item_type=ItemDescription.ITEM_TYPE_PEDESTRIAN,
                            item_index=ped_ind,
                            item_state_index=WebotsVehicle.STATE_ID_POSITION_X))
        sim_environment.data_log_description_list.append(
            ItemDescription(item_type=ItemDescription.ITEM_TYPE_PEDESTRIAN,
                            item_index=ped_ind,
                            item_state_index=WebotsVehicle.STATE_ID_POSITION_Y))

    sim_environment.data_log_period_ms = EXPERIMENT_WORLD_TIME_STEP_MS

    # ----- Detection Evaluation Configurations:
    sim_environment.detection_evaluation_config_list.append(DetectionEvaluationConfig(vehicle_id=1,
                                                                                      sensor_type='overall',
                                                                                      eval_type='localization',
                                                                                      eval_alg='d_square'))
    sim_environment.detection_evaluation_config_list[-1].add_target_object('vehicle', 2)

    sim_environment.detection_evaluation_config_list.append(DetectionEvaluationConfig(vehicle_id=1,
                                                                                      sensor_type='radar',
                                                                                      eval_type='localization',
                                                                                      eval_alg='d_square'))
    sim_environment.detection_evaluation_config_list[-1].add_target_object('vehicle', 2)

    sim_environment.detection_evaluation_config_list.append(DetectionEvaluationConfig(vehicle_id=1,
                                                                                      sensor_type='camera',
                                                                                      eval_type='localization',
                                                                                      eval_alg='d_square'))
    sim_environment.detection_evaluation_config_list[-1].add_target_object('vehicle', 2)

    # ----- Visibility Evaluation configurations:
    sim_environment.visibility_evaluation_config_list.append(
        VisibilityConfig(sensor=VisibilitySensor(hor_fov=math.pi / 3.0,
                                                 max_range=100,
                                                 position=(0.0, 0.0, 1.3),
                                                 x_rotation=0.0),
                         object_list=[('Car', 2)],
                         vehicle_id=1))

    # ----- Periodic Reporting Configurations:
    sim_environment.periodic_reporting_config_list.append(
        PeriodicReportingConfig(item_type='vehicle',
                                item_id=PeriodicReportingConfig.ALL_IDS,
                                report_type='position',
                                reporting_period=PeriodicReportingConfig.REPORT_AT_EVERY_STEP))
    sim_environment.periodic_reporting_config_list.append(
        PeriodicReportingConfig(item_type='vehicle',
                                item_id=PeriodicReportingConfig.ALL_IDS,
                                report_type='rotation',
                                reporting_period=PeriodicReportingConfig.REPORT_AT_EVERY_STEP))
    sim_environment.periodic_reporting_config_list.append(
        PeriodicReportingConfig(item_type='vehicle',
                                item_id=PeriodicReportingConfig.ALL_IDS,
                                report_type='box_corners',
                                reporting_period=PeriodicReportingConfig.REPORT_ONLY_ONCE))

    # ----- Create Trajectory dictionary for later reference:
    sim_environment.populate_simulation_trace_dict()
    return sim_environment


def run_single_test(sim_config, parameters_dict):
    """Execute one simulation with given configuration."""
    test_run_success = False
    retry_count = 0
    extended_trajectory = []
    simulator_instance = None
    sim_env_configurator = None
    sim_environment = None
    while not test_run_success and retry_count < 3:
        retry_count += 1
        try:
            sim_environment = define_sim_environment(parameters_dict)

            sim_env_configurator = SimEnvironmentConfigurator(sim_config=sim_config)
            (is_connected, simulator_instance) = sim_env_configurator.connect()
            if not is_connected:
                raise ValueError('Could not connect!')
            sim_env_configurator.setup_sim_environment(sim_environment)
            trajectory = sim_env_configurator.run_simulation_get_trace()
            trajectory_extensions = []
            for vhc_ind in range(1, len(sim_environment.agent_vehicles_list) + 1):
                trajectory_extensions.append(
                    experiment_tools.TrajectoryExtensionData(
                        data=experiment_tools.TrajectoryExtensionData.EGO_TO_OBJECT_DISTANCE,
                        related_item=ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                                                     item_index=vhc_ind,
                                                     item_state_index=0)))
            for vhc_ind in range(1, len(sim_environment.agent_vehicles_list) + 1):
                trajectory_extensions.append(
                    experiment_tools.TrajectoryExtensionData(
                        data=experiment_tools.TrajectoryExtensionData.EGO_TO_OBJECT_FUTURE_DISTANCE,
                        related_item=ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                                                     item_index=vhc_ind,
                                                     item_state_index=0)))
                trajectory_extensions.append(
                    experiment_tools.TrajectoryExtensionData(
                        data=experiment_tools.TrajectoryExtensionData.EGO_TO_OBJECT_FUTURE_MIN_DISTANCE_TIME,
                        related_item=ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                                                     item_index=vhc_ind,
                                                     item_state_index=0)))

            extended_trajectory = experiment_tools.extend_trajectory_with_requested_data(
                trajectory=trajectory, sim_environment=sim_environment, list_of_new_data=trajectory_extensions)
            test_run_success = True
        except Exception as ex:
            print('ERROR: ' + repr(ex))
            # WerFault.exe is something which generally pops up when Webots crashes on Windows
            sim_config_tools.kill_process_by_name('WerFault.exe')
            sim_config_tools.kill_webots_by_name()
            time.sleep(2.0)
            sim_config_tools.start_webots(sim_config.world_file, False)
            time.sleep(2.0)
            del sim_env_configurator
            sim_env_configurator = None
            sim_environment = None

    return extended_trajectory, simulator_instance, sim_environment


def run_ca_tests(exp_to_run=None, ca_strength=2):
    """Run CA tests. If tests_to_run is given run those otherwise run all."""
    exp_file_path = os.path.dirname(os.path.realpath(__file__))
    exp_file_name = os.path.splitext(os.path.basename(__file__))[0]
    environment_config_dict = experiment_tools.load_environment_configuration(exp_file_path=exp_file_path,
                                                                              exp_file_name=exp_file_name)
    sim_config = SIM_CONFIG
    sim_config.world_file = environment_config_dict['world_file_path'] + environment_config_dict['world_file_name']

    dict_of_parameter_dict = \
        covering_array_utilities.load_parameters_from_covering_array(environment_config_dict,
                                                                     parameter_name_type_dict=PARAMETER_NAME_TYPE_DICT,
                                                                     ca_strength=ca_strength,
                                                                     exp_type='CA',
                                                                     exp_to_run=exp_to_run)

    if environment_config_dict['is_save_exp_results_file']:
        results_file_name = environment_config_dict['exp_results_folder'] + environment_config_dict['exp_short_name']\
                            + '_ca_' + str(ca_strength) + '_way_TEST_RESULTS.csv'
        # If the results file exists, use it so that you can only change the values for the experiments you run.
        try:
            results_data_frame = covering_array_utilities.load_experiment_results_data(results_file_name)
        except Exception as ex:
            print('Results file not found, creating. (Err Code: {})'.format(repr(ex)))
            ca_exp_file_name = environment_config_dict['exp_config_folder'] + environment_config_dict['exp_short_name']\
                + '_ca_' + str(ca_strength) + '_way.csv'
            exp_data_frame = covering_array_utilities.load_experiment_data(ca_exp_file_name, header_line_count=6)
            results_data_frame = exp_data_frame.copy()
    else:
        # Those will never be used. Added just to suppress warnings.
        results_file_name = None
        results_data_frame = None
    data = {}
    for key, value in dict_of_parameter_dict.items():
        (trajectory, simulator_instance, sim_environment) = run_single_test(sim_config=sim_config,
                                                                            parameters_dict=value)
        data[key] = trajectory[:]
        print('Exp[{}] is finished'.format(key))
        if environment_config_dict['is_save_trajectory_files']:
            trajectory_file_name_prefix = environment_config_dict['exp_short_name'] + '_CA_trajectory'
            trajectory_file_name = \
                environment_config_dict['trajectory_log_folder'] + trajectory_file_name_prefix + '_{}.pkl'.format(key)
            trajectory_tools.save_trajectory_to_file(trajectory, trajectory_file_name)
            print('Exp[{}] trajectory saved.'.format(key))
        if environment_config_dict['is_save_exp_results_file']:
            covering_array_utilities.save_experiment_results(results_file_name, results_data_frame)
    return data


def run_for_falsification(exp_index, parameter_array, falsif_run_number, staliro_run_count, sim_duration,
                          falsif_test_type):
    exp_index = int(round(exp_index))
    exp_file_path = os.path.dirname(os.path.realpath(__file__))
    exp_file_name = os.path.splitext(os.path.basename(__file__))[0]
    environment_config_dict = experiment_tools.load_environment_configuration(exp_file_path=exp_file_path,
                                                                              exp_file_name=exp_file_name)
    sim_config = SIM_CONFIG
    sim_config.sim_duration_ms = sim_duration
    sim_config.world_file = environment_config_dict['world_file_path'] + environment_config_dict['world_file_name']
    dict_of_parameter_dict = \
        covering_array_utilities.load_parameters_from_covering_array(environment_config_dict,
                                                                     parameter_name_type_dict=PARAMETER_NAME_TYPE_DICT,
                                                                     ca_strength=2,
                                                                     exp_type='CA',
                                                                     exp_to_run=exp_index)
    parameter_dict = dict_of_parameter_dict[exp_index]
    for (param_ind, param_value) in enumerate(parameter_array):
        parameter_dict[FALSIFICATION_PARAMETER_INDEX_MAP[param_ind]] = param_value
    print('Running Experiment {}'.format(exp_index))
    try:
        # Try catch: because flush() fails when it is called from matlab.
        sys.stdout.flush()
    except Exception as ex:
        print('ERROR in ivjournal_test_1: ' + repr(ex))
    (trajectory, simulator_instance, sim_environment) = run_single_test(sim_config=sim_config,
                                                                        parameters_dict=parameter_dict)
    if environment_config_dict['is_save_trajectory_files']:
        trajectory_file_name = environment_config_dict['trajectory_log_folder'] + \
                               'fals_{}_{}_{}_exp_{}_trajectory.pkl'.format(falsif_test_type,
                                                                            int(round(falsif_run_number)),
                                                                            int(round(staliro_run_count)),
                                                                            exp_index)
        trajectory_tools.save_trajectory_to_file(trajectory, trajectory_file_name)

    if environment_config_dict['is_save_exp_results_file']:
        results_file_name = environment_config_dict['exp_results_folder'] + environment_config_dict['exp_short_name']\
                            + '_ca_' + str(2) + '_way_TEST_RESULTS.csv'
        # If the results file exists, use it so that you can only change the values for the experiments you run.
        try:
            results_data_frame = covering_array_utilities.load_experiment_results_data(results_file_name)
        except Exception as ex:
            print('Results file not found, creating. (Err Code: {})'.format(repr(ex)))
            ca_exp_file_name = environment_config_dict['exp_config_folder'] + \
                environment_config_dict['exp_short_name'] + '_ca_' + str(2) + '_way.csv'
            exp_data_frame = covering_array_utilities.load_experiment_data(ca_exp_file_name, header_line_count=6)
            results_data_frame = exp_data_frame.copy()
        covering_array_utilities.save_experiment_results(results_file_name, results_data_frame)

    return experiment_tools.npArray2Matlab(trajectory)


def run_falsification(exp_index, parameter_array, falsif_run_number, staliro_run_count, sim_duration, falsif_test_type):
    exp_index = int(round(exp_index))
    exp_file_path = os.path.dirname(os.path.realpath(__file__))
    exp_file_name = os.path.splitext(os.path.basename(__file__))[0]
    environment_config_dict = experiment_tools.load_environment_configuration(exp_file_path=exp_file_path,
                                                                              exp_file_name=exp_file_name)
    sim_config = SIM_CONFIG
    sim_config.sim_duration_ms = sim_duration
    sim_config.world_file = environment_config_dict['world_file_path'] + environment_config_dict['world_file_name']

    parameter_dict = DEFAULT_PARAMETERS_DICT

    for (param_ind, param_value) in enumerate(parameter_array):
        parameter_dict[FALSIFICATION_PARAMETER_INDEX_MAP[param_ind]] = param_value
    print('Running Experiment {}'.format(exp_index))
    try:
        # Try catch: because flush() fails when it is called from matlab.
        sys.stdout.flush()
    except Exception as ex:
        print('ERROR in ivjournal_test_3: ' + repr(ex))
    (trajectory, simulator_instance, sim_environment) = run_single_test(sim_config=sim_config,
                                                                        parameters_dict=parameter_dict)
    if environment_config_dict['is_save_trajectory_files']:
        trajectory_file_name = environment_config_dict['trajectory_log_folder'] + \
                               'fals_{}_{}_{}_exp_{}_trajectory.pkl'.format(falsif_test_type,
                                                                            int(round(falsif_run_number)),
                                                                            int(round(staliro_run_count)),
                                                                            exp_index)
        trajectory_tools.save_trajectory_to_file(trajectory, trajectory_file_name)

    if environment_config_dict['is_save_exp_results_file']:
        results_file_name = environment_config_dict['exp_results_folder'] + environment_config_dict['exp_short_name']\
                            + '_ca_' + str(2) + '_way_TEST_RESULTS.csv'
        # If the results file exists, use it so that you can only change the values for the experiments you run.
        try:
            results_data_frame = covering_array_utilities.load_experiment_results_data(results_file_name)
        except Exception as ex:
            print('Results file not found, creating. (Err Code: {})'.format(repr(ex)))
            ca_exp_file_name = environment_config_dict['exp_config_folder'] + \
                environment_config_dict['exp_short_name'] + '_ca_' + str(2) + '_way.csv'
            exp_data_frame = covering_array_utilities.load_experiment_data(ca_exp_file_name, header_line_count=6)
            results_data_frame = exp_data_frame.copy()
        covering_array_utilities.save_experiment_results(results_file_name, results_data_frame)

    return experiment_tools.npArray2Matlab(trajectory)


def plot_detection_data_from_trajectory(trajectory, sim_environment):
    import matplotlib.pyplot as plt

    time_ind = sim_environment.simulation_trace_dict[(ItemDescription.ITEM_TYPE_TIME, 0, 0)]
    data_plot_list = []
    for detection_eval_config in sim_environment.detection_evaluation_config_list:
        for target_obj_ind in range(len(detection_eval_config.target_objs)):
            data_plot_list.append((detection_eval_config.get_target_obj_info_as_dictionary_key(target_obj_ind),
                                   detection_eval_config))
    for visibility_eval_config in sim_environment.visibility_evaluation_config_list:
        for target_obj_ind in range(len(visibility_eval_config.object_list)):
            data_plot_list.append((visibility_eval_config.get_target_obj_info_as_dictionary_key(target_obj_ind),
                                   visibility_eval_config))
    f, axarr = plt.subplots(len(data_plot_list), figsize=(10, int(len(data_plot_list)*2)))
    for (item_ind, data_plot_item) in enumerate(data_plot_list):
        traj_key = data_plot_item[0]
        traj_ind = sim_environment.simulation_trace_dict[traj_key]
        data = data_plot_item[1]
        if isinstance(data, VisibilityConfig):
            title_str = 'Visibility:'
        else:
            title_str = 'Detection Perf.:'
        for element in traj_key:
            title_str += ' '
            title_str += str(element)
        axarr[item_ind].plot(np.transpose(trajectory[:, time_ind]), np.transpose(trajectory[:, traj_ind]))
        axarr[item_ind].set_title(title_str)
    plt.show()


def plot_risk_data_from_trajectory(trajectory, sim_environment):
    import matplotlib.pyplot as plt

    time_ind = sim_environment.simulation_trace_dict[(ItemDescription.ITEM_TYPE_TIME, 0, 0)]
    cur_ind = len(sim_environment.simulation_trace_dict) - 1
    data_plot_list = []
    for agent_vhc in sim_environment.agent_vehicles_list:
        data_plot_list.append((cur_ind, 'distance'))
        cur_ind += 1
    for agent_vhc in sim_environment.agent_vehicles_list:
        data_plot_list.append((cur_ind, 'min distance in future'))
        cur_ind += 1
        data_plot_list.append((cur_ind, 'time for min distance'))
        cur_ind += 1
    data_plot_list.append((
        sim_environment.simulation_trace_dict[(ItemDescription.ITEM_TYPE_VEHICLE_CONTROL, 0,
                                               ItemDescription.VEHICLE_CONTROL_THROTTLE)], 'throttle'))
    f, axarr = plt.subplots(len(data_plot_list), figsize=(10, int(len(data_plot_list)*2)))
    for (item_ind, data_plot_item) in enumerate(data_plot_list):
        traj_ind = data_plot_item[0]
        title_str = data_plot_item[1]
        axarr[item_ind].plot(np.transpose(trajectory[:, time_ind]), np.transpose(trajectory[:, traj_ind]))
        axarr[item_ind].set_title(title_str)
    plt.show()


def plot_angular_velocity_from_trajectory(trajectory, sim_environment):
    import matplotlib.pyplot as plt

    time_ind = sim_environment.simulation_trace_dict[(ItemDescription.ITEM_TYPE_TIME, 0, 0)]
    data_plot_list = []
    for (agent_ind, agent_vhc) in enumerate(sim_environment.agent_vehicles_list):
        data_plot_list.append((
            sim_environment.simulation_trace_dict[(ItemDescription.ITEM_TYPE_VEHICLE, agent_ind+1,
                                                   WebotsVehicle.STATE_ID_YAW_RATE)], 'angular velocity'))
    f, axarr = plt.subplots(len(data_plot_list), figsize=(10, int(len(data_plot_list)*2)))
    for (item_ind, data_plot_item) in enumerate(data_plot_list):
        traj_ind = data_plot_item[0]
        title_str = data_plot_item[1]
        axarr[item_ind].plot(np.transpose(trajectory[:, time_ind]), np.transpose(trajectory[:, traj_ind]))
        axarr[item_ind].set_title(title_str)
    plt.show()


def main():
    exp_file_path = os.path.dirname(os.path.realpath(__file__))
    exp_file_name = os.path.splitext(os.path.basename(__file__))[0]
    environment_config_dict = experiment_tools.load_environment_configuration(exp_file_path=exp_file_path,
                                                                              exp_file_name=exp_file_name)
    sim_config = SIM_CONFIG
    sim_config.world_file = environment_config_dict['world_file_path'] + environment_config_dict['world_file_name']
    parameters_dict = DEFAULT_PARAMETERS_DICT

    # parameter_array = [20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # parameter_array = [16.1421, -0.0239, 0.0810, -0.0368, 0.1593, -0.0159, 1.0811, 0.0315, 0.2373, 51.4693, 10.7839]
    parameter_array = [16.2204, 0.1911, -0.4351, -0.1201, 0.6786, 0.2153, 0.9048, -0.2676, 0.3415, 53.9384, 12.2735]
    parameter_array = [26.2204, -0.7411, 5.4351, -0.1201, 0.6786, 2.2153, 20.9048, -2.2676, 10.3415, 53.9384, 16.2735]
    for (param_ind, param_value) in enumerate(parameter_array):
        parameters_dict[FALSIFICATION_PARAMETER_INDEX_MAP[param_ind]] = param_value

    (trajectory, simulator_instance, sim_environment) = run_single_test(sim_config=sim_config,
                                                                        parameters_dict=parameters_dict)
    plot_risk_data_from_trajectory(trajectory, sim_environment)


if __name__ == "__main__":
    main()
