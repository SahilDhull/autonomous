"""Case Study for IV '2018
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import time
import sys
import os
import pandas as pd
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_PATH + '/../../')
from Sim_ATAV.simulation_control.sim_data import SimData
from Sim_ATAV.simulation_control.webots_sensor import WebotsSensor
from Sim_ATAV.simulation_control.webots_fog import WebotsFog
from Sim_ATAV.simulation_control.webots_pedestrian import WebotsPedestrian
from Sim_ATAV.simulation_control.webots_vehicle import WebotsVehicle
from Sim_ATAV.simulation_control.heart_beat import HeartBeatConfig, HeartBeat
from Sim_ATAV.simulation_control.item_description import ItemDescription
from Sim_ATAV.simulation_configurator.simulation_communication_interface import SimulationCommunicationInterface
from Sim_ATAV.simulation_configurator import sim_config_tools
from Sim_ATAV.simulation_configurator import covering_array_utilities
from Sim_ATAV.simulation_configurator import trajectory_tools
from Sim_ATAV.simulation_configurator import experiment_tools


FILE_PATH = os.path.dirname(os.path.realpath(__file__))
LANE_WIDTH = 3.5
IS_ONLINE_CLASSIFIER = False

PED_COLOR_DICT = {1:[1.0, 0.0, 0.0],
                  2:[0.0, 1.0, 0.0],
                  3:[0.0, 0.0, 1.0],
                  4:[1.0, 1.0, 1.0],
                  5:[0.0, 0.0, 0.0]}

CAR_COLOR_DICT = {1:[1.0, 0.0, 0.0],
                  2:[0.0, 1.0, 0.0],
                  3:[0.0, 0.0, 1.0],
                  4:[1.0, 1.0, 1.0],
                  5:[0.0, 0.0, 0.0]}

INIT_VALS_EGO_POSITION = 0
INIT_VALS_AGENT_1_POSITION = 1
INIT_VALS_PED1_SPEED_IND = 2
INIT_VALS_COUNT = 3

try:
    EXP_FILE_PATH = FILE_PATH
    CURRENT_ENV_CONFIG_INDEX = 0
    env_config_file_name = EXP_FILE_PATH + '/iv2018_exp1_env_config.csv'
    env_config_data_frame = pd.read_csv(env_config_file_name, index_col=0)
    env_config_all_fields = env_config_data_frame.iloc[[CURRENT_ENV_CONFIG_INDEX]]
    EXP_SHORT_NAME = env_config_all_fields['exp_short_name'].iloc[0]
    exp_config_folder_is_absolute = env_config_all_fields['exp_config_folder_is_absolute'].iloc[0]
    if exp_config_folder_is_absolute:
        EXP_CONFIG_FOLDER = env_config_all_fields['exp_config_folder'].iloc[0] + '/'
    else:
        EXP_CONFIG_FOLDER = EXP_FILE_PATH + '/' + env_config_all_fields['exp_config_folder'].iloc[0] + '/'
    world_file_folder_is_absolute = env_config_all_fields['world_file_folder_is_absolute'].iloc[0]
    if world_file_folder_is_absolute:
        WORLD_FILE_PATH = env_config_all_fields['world_file_folder'].iloc[0] + '/'
    else:
        WORLD_FILE_PATH = EXP_FILE_PATH + '/' + env_config_all_fields['world_file_folder'].iloc[0] + '/'
    WORLD_FILE_NAME = env_config_all_fields['world_file_name'].iloc[0]
    SAVE_TRAJECTORY_FILES = env_config_all_fields['is_save_trajectory_files'].iloc[0]
    traj_log_folder_is_absolute = env_config_all_fields['traj_log_folder_is_absolute'].iloc[0]
    if traj_log_folder_is_absolute:
        TRAJ_LOG_FOLDER = env_config_all_fields['traj_log_folder'].iloc[0] + '/'
    else:
        TRAJ_LOG_FOLDER = EXP_FILE_PATH + '/' + env_config_all_fields['traj_log_folder'].iloc[0] + '/'
    SAVE_EXP_RESULTS_FILE = env_config_all_fields['is_save_exp_results_file'].iloc[0]
    exp_results_folder_is_absolute = env_config_all_fields['exp_results_folder_is_absolute'].iloc[0]
    if exp_results_folder_is_absolute:
        EXP_RESULTS_FOLDER = env_config_all_fields['exp_results_folder'].iloc[0] + '/'
    else:
        EXP_RESULTS_FOLDER = EXP_FILE_PATH + '/' + env_config_all_fields['exp_results_folder'].iloc[0] + '/'
except:
    EXP_SHORT_NAME = 'iv2018_exp1'
    EXP_CONFIG_FOLDER = EXP_FILE_PATH + '/exp_config/'
    WORLD_FILE_PATH = EXP_FILE_PATH + '/../../Webots_Projects/worlds/'
    WORLD_FILE_NAME = EXP_SHORT_NAME + '_world.wbt'
    SAVE_TRAJECTORY_FILES = False
    TRAJ_LOG_FOLDER = EXP_FILE_PATH + '/trajectory_logs/'
    SAVE_EXP_RESULTS_FILE = False
    EXP_RESULTS_FOLDER = EXP_FILE_PATH + '/exp_results'

COLUMNS_LIST = ['agent_vhc_1_model', 'agent_vhc_1_color',\
                'agent_vhc_2_model', 'agent_vhc_2_color',\
                'agent_vhc_3_model', 'agent_vhc_3_color',\
                'agent_vhc_4_model', 'agent_vhc_4_color',\
                'agent_vhc_5_model', 'agent_vhc_5_color',\
                'ped_1_shirt_color', 'ped_1_pants_color',\
                'fog', \
                'ego_vhc_position', \
                'agent_vhc_1_position', \
                'ped_1_speed']

def run_single_test(current_experiment, sim_config, existing_simulator_instance, init_vals):
    test_run_success = False
    retry_count = 0
    trajectory = []
    min_det_perf = 10000
    simulator_instance = None
    while not test_run_success and retry_count < 3:
        retry_count += 1
        try:
            pedestrian_list = []
            ego_vhc_list = []
            dummy_vhc_list = []
            road_disturbance_list = []

            # First, try to connect once. If you can't connect, assume Webots is not already running and start Webots.
            comm_interface = SimulationCommunicationInterface(sim_config.server_ip, sim_config.server_port, 3)
            simulator_instance = None  # simulator_instance will keep the pid of the Webots if we need to start it here.
            if comm_interface.comm_module is None:
                comm_interface = None
                if existing_simulator_instance is not None:
                    sim_config_tools.kill_webots_pid(existing_simulator_instance.pid)
                    time.sleep(1.0)
                simulator_instance = sim_config_tools.start_webots(sim_config.world_file, False)
                time.sleep(10.0)

            # Define VEHICLES:
            cur_vhc_id = 0
            # Ego vehicle
            vhc_obj = sim_config_tools.create_vehicle_object()
            vhc_long_position = init_vals[INIT_VALS_EGO_POSITION]
            vhc_obj.current_position = [0.0, 0.35, vhc_long_position]
            vhc_obj.current_orientation = 0.0
            vhc_obj.rotation = [0.0, 1.0, 0.0, 0.0]
            cur_vhc_id += 1
            vhc_obj.vhc_id = cur_vhc_id
            vhc_obj.color = [1.0, 1.0, 0.0]
            vhc_obj.set_vehicle_model('ToyotaPrius')
            vhc_obj.controller = 'emerg_coll_avoidance_2'
            vhc_obj.controller_arguments.append(str(1.0))
            vhc_obj.controller_arguments.append(str(IS_ONLINE_CLASSIFIER))
            vhc_obj.controller_arguments.append(str(vhc_obj.current_position[0]))
            vhc_obj.controller_arguments.append(str(cur_vhc_id))
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
            ego_vhc_list.append(vhc_obj)

            # Dummy vehicles
            # Dummy 1:
            cur_vhc_id += 1
            vhc_obj = sim_config_tools.create_vehicle_object()
            vhc_model = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_{}_model'.format(int(cur_vhc_id-1)))
            vhc_color_enum = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_{}_color'.format(int(cur_vhc_id-1)))
            vhc_long_position = vhc_long_position = init_vals[INIT_VALS_AGENT_1_POSITION]
            x_pos = LANE_WIDTH
            vhc_obj.current_position = [x_pos, 0.35, vhc_long_position]
            vhc_obj.current_orientation = 0.0
            vhc_obj.rotation = [0.0, 1.0, 0.0, 0.0]
            vhc_obj.vhc_id = cur_vhc_id
            vhc_obj.set_vehicle_model(vhc_model)
            vhc_obj.color = CAR_COLOR_DICT[vhc_color_enum]
            dummy_vhc_list.append(vhc_obj)

            # Dummy 2:
            cur_vhc_id += 1
            vhc_obj = sim_config_tools.create_vehicle_object()
            vhc_model = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_{}_model'.format(int(cur_vhc_id-1)))
            vhc_color_enum = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_{}_color'.format(int(cur_vhc_id-1)))
            vhc_long_position = 120.0
            x_pos = 0.0
            vhc_obj.current_position = [x_pos, 0.35, vhc_long_position]
            vhc_obj.current_orientation = 0.0
            vhc_obj.rotation = [0.0, 1.0, 0.0, 0.0]
            vhc_obj.vhc_id = cur_vhc_id
            vhc_obj.set_vehicle_model(vhc_model)
            vhc_obj.color = CAR_COLOR_DICT[vhc_color_enum]
            dummy_vhc_list.append(vhc_obj)

            # Dummy 3:
            cur_vhc_id += 1
            vhc_obj = sim_config_tools.create_vehicle_object()
            vhc_model = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_{}_model'.format(int(cur_vhc_id-1)))
            vhc_color_enum = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_{}_color'.format(int(cur_vhc_id-1)))
            vhc_long_position = 90.0
            x_pos = -LANE_WIDTH - 0.5
            vhc_obj.current_position = [x_pos, 0.35, vhc_long_position]
            vhc_obj.current_orientation = 0.0
            vhc_obj.rotation = [0.0, 1.0, 0.0, 0.0]
            vhc_obj.vhc_id = cur_vhc_id
            vhc_obj.set_vehicle_model(vhc_model)
            vhc_obj.color = CAR_COLOR_DICT[vhc_color_enum]
            dummy_vhc_list.append(vhc_obj)

            # Dummy 4:
            cur_vhc_id += 1
            vhc_obj = sim_config_tools.create_vehicle_object()
            vhc_model = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_{}_model'.format(int(cur_vhc_id-1)))
            vhc_color_enum = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_{}_color'.format(int(cur_vhc_id-1)))
            vhc_long_position = 70.0
            x_pos = -LANE_WIDTH - 0.5
            vhc_obj.current_position = [x_pos, 0.35, vhc_long_position]
            vhc_obj.current_orientation = 0.0
            vhc_obj.rotation = [0.0, 1.0, 0.0, 0.0]
            vhc_obj.vhc_id = cur_vhc_id
            vhc_obj.set_vehicle_model(vhc_model)
            vhc_obj.color = CAR_COLOR_DICT[vhc_color_enum]
            dummy_vhc_list.append(vhc_obj)

            # Dummy 5:
            cur_vhc_id += 1
            vhc_obj = sim_config_tools.create_vehicle_object()
            vhc_model = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_{}_model'.format(int(cur_vhc_id-1)))
            vhc_color_enum = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_{}_color'.format(int(cur_vhc_id-1)))
            vhc_long_position = 53.0
            x_pos = -LANE_WIDTH - 0.5
            vhc_obj.current_position = [x_pos, 0.35, vhc_long_position]
            vhc_obj.current_orientation = 0.0
            vhc_obj.rotation = [0.0, 1.0, 0.0, 0.0]
            vhc_obj.vhc_id = cur_vhc_id
            vhc_obj.set_vehicle_model(vhc_model)
            vhc_obj.color = CAR_COLOR_DICT[vhc_color_enum]
            dummy_vhc_list.append(vhc_obj)

            # Dummy 6:
            cur_vhc_id += 1
            vhc_obj = sim_config_tools.create_vehicle_object()
            vhc_model = 'ToyotaPriusSimple'
            vhc_color_enum = 5
            vhc_long_position = 140.0
            x_pos = LANE_WIDTH
            vhc_obj.current_position = [x_pos, 0.35, vhc_long_position]
            vhc_obj.current_orientation = 0.0
            vhc_obj.rotation = [0.0, 1.0, 0.0, 0.0]
            vhc_obj.vhc_id = cur_vhc_id
            vhc_obj.set_vehicle_model(vhc_model)
            vhc_obj.color = CAR_COLOR_DICT[vhc_color_enum]
            dummy_vhc_list.append(vhc_obj)

            # Define PEDESTRIANS:
            cur_ped_id = 0
            # Ped 1
            pedestrian = WebotsPedestrian()
            cur_ped_id += 1
            pedestrian.ped_id = cur_ped_id
            ped_shirt_color_selection = \
                covering_array_utilities.get_field_value_for_current_experiment(current_experiment, \
                                                    'ped_{}_shirt_color'.format(pedestrian.ped_id))
            ped_pants_color_selection = \
                covering_array_utilities.get_field_value_for_current_experiment(current_experiment, \
                                                    'ped_{}_pants_color'.format(pedestrian.ped_id))
            ped_x_pos = -5.5
            ped_y_pos = 58
            ped_speed = init_vals[INIT_VALS_PED1_SPEED_IND]
            pedestrian.current_position = [ped_x_pos, 1.3, ped_y_pos]
            pedestrian.shirt_color = PED_COLOR_DICT[ped_shirt_color_selection]
            pedestrian.pants_color = PED_COLOR_DICT[ped_pants_color_selection]
            pedestrian.target_speed = ped_speed
            pedestrian.trajectory = [ped_x_pos, ped_y_pos, 1.0, ped_y_pos, ped_x_pos, ped_y_pos, ped_x_pos, ped_y_pos+100]
            pedestrian_list.append(pedestrian)

            # Ped 2
            pedestrian = WebotsPedestrian()
            cur_ped_id += 1
            pedestrian.ped_id = cur_ped_id
            ped_x_pos = -5.5
            ped_y_pos = 145.0
            ped_speed = 1.0
            pedestrian.current_position = [ped_x_pos, 1.3, ped_y_pos]
            pedestrian.target_speed = ped_speed
            pedestrian.trajectory = [ped_x_pos, ped_y_pos, -ped_x_pos, ped_y_pos]
            pedestrian_list.append(pedestrian)

            # Ped 3
            pedestrian = WebotsPedestrian()
            cur_ped_id += 1
            pedestrian.ped_id = cur_ped_id
            ped_x_pos = 5.5
            ped_y_pos = 148.0
            ped_speed = 1.0
            pedestrian.current_position = [ped_x_pos, 1.3, ped_y_pos]
            pedestrian.target_speed = ped_speed
            pedestrian.trajectory = [ped_x_pos, ped_y_pos, -ped_x_pos, ped_y_pos]
            pedestrian_list.append(pedestrian)

            # Fog:
            fog = WebotsFog()
            visibility_range = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'fog')
            fog.visibility_range = float(visibility_range)

            # Communicate:
            # try:
            if comm_interface is None:
                comm_interface = SimulationCommunicationInterface(server_address=sim_config.server_ip,
                                                                  server_port=sim_config.server_port,
                                                                  max_connection_retry=100)
                time.sleep(0.5)
            if comm_interface is not None:
                if not comm_interface.add_fog_to_simulation(fog, True):
                    print('ADD FOG error')
                time.sleep(0.01)
                for road_disturbance in road_disturbance_list:
                    if not comm_interface.add_road_disturbance_to_simulation(road_disturbance, True):
                        print('ADD ROAD DISTURBANCE error')
                    time.sleep(0.01)
                for pedestrian in pedestrian_list:
                    if not comm_interface.add_pedestrian_to_simulation(pedestrian, True):
                        print('ADD PEDESTRIAN error')
                    time.sleep(0.01)  # Give some time to Webots to modify simulation environment

                for vhc_obj in ego_vhc_list:
                    if not comm_interface.add_vehicle_to_simulation(vhc_obj, False, True):
                        print('ADD EGO VEHICLE error')
                    time.sleep(0.01)  # Give some time to Webots to modify simulation environment

                for vhc_obj in dummy_vhc_list:
                    if not comm_interface.add_vehicle_to_simulation(vhc_obj, True, True):
                        print('ADD DUMMY VEHICLE error')
                    time.sleep(0.01)  # Give some time to Webots to modify simulation environment

                if not comm_interface.set_initial_state(ItemDescription.ITEM_TYPE_VEHICLE,
                                                        0,
                                                        WebotsVehicle.STATE_ID_VELOCITY_Y,
                                                        20.0):
                    print("COULD NOT SET INITIAL VELOCITY FOR EGO VEHICLE!")

                if not comm_interface.set_stop_before_collision_item(ItemDescription.ITEM_TYPE_VEHICLE,
                                                                     ItemDescription.ITEM_INDEX_ALL,
                                                                     ItemDescription.ITEM_TYPE_VEHICLE,
                                                                     0):
                    print("STOP BEFORE COLLISION setting error")

                if not comm_interface.set_stop_before_collision_item(ItemDescription.ITEM_TYPE_PEDESTRIAN,
                                                                     ItemDescription.ITEM_INDEX_ALL,
                                                                     ItemDescription.ITEM_TYPE_VEHICLE,
                                                                     0):
                    print("STOP BEFORE COLLISION setting error")

                if not comm_interface.set_heart_beat_config(HeartBeatConfig.WITHOUT_SYNC, 2000):
                    print('SET HEART BEAT error')

                if not comm_interface.set_view_follow_point(ItemDescription.ITEM_TYPE_VEHICLE, 0):
                    print('SET VIEW FOLLOW POINT error')

                if not comm_interface.set_view_point_position([ego_vhc_list[0].current_position[0], \
                                                               ego_vhc_list[0].current_position[1] + 5.0, \
                                                               ego_vhc_list[0].current_position[2] - 20.0]):
                    print('SET VIEW POINT POSITION error')

                # Describe the simulation trajectory:
                traj_dict = {}
                cur_traj_ind = 0
                # Time -> trajectory
                if not comm_interface.add_data_log_description(ItemDescription.ITEM_TYPE_TIME, 0, 0):
                    print('ADD DATA LOG error')
                traj_dict[(ItemDescription.ITEM_TYPE_TIME, 0, 0)] = cur_traj_ind
                cur_traj_ind += 1
                # vehicle(s) x, y, orientation, speed -> trajectory
                for vhc_ind in range(len(ego_vhc_list) + len(dummy_vhc_list)):
                    if not comm_interface.add_data_log_description(ItemDescription.ITEM_TYPE_VEHICLE,
                                                                   vhc_ind,
                                                                   WebotsVehicle.STATE_ID_POSITION_X):
                        print('ADD DATA LOG error')
                    traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, vhc_ind, WebotsVehicle.STATE_ID_POSITION_X)] = cur_traj_ind
                    cur_traj_ind += 1
                    if not comm_interface.add_data_log_description(ItemDescription.ITEM_TYPE_VEHICLE,
                                                                   vhc_ind,
                                                                   WebotsVehicle.STATE_ID_POSITION_Y):
                        print('ADD DATA LOG error')
                    traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, vhc_ind, WebotsVehicle.STATE_ID_POSITION_Y)] = cur_traj_ind
                    cur_traj_ind += 1
                    if not comm_interface.add_data_log_description(ItemDescription.ITEM_TYPE_VEHICLE,
                                                                   vhc_ind,
                                                                   WebotsVehicle.STATE_ID_ORIENTATION):
                        print('ADD DATA LOG error')
                    traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, vhc_ind, WebotsVehicle.STATE_ID_ORIENTATION)] = cur_traj_ind
                    cur_traj_ind += 1
                    if not comm_interface.add_data_log_description(ItemDescription.ITEM_TYPE_VEHICLE,
                                                                   vhc_ind,
                                                                   WebotsVehicle.STATE_ID_SPEED):
                        print('ADD DATA LOG error')
                    traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, vhc_ind, WebotsVehicle.STATE_ID_SPEED)] = cur_traj_ind
                    cur_traj_ind += 1

                # pedestrian(s) x, y -> trajectory
                for ped_ind in range(len(pedestrian_list)):
                    if not comm_interface.add_data_log_description(ItemDescription.ITEM_TYPE_PEDESTRIAN,
                                                                   ped_ind,
                                                                   WebotsPedestrian.STATE_ID_POSITION_X):
                        print('ADD DATA LOG error')
                    traj_dict[(ItemDescription.ITEM_TYPE_PEDESTRIAN, ped_ind, WebotsVehicle.STATE_ID_POSITION_X)] = cur_traj_ind
                    cur_traj_ind += 1
                    if not comm_interface.add_data_log_description(ItemDescription.ITEM_TYPE_PEDESTRIAN,
                                                                   ped_ind,
                                                                   WebotsPedestrian.STATE_ID_POSITION_Y):
                        print('ADD DATA LOG error')
                    traj_dict[(ItemDescription.ITEM_TYPE_PEDESTRIAN, ped_ind, WebotsVehicle.STATE_ID_POSITION_Y)] = cur_traj_ind
                    cur_traj_ind += 1

                # collected perf type, which vehicle is collecting (index), performance of which object's detection (id).
                # !!! object's (id) is not equal to its index !!!
                for dummy_vhc in dummy_vhc_list:
                    if not comm_interface.add_data_log_description(ItemDescription.ITEM_TYPE_VEHICLE_DET_PERF,
                                                                   0,
                                                                   dummy_vhc.vhc_id):
                        print('ADD DATA LOG error')
                    traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE_DET_PERF, 0, dummy_vhc.vhc_id)] = cur_traj_ind
                    cur_traj_ind += 1

                for ped in pedestrian_list:
                    if not comm_interface.add_data_log_description(ItemDescription.ITEM_TYPE_PED_DET_PERF,
                                                                   0,
                                                                   ped.ped_id):
                        print('ADD DATA LOG error')
                    traj_dict[(ItemDescription.ITEM_TYPE_PED_DET_PERF, 0, ped.ped_id)] = cur_traj_ind
                    cur_traj_ind += 1

                trajectory_data_step = 10
                if not comm_interface.set_data_log_period_ms(trajectory_data_step):
                    print('SET DATA LOG PERIOD error')
                traj_dict['time_step'] = trajectory_data_step

                # Set the radio message broadcasts for the simulation.
                if not comm_interface.set_periodic_reporting('vehicle', 'position', 0, 0):
                    print('SET PERIODIC REPORTING error')
                if not comm_interface.set_periodic_reporting('vehicle', 'rotation', 0, 0):
                    print('SET PERIODIC REPORTING error')
                if not comm_interface.set_periodic_reporting('vehicle', 'box_corners', 0, -1):
                    print('SET PERIODIC REPORTING error')
                if not comm_interface.set_periodic_reporting('pedestrian', 'position', 0, 0):
                    print('SET PERIODIC REPORTING error')
                if not comm_interface.set_periodic_reporting('pedestrian', 'rotation', 0, 0):
                    print('SET PERIODIC REPORTING error')
                if not comm_interface.set_periodic_reporting('pedestrian', 'box_corners', 0, -1):
                    print('SET PERIODIC REPORTING error')

                # Start Simulation
                time.sleep(0.2)
                if not comm_interface.start_simulation(sim_config.sim_duration_ms,
                                                       sim_config.sim_step_size,
                                                       sim_config.run_config_arr[0].simulation_run_mode):
                    print('START SIMULATION error')
                time.sleep(8.0)
                simulation_continues = True
                while simulation_continues:
                    received_heart_beat = comm_interface.receive_heart_beat()
                    if received_heart_beat is not None:
                        if received_heart_beat.simulation_status == HeartBeat.SIMULATION_STOPPED:
                            simulation_continues = False

                collected_data = comm_interface.get_data_log()
                (trajectory, min_det_perf) = experiment_tools.extend_trajectory(collected_data, traj_dict, dummy_vhc_list)

                if existing_simulator_instance is not None:
                    if not comm_interface.restart_simulation():
                        print('RESTART SIMULATION error')
                    time.sleep(0.1)
                comm_interface.disconnect_from_simulator()
                del comm_interface
                comm_interface = None
                if existing_simulator_instance is None:
                    sim_config_tools.kill_webots_by_name()
                    time.sleep(0.3)
                    sim_config_tools.start_webots(sim_config.world_file, False)
                    time.sleep(2.0)
                test_run_success = True
            else:
                sim_config_tools.kill_process_by_name('WerFault.exe')  # This is something which generally pops up when Webots crashes
                sim_config_tools.kill_webots_by_name()
                time.sleep(2.0)
                sim_config_tools.start_webots(sim_config.world_file, False)
                time.sleep(2.0)
        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            sim_config_tools.kill_process_by_name('WerFault.exe')  # This is something which generally pops up when Webots crashes
            sim_config_tools.kill_webots_by_name()
            time.sleep(2.0)
            sim_config_tools.start_webots(sim_config.world_file, False)
            time.sleep(2.0)

    return trajectory, min_det_perf, simulator_instance

def get_initial_sample_and_range(exp_ind, exp_file_name, critical_param_list):
    exp_ind = int(round(exp_ind))
    data_frame = covering_array_utilities.load_experiment_data(exp_file_name)
    init_sample = []
    init_range = []
    for critical_param in critical_param_list:
        field_name = COLUMNS_LIST[critical_param]
        field_val = covering_array_utilities.get_experiment_field_value(data_frame, exp_ind, field_name)
        init_sample.append(field_val)
        if field_name in ['agent_vhc_1_position']:
            range_min = max(80.0, field_val - 10.0)
            range_max = min(110.0, field_val + 10.0)
        elif field_name in ['ego_vhc_position']:
            range_min = max(-10.0, field_val - 10.0)
            range_max = min(20.0, field_val + 10.0)
        elif field_name in ['ped_1_speed']:
            range_min = max(2.0, field_val - 1.0)
            range_max = min(5.0, field_val + 1.0)
        else:
            print('UNEXPECTED FIELD: '+field_name)
            range_min = 0.0
            range_max = 0.0
        init_range.append([range_min, range_max])
    return (init_sample, init_range)

def convert_model_enum_to_text(model_enum):
    if model_enum == 1:
        model_text = 'ToyotaPriusSimple'
    elif model_enum == 2:
        model_text = 'BmwX5Simple'
    elif model_enum == 3:
        model_text = 'CitroenCZeroSimple'
    elif model_enum == 4:
        model_text = 'LincolnMKZSimple'
    else:
        model_text = 'RangeRoverSportSVRSimple'
    return model_text

def convert_ped_distance_enum_to_value(ped_distance_enum):
    # (1, .. 5) -> (15, 20, 25, 30, 35)
    return 5*(ped_distance_enum+2)

def convert_vhc_speed_enum_to_value(vhc_speed_enum):
    # (1,2) -> (5, 10)
    return 5*vhc_speed_enum

def convert_vhc_distance_enum_to_value(vhc_distance_enum):
    # (1, .. 4) -> (50, 100, 150, 200)
    return 50*vhc_distance_enum

def convert_agent_2_pos_enum_to_value(vhc_position_enum):
    # (1, 2, 3, 4) -> (80, 90, 100, 110)
    return 70.0 + (vhc_position_enum*10.0)

def convert_ego_pos_enum_to_value(vhc_position_enum):
    # (1, 2, 3, 4) -> (-10, 0, 10, 20)
    return -20.0 + (vhc_position_enum*10.0)

def convert_fog_existence_enum_to_distance_value(fog_existence_enum):
    # (0, 1) -> (900, 100)
    if fog_existence_enum == 1:
        fog_distance = 100
    else:
        fog_distance = 900
    return fog_distance

def create_uniform_random_experiments(csv_file_name, num_of_experiments, random_number_seed_file=None, discretize_cont_params=True):
    if random_number_seed_file is None:
        experiment_tools.save_random_number_generator_state(csv_file_name + '.random_number_state.pkl')
    else:
        experiment_tools.restore_random_number_generator_state(random_number_seed_file)
    # type_range_conversion_func_dict is a fancy dictionary:
    # It has the type of the field, minimum and maximum range and the function to convert resulting value to the actual field value.
    # For example for side, we sample 1 or 2
    # and 1 should be converted to 'LEFT', 2 should be converted to 'RIGHT'
    # we give the name of the function which is doing that conversion (None if not needed).
    type_range_conversion_func_dict = {}
    agent_vhc_number_list = [1, 2, 3, 4, 5]
    for obj_num in agent_vhc_number_list:
        type_range_conversion_func_dict['agent_vhc_{}_model'.format(obj_num)] = (int, 1, 5, convert_model_enum_to_text)
        type_range_conversion_func_dict['agent_vhc_{}_color'.format(obj_num)] = (int, 1, 5, None)

    pedestrian_number_list = [1]
    for obj_num in pedestrian_number_list:
        type_range_conversion_func_dict['ped_{}_shirt_color'.format(obj_num)] = (int, 1, 5, None)
        type_range_conversion_func_dict['ped_{}_pants_color'.format(obj_num)] = (int, 1, 5, None)
    type_range_conversion_func_dict['fog'] = (int, 0, 1, convert_fog_existence_enum_to_distance_value)
    if discretize_cont_params:
        type_range_conversion_func_dict['agent_vhc_1_position'] = (int, 1, 4, convert_agent_2_pos_enum_to_value)
        type_range_conversion_func_dict['ego_vhc_position'] = (int, 1, 4, convert_ego_pos_enum_to_value)
        type_range_conversion_func_dict['ped_1_speed'] = (int, 2, 5, None)
    else:
        type_range_conversion_func_dict['agent_vhc_1_position'] = (float, 80.0, 110.0, None)
        type_range_conversion_func_dict['ego_vhc_position'] = (float, -10.0, 20.0, None)
        type_range_conversion_func_dict['ped_1_speed'] = (float, 2.0, 5.0, None)

    num_columns = len(COLUMNS_LIST)
    exp_data_list = []
    for _exp_ind in range(num_of_experiments):
        exp_data = [0.0]*num_columns
        for (param_ind, param_key) in enumerate(COLUMNS_LIST):
            (param_type, param_min, param_max, param_conv_func) = type_range_conversion_func_dict[param_key]
            param_val = experiment_tools.get_random_number_in_range(min_val=param_min, max_val=param_max, num_type=param_type)
            if param_conv_func is not None:
                param_val = param_conv_func(param_val)
            exp_data[param_ind] = param_val
        exp_data_list.append(exp_data)
    exp_data_frame = pd.DataFrame(exp_data_list, columns=COLUMNS_LIST)
    if csv_file_name is not None:
        exp_data_frame.to_csv(csv_file_name)
    return exp_data_frame

def run_test(sim_config, exp_file_name, results_file_name, trajectory_file_name_prefix, exp_type, exp_to_run=None):
    # Load the covering array experiments from a csv file.
    data = {}
    if exp_type == 'CA':
        exp_data_frame = covering_array_utilities.load_experiment_data(exp_file_name, header_line_count=6)
    else:
        exp_data_frame = covering_array_utilities.load_experiment_data(exp_file_name, header_line_count=0, index_col=0)
    num_of_experiments = len(exp_data_frame.index)
    # If the results file exists, use it so that you can only change the values for the experiments you run.
    try:
        results_data_frame = covering_array_utilities.load_experiment_results_data(results_file_name)
    except:
        results_data_frame = exp_data_frame.copy()
        results_data_frame = covering_array_utilities.add_column_to_data_frame(results_data_frame, 'perc_perf', 10000.0)

    #tests_robustness_list = 10000.0*np.ones(num_of_experiments, dtype=np.float)
    print('Number of experiments to run: {}'.format(num_of_experiments))
    simulator_instance = None
    if exp_to_run is None:
        experiment_set = range(num_of_experiments)
    else:
        experiment_set = list(exp_to_run)

    for exp_ind in experiment_set:
        print(' --- Running Experiment {} / {} ---'.format(exp_ind + 1, num_of_experiments))
        current_experiment = covering_array_utilities.get_experiment_all_fields(exp_data_frame, exp_ind)

        init_vals = [0.0]*INIT_VALS_COUNT
        ego_vhc_position = float(covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'ego_vhc_position'))
        init_vals[INIT_VALS_EGO_POSITION] = ego_vhc_position
        ped_1_speed = float(covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'ped_1_speed'))
        init_vals[INIT_VALS_PED1_SPEED_IND] = ped_1_speed
        agent_vhc_1_position = float(covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_1_position'))
        init_vals[INIT_VALS_AGENT_1_POSITION] = agent_vhc_1_position
        (trajectory, min_det_perf, _new_sim_instance) = run_single_test(current_experiment,
                                                                       sim_config,
                                                                       simulator_instance,
                                                                       init_vals)
        data[exp_ind] = trajectory[:]
        print('Exp {} is finished'.format(exp_ind + 1))
        if SAVE_TRAJECTORY_FILES:
            trajectory_file_name = TRAJ_LOG_FOLDER + trajectory_file_name_prefix + '_{}.pkl'.format(exp_ind)
            trajectory_tools.save_trajectory_to_file(trajectory, trajectory_file_name)
            print('Exp {} trajectory saved.'.format(exp_ind + 1))
        covering_array_utilities.set_experiment_field_value(results_data_frame, exp_ind, 'perc_perf', min_det_perf)
        if SAVE_EXP_RESULTS_FILE:
            covering_array_utilities.save_experiment_results(results_file_name, results_data_frame)

    return data

def run_selected_test(test_no, init_cond, sim_duration, exp_file_name, results_file_name):
    # Load the covering array experiments from csv file.
    test_no = int(round(test_no))
    sim_config = sim_config_tools.SimulationConfig(1)
    sim_config.run_config_arr.append(sim_config_tools.RunConfig())
    sim_config.run_config_arr[0].simulation_run_mode = SimData.SIM_TYPE_RUN
    sim_config.sim_duration_ms = sim_duration
    sim_config.sim_step_size = 10
    sim_config.world_file = WORLD_FILE_PATH + WORLD_FILE_NAME

    exp_data_frame = covering_array_utilities.load_experiment_data(exp_file_name)
    num_of_experiments = len(exp_data_frame.index)
    simulator_instance = None
    trajectory = []
    exp_ind = test_no
    print(' --- Running Experiment {} / {} ---'.format(exp_ind + 1, num_of_experiments))
    try:
        sys.stdout.flush()
    except:
        pass
    current_experiment = covering_array_utilities.get_experiment_all_fields(exp_data_frame, exp_ind)
    (trajectory, _min_det_perf, new_sim_instance) = run_single_test(current_experiment, \
                                                                   sim_config, \
                                                                   simulator_instance, \
                                                                   init_cond)
    if new_sim_instance is not None:  # sun_single_test function started a webots instance
        simulator_instance = new_sim_instance
    if SAVE_TRAJECTORY_FILES:
        trajectory_file_name = TRAJ_LOG_FOLDER + 'int_4_trajectory_{}.pkl'.format(exp_ind)
        trajectory_tools.save_trajectory_to_file(trajectory, trajectory_file_name)

    # If Webots was not already running (we started the process), then we kill it:
    if simulator_instance is not None:
        sim_config_tools.kill_webots_pid(simulator_instance.pid)

    return experiment_tools.npArray2Matlab(trajectory)

def run_selected_test_critical_params(test_no, SA_Run_Number, staliro_run_count, critical_params, init_cond, sim_duration, exp_file_name, results_file_name, falsif_test_type):
    test_no = int(round(test_no))
    sim_config = sim_config_tools.SimulationConfig(1)
    sim_config.run_config_arr.append(sim_config_tools.RunConfig())
    sim_config.run_config_arr[0].simulation_run_mode = SimData.SIM_TYPE_RUN
    sim_config.sim_duration_ms = sim_duration
    sim_config.sim_step_size = 10
    sim_config.world_file = WORLD_FILE_PATH + WORLD_FILE_NAME

    exp_data = get_exp_data_from_csv(exp_file_name, test_no, critical_params, init_cond)
    exp_data_frame = add_experiment_to_csv(results_file_name, exp_data)
    num_of_experiments = len(exp_data_frame.index)

    simulator_instance = None
    trajectory = []
    exp_ind = test_no
    print(' --- Running Experiment {} / {} ---'.format(exp_ind + 1, num_of_experiments))
    try:
        # Try catch: fails when it is called from matlab.
        sys.stdout.flush()
    except:
        pass

    current_experiment = covering_array_utilities.get_experiment_all_fields(exp_data_frame, num_of_experiments-1)  # use the last exp
    init_vals = [0.0]*INIT_VALS_COUNT
    ego_vhc_position = float(covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'ego_vhc_position'))
    init_vals[INIT_VALS_EGO_POSITION] = ego_vhc_position
    ped_1_speed = float(covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'ped_1_speed'))
    init_vals[INIT_VALS_PED1_SPEED_IND] = ped_1_speed
    agent_vhc_1_position = float(covering_array_utilities.get_field_value_for_current_experiment(current_experiment, 'agent_vhc_1_position'))
    init_vals[INIT_VALS_AGENT_1_POSITION] = agent_vhc_1_position
    (trajectory, min_det_perf, _new_sim_instance) = run_single_test(current_experiment,
                                                                   sim_config,
                                                                   simulator_instance,
                                                                   init_vals)
    if SAVE_TRAJECTORY_FILES:
        trajectory_file_name = TRAJ_LOG_FOLDER + 'fals_{}_{}_{}_exp_{}_trajectory_{}.pkl'.format(falsif_test_type, int(round(SA_Run_Number)), int(round(staliro_run_count)), exp_ind, num_of_experiments-1)
        trajectory_tools.save_trajectory_to_file(trajectory, trajectory_file_name)
    covering_array_utilities.set_experiment_field_value(exp_data_frame, num_of_experiments-1, 'perc_perf', min_det_perf)
    if SAVE_EXP_RESULTS_FILE:
        covering_array_utilities.save_experiment_results(results_file_name, exp_data_frame)

    return experiment_tools.npArray2Matlab(trajectory)

def add_experiment_to_csv(exp_file_name, exp_data):
    columns_list = ['agent_vhc_1_model', 'agent_vhc_1_color',\
                'agent_vhc_2_model', 'agent_vhc_2_color',\
                'agent_vhc_3_model', 'agent_vhc_3_color',\
                'agent_vhc_4_model', 'agent_vhc_4_color',\
                'agent_vhc_5_model', 'agent_vhc_5_color',\
                'ped_1_shirt_color', 'ped_1_pants_color',\
                'fog', \
                'ego_vhc_position', \
                'agent_vhc_1_position', \
                'ped_1_speed', 'perc_perf']
    exp_data_extended = [0.0]*len(columns_list)
    for data_ind in range(len(exp_data)):
        exp_data_extended[data_ind] = exp_data[data_ind]
    exp_data_extended[-1] = 10000  # perc_perf

    exp_data_list = []
    exp_data_list.append(exp_data_extended)
    new_data_frame = pd.DataFrame(exp_data_list, columns=columns_list)
    try:
        exp_data_frame = covering_array_utilities.load_experiment_data(exp_file_name, header_line_count=0, index_col=0)
        exp_data_frame = exp_data_frame.append(new_data_frame, ignore_index=True)
    except:
        exp_data_frame = new_data_frame
    if exp_file_name is not None:
        exp_data_frame.to_csv(exp_file_name)
    return exp_data_frame

def get_exp_data_from_csv(exp_file_name, exp_ind, indices_to_change, new_values):
    exp_data_frame = covering_array_utilities.load_experiment_data(exp_file_name)
    current_experiment = covering_array_utilities.get_experiment_all_fields(exp_data_frame, exp_ind)
    exp_data = []
    for param_name in COLUMNS_LIST:
        param_val = covering_array_utilities.get_field_value_for_current_experiment(current_experiment, param_name)
        exp_data.append(param_val)
    for (new_value_ind, param_ind) in enumerate(indices_to_change):
        exp_data[param_ind] = new_values[new_value_ind]
    return exp_data

def main():
    """Do combinatorial testing for an intersection scenario."""
    sim_config = sim_config_tools.SimulationConfig(1)
    sim_config.run_config_arr.append(sim_config_tools.RunConfig())
    sim_config.run_config_arr[0].simulation_run_mode = SimData.SIM_TYPE_RUN
    sim_config.sim_duration_ms = 10000
    sim_config.sim_step_size = 10
    sim_config.world_file = WORLD_FILE_PATH + WORLD_FILE_NAME
    #test_type = 'UR'
    test_type = 'CA'
    #test_type = 'UR_D_and_C'
    print('Test Type: ' + test_type)
    test_index_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    if test_type == 'UR':
        for test_ind in test_index_list:
            TEST_SIZE = 200
            print('Test index: {}'.format(test_ind))
            test_name = EXP_SHORT_NAME + '_UR_{}'.format(test_ind)
            exp_file_name = EXP_CONFIG_FOLDER + test_name + '_experiments.csv'
            create_uniform_random_experiments(exp_file_name, TEST_SIZE)
            results_file_name = EXP_RESULTS_FOLDER + test_name + '_TEST_RESULTS.csv'
            trajectory_file_name_prefix = test_name + '_trajectory'
            run_test(sim_config, exp_file_name, results_file_name, trajectory_file_name_prefix, test_type)
    elif test_type == 'UR_D_and_C':
        for test_ind in test_index_list:
            TEST_SIZE = 200
            print('Test index: {}'.format(test_ind))
            test_name = EXP_SHORT_NAME + '_Global_UR_{}'.format(test_ind)
            exp_file_name = EXP_CONFIG_FOLDER + test_name + '_experiments.csv'
            create_uniform_random_experiments(exp_file_name, TEST_SIZE, discretize_cont_params=False)
            results_file_name = EXP_RESULTS_FOLDER + test_name + '_TEST_RESULTS.csv'
            trajectory_file_name_prefix = test_name + '_trajectory'
            run_test(sim_config, exp_file_name, results_file_name, trajectory_file_name_prefix, test_type)
    elif test_type == 'CA':
        exp_file_name = EXP_CONFIG_FOLDER + 'straight_road_pedestrian_jump_ca_2_way.csv'
        results_file_name = EXP_RESULTS_FOLDER + 'straight_road_pedestrian_jump_ca_2_way_TEST_RESULTS.csv'
        trajectory_file_name_prefix = EXP_SHORT_NAME + '_CA_trajectory'
        run_test(sim_config, exp_file_name, results_file_name, trajectory_file_name_prefix, test_type)
    elif test_type == 'replay_CA':
        exp_file_name = EXP_CONFIG_FOLDER + 'straight_road_pedestrian_jump_ca_2_way_modified.csv'
        results_file_name = EXP_RESULTS_FOLDER + 'straight_road_pedestrian_jump_ca_2_way_replay_TEST_RESULTS.csv'
        trajectory_file_name_prefix = EXP_SHORT_NAME + '_replay_CA_trajectory'
        run_test(sim_config, exp_file_name, results_file_name, trajectory_file_name_prefix, 'CA', exp_to_run=[27])
    elif test_type == 'replay_Falsif':
        exp_file_name = EXP_CONFIG_FOLDER + '/replay_experiments/' + 'straight_road_pedestrian_jump_ca_2_way_Falsif_SA_Taliro_2.csv'
        results_file_name = EXP_RESULTS_FOLDER + '/replay_experiments/' + 'straight_road_pedestrian_jump_ca_2_way_Falsif_SA_Taliro_2_replay_TEST_RESULTS.csv'
        trajectory_file_name_prefix = EXP_SHORT_NAME + '_replay_trajectory_Falsif_SA'
        run_test(sim_config, exp_file_name, results_file_name, trajectory_file_name_prefix, test_type, exp_to_run=[55])

if __name__ == "__main__":
    main()
