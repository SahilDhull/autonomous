import os
import sys
import math
import time
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')  # This is needed for the calls from Matlab
from Sim_ATAV.simulation_control.sim_data import SimData
from Sim_ATAV.simulation_control.webots_sensor import WebotsSensor
from Sim_ATAV.simulation_control.webots_fog import WebotsFog
from Sim_ATAV.simulation_control.webots_pedestrian import WebotsPedestrian
from Sim_ATAV.simulation_control.webots_vehicle import WebotsVehicle
from Sim_ATAV.simulation_control.webots_road import WebotsRoad
from Sim_ATAV.simulation_control.webots_road_disturbance import WebotsRoadDisturbance
from Sim_ATAV.simulation_control.heart_beat import HeartBeatConfig
from Sim_ATAV.simulation_control.item_description import ItemDescription
from Sim_ATAV.simulation_control.webots_controller_parameter import WebotsControllerParameter
from Sim_ATAV.simulation_control.webots_sim_object import WebotsSimObject
from Sim_ATAV.simulation_configurator import sim_config_tools
from Sim_ATAV.simulation_configurator.sim_environment import SimEnvironment
from Sim_ATAV.simulation_control.initial_state_config import InitialStateConfig
from Sim_ATAV.simulation_configurator.view_follow_config import ViewFollowConfig
from Sim_ATAV.simulation_configurator.sim_environment_configurator import SimEnvironmentConfigurator
from Sim_ATAV.simulation_configurator import covering_array_utilities
from Sim_ATAV.simulation_configurator import experiment_tools


def run_test(ego_init_speed_m_s=10.0, ego_x_pos=20.0, pedestrian_speed=3.0, sim_duration=15000, for_matlab=False):
    """Runs a test with the given arguments"""

    sim_environment = SimEnvironment()
    # --- Add road
    road = WebotsRoad(number_of_lanes=3)
    road.rotation = [0, 1, 0, -math.pi / 2]
    road.position = [1000, 0.02, 0]
    road.length = 2000.0
    sim_environment.road_list.append(road)

    # ----- Define VEHICLES:
    # Ego vehicle
    vhc_obj = WebotsVehicle()
    vhc_obj.current_position = [ego_x_pos, 0.35, 0.0]
    vhc_obj.current_orientation = math.pi/2
    vhc_obj.rotation = [0.0, 1.0, 0.0, vhc_obj.current_orientation]
    vhc_obj.vhc_id = 1
    vhc_obj.color = [1.0, 1.0, 0.0]
    vhc_obj.set_vehicle_model('ToyotaPrius')
    vhc_obj.controller = 'automated_driving_with_fusion2'
    vhc_obj.is_controller_name_absolute = True
    vhc_obj.controller_arguments.append('Toyota')
    vhc_obj.controller_arguments.append('70.0')
    vhc_obj.controller_arguments.append('0.0')
    vhc_obj.controller_arguments.append('1')
    vhc_obj.controller_arguments.append('True')
    vhc_obj.controller_arguments.append('False')
    vhc_obj.controller_arguments.append('0')

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
    vhc_obj.sensor_array[-1].sensor_type = 'GPS'
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_type = 'Radar' # 'Radar' #'DelphiESR'
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.FRONT
    vhc_obj.sensor_array[-1].add_sensor_field('name', '"radar"')
    sim_environment.ego_vehicles_list.append(vhc_obj)

    # ----- Agent vehicles
    # Agent:
    vhc_obj = WebotsVehicle()
    vhc_obj.current_position = [300.0, 0.35, 3.5]
    vhc_obj.current_orientation = 0.0
    vhc_obj.rotation = [0.0, 1.0, 0.0, -math.pi/2]
    vhc_obj.vhc_id = 2
    vhc_obj.set_vehicle_model('TeslaModel3')
    vhc_obj.color = [1.0, 0.0,  0.0]
    vhc_obj.controller = 'path_and_speed_follower'
    vhc_obj.controller_arguments.append('20.0')
    vhc_obj.controller_arguments.append('True')
    vhc_obj.controller_arguments.append('3.5')
    vhc_obj.controller_arguments.append('2')
    vhc_obj.controller_arguments.append('False')
    vhc_obj.controller_arguments.append('False')
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
    vhc_obj.sensor_array[-1].sensor_type = 'GPS'
    sim_environment.agent_vehicles_list.append(vhc_obj)

    # ----- Define PEDESTRIANS:
    # Pedestrian 1
    pedestrian = WebotsPedestrian()
    pedestrian.ped_id = 1
    pedestrian.current_position = [50.0, 1.3, 0.0]
    pedestrian.shirt_color = [0.0, 0.0, 0.0]
    pedestrian.pants_color = [0.0, 0.0, 1.0]
    pedestrian.target_speed = pedestrian_speed
    pedestrian.trajectory = [50.0, 0.0, 80.0, -3.0, 200.0, 0.0]
    pedestrian.controller = 'pedestrian_control'
    sim_environment.pedestrians_list.append(pedestrian)

    # ----- Fog:
    sim_environment.fog = WebotsFog()
    sim_environment.fog.visibility_range = 700.0

    # ----- Road Disturbances:
    road_disturbance = WebotsRoadDisturbance()
    road_disturbance.disturbance_type = WebotsRoadDisturbance.TRIANGLE_DOUBLE_SIDED
    road_disturbance.rotation = [0, 1, 0, -math.pi / 2.0]
    road_disturbance.position = [40, 0, 0]
    road_disturbance.width = 3.5
    road_disturbance.length = 3
    road_disturbance.height = 0.04
    road_disturbance.inter_object_spacing = 0.5
    sim_environment.road_disturbances_list.append(road_disturbance)

    # ----- Stop sign:
    sim_obj = WebotsSimObject()
    sim_obj.object_name = 'StopSign'
    sim_obj.object_parameters.append(('translation', '40 0 6'))
    sim_obj.object_parameters.append(('rotation', '0 1 0 1.5708'))
    sim_environment.generic_sim_objects_list.append(sim_obj)

    # ----- Initial State Configurations:
    sim_environment.initial_state_config_list.append(
        InitialStateConfig(item=ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                                                item_index=0,
                                                item_state_index=WebotsVehicle.STATE_ID_VELOCITY_X),
                           value=ego_init_speed_m_s))

    # ----- Controller Parameters:
    # Ego Target Path:
    target_pos_list = [[-1000.0, 0.0],
                       [1000.0, 0.0]]

    for target_pos in target_pos_list:
        sim_environment.controller_params_list.append(
            WebotsControllerParameter(vehicle_id=1,
                                      parameter_name='target_position',
                                      parameter_data=target_pos))

    # Agent Target Path:
    target_pos_list = [[1000.0, 3.5],
                       [145.0, 3.5],
                       [110.0, -3.5],
                       [-1000.0, -3.5]]

    for target_pos in target_pos_list:
        sim_environment.controller_params_list.append(
            WebotsControllerParameter(vehicle_id=2,
                                      parameter_name='target_position',
                                      parameter_data=target_pos))

    # ----- Heart Beat Configuration:
    sim_environment.heart_beat_config = HeartBeatConfig(sync_type=HeartBeatConfig.WITHOUT_SYNC,
                                                        period_ms=2000)

    # ----- View Follow configuration:
    sim_environment.view_follow_config = \
        ViewFollowConfig(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                         item_index=0,
                         position=[sim_environment.ego_vehicles_list[0].current_position[0] - 15.0,
                                   sim_environment.ego_vehicles_list[0].current_position[1] + 3.0,
                                   sim_environment.ego_vehicles_list[0].current_position[2]],
                         rotation=[0.0, 1.0, 0.0, -sim_environment.ego_vehicles_list[0].current_orientation])

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

    for ped_ind in range(len(sim_environment.pedestrians_list)):
        sim_environment.data_log_description_list.append(
            ItemDescription(item_type=ItemDescription.ITEM_TYPE_PEDESTRIAN,
                            item_index=ped_ind,
                            item_state_index=WebotsVehicle.STATE_ID_POSITION_X))
        sim_environment.data_log_description_list.append(
            ItemDescription(item_type=ItemDescription.ITEM_TYPE_PEDESTRIAN,
                            item_index=ped_ind,
                            item_state_index=WebotsVehicle.STATE_ID_POSITION_Y))

    sim_environment.data_log_period_ms = 10

    # ----- Create Trajectory dictionary for later reference:
    sim_environment.populate_simulation_trace_dict()

    sim_config = sim_config_tools.SimulationConfig(1)
    sim_config.run_config_arr.append(sim_config_tools.RunConfig())
    sim_config.run_config_arr[0].simulation_run_mode = SimData.SIM_TYPE_RUN
    sim_config.sim_duration_ms = sim_duration
    sim_config.sim_step_size = 10
    sim_config.world_file = '../Webots_Projects/worlds/empty_world.wbt'

    sim_env_configurator = SimEnvironmentConfigurator(sim_config=sim_config)
    (is_connected, simulator_instance) = sim_env_configurator.connect(max_connection_retry=3)
    if not is_connected:
        raise ValueError('Could not connect!')
    sim_env_configurator.setup_sim_environment(sim_environment)
    trajectory = sim_env_configurator.run_simulation_get_trace()
    if for_matlab:
        trajectory = experiment_tools.npArray2Matlab(trajectory)
    time.sleep(1)  # Wait for Webots to reload the world.
    return trajectory


def run_covering_array_tests():
    """Runs all tests from the covering array csv file"""
    exp_file_name = 'TutorialExample_CA_2way.csv'  # csv file containing the test cases

    # Read all experiment into a table:
    exp_data_frame = covering_array_utilities.load_experiment_data(exp_file_name, header_line_count=6)

    # Decide number of experiments based on the number of entries in the table.
    num_of_experiments = len(exp_data_frame.index)

    trajectories_dict = {}  # A dictionary data structure to keep simulation traces.

    for exp_ind in range(num_of_experiments):  # For each test case
        # Read the current test case
        current_experiment = covering_array_utilities.get_experiment_all_fields(
            exp_data_frame, exp_ind)

        # Read the parameters from the current test case:
        ego_init_speed = float(
            covering_array_utilities.get_field_value_for_current_experiment(
                current_experiment, 'ego_init_speed'))
        ego_x_position = float(
            covering_array_utilities.get_field_value_for_current_experiment(
                current_experiment, 'ego_x_position'))
        pedestrian_speed = float(
            covering_array_utilities.get_field_value_for_current_experiment(
                current_experiment, 'pedestrian_speed'))

        print('Running test case {} of {} with parameters: {}, {}, {}.'.format(exp_ind+1, num_of_experiments, ego_init_speed, ego_x_position, pedestrian_speed))
        # Execute the test case and record the resulting simulation trace:
        trajectories_dict[exp_ind] = run_test(ego_init_speed_m_s=ego_init_speed,
                                              ego_x_pos=ego_x_position, pedestrian_speed=pedestrian_speed)
        time.sleep(2)  # Wait for Webots to reload the world.
    return trajectories_dict

# run_test()
# run_covering_array_tests()
