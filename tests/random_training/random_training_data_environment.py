"""Defines the ClassificationClient class
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
import math
import numpy as np
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_PATH + '/../../')
from Sim_ATAV.simulation_control.sim_data import SimData
from Sim_ATAV.simulation_control.webots_sensor import WebotsSensor
from Sim_ATAV.simulation_control.webots_pedestrian import WebotsPedestrian
from Sim_ATAV.simulation_control.heart_beat import HeartBeatConfig, HeartBeat
from Sim_ATAV.simulation_control.item_description import ItemDescription
from Sim_ATAV.common.coordinate_system import CoordinateSystem
from Sim_ATAV.simulation_configurator.simulation_communication_interface import SimulationCommunicationInterface
from Sim_ATAV.simulation_configurator import sim_config_tools


EXP_FILE_NAME = FILE_PATH + '/experiment_configs/intersection2_CA_3_way.csv'
RESULTS_FILE_NAME = FILE_PATH + '/experiment_configs/intersection2_CA_3_way_TEST_RESULTS.csv'
SHIRT_COLOR_DICT = {1:[1.0, 0.0, 0.0],
                    2:[0.0, 1.0, 0.0],
                    3:[0.0, 0.0, 0.0],
                    4:[1.0, 1.0, 1.0],
                    5:[0.0, 0.0, 0.0]}

CAR_COLOR_DICT = {1:[1.0, 0.0, 0.0],
                  2:[0.0, 1.0, 0.0],
                  3:[0.0, 0.0, 0.0],
                  4:[1.0, 1.0, 1.0],
                  5:[0.0, 0.0, 0.0]}

RECT_OF_INTEREST_HALF_WIDTH = 1.5
RECT_OF_INTEREST_DEPTH = 50
VEHICLE_AXIS_TO_FRONT_LEN = 3.68


def run_single_test(current_experiment, sim_config, existing_simulator_instance):
    pedestrian_list = []
    ego_vhc_list = []
    dummy_vhc_list = []

    # Define VEHICLES:
    # Ego vehicle
    cur_vhc_id = 1
    vhc_obj = sim_config_tools.create_vehicle_object()
    x_rand = np.random.rand(1)
    init_x_pos = x_rand[0]*(40.0) - 20.0
    vhc_obj.current_position = [init_x_pos, 0.4, -100.0]
    rot = 0.0
    vhc_obj.current_orientation = rot
    vhc_obj.rotation = [0.0, 1.0, 0.0, rot]
    vhc_obj.vhc_id = cur_vhc_id
    vhc_obj.color = [1.0, 1.0, 0.0]
    vhc_obj.vehicle_model = 'ToyotaPrius'
    vhc_obj.controller = 'training_data_generator' #'classification_evaluator'
    vhc_obj.controller_arguments.append(str(current_experiment*15))
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
    ego_vhc_list.append(vhc_obj)
    cur_vhc_id += 1

    # Dummy vehicles
    for dummy_ind in range(8):
        exists_rand = np.random.rand(1)
        if exists_rand > 0.5:
            # Dummy 1:
            vhc_obj = sim_config_tools.create_vehicle_object()
            x_rand = np.random.rand(1)
            x_pos = x_rand[0]*(80.0) - 40.0
            y_rand = np.random.rand(1)
            y_pos = y_rand[0]*(180.0) - 80.0
            vhc_obj.current_position = [x_pos, 0.4, y_pos]
            rot_rand = np.random.rand(1)
            rot = rot_rand*math.pi*2.0
            vhc_obj.current_orientation = rot
            vhc_obj.rotation = [0.0, 1.0, 0.0, rot]
            vhc_obj.vhc_id = cur_vhc_id
            model_rand = np.random.rand(1)
            if model_rand < 0.2:
                vhc_obj.vehicle_model = 'CitroenCZero'
            elif model_rand < 0.4:
                vhc_obj.vehicle_model = 'BmwX5'
            elif model_rand < 0.6:
                vhc_obj.vehicle_model = 'ToyotaPrius'
            elif model_rand < 0.8:
                vhc_obj.vehicle_model = 'LincolnMKZ'
            else:
                vhc_obj.vehicle_model = 'RangeRoverSportSVR'
            vhc_obj.color = np.random.rand(3)
            vhc_obj.controller = 'simple_controller'
            speed_rand = np.random.rand(1)
            speed = speed_rand[0]/2.0+0.5
            vhc_obj.controller_arguments.append(str(speed))
            vhc_obj.sensor_array.append(WebotsSensor())
            vhc_obj.sensor_array[0].sensor_location = WebotsSensor.CENTER
            vhc_obj.sensor_array[0].sensor_type = 'Compass'
            vhc_obj.sensor_array[0].add_sensor_field('name', '"vut_compass"')
            vhc_obj.sensor_array[0].add_sensor_field('yAxis', 'FALSE')
            vhc_obj.sensor_array.append(WebotsSensor())
            vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
            vhc_obj.sensor_array[-1].sensor_type = 'GPS'
            vhc_obj.sensor_array[-1].add_sensor_field('name', '"vut_gps"')
            vhc_obj.sensor_array.append(WebotsSensor())
            vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.CENTER
            vhc_obj.sensor_array[-1].sensor_type = 'Receiver'
            vhc_obj.sensor_array[-1].add_sensor_field('name', '"receiver"')
            vhc_obj.sensor_array[-1].add_sensor_field('channel', '1')
            dummy_vhc_list.append(vhc_obj)
            cur_vhc_id += 1

    # Define PEDESTRIANS:
    cur_ped_id = 1
    for _ped_ind in range(8):
        pedestrian = WebotsPedestrian()
        pedestrian.ped_id = cur_ped_id
        x_rand = np.random.rand(1)
        x_pos = x_rand[0]*(200.0) - 100.0
        y_rand = np.random.rand(1)
        y_pos = y_rand[0]*(200.0) - 100.0
        pedestrian.current_position = [x_pos, 1.3, y_pos]
        pedestrian.shirt_color = np.random.rand(3)
        pedestrian.pants_color = np.random.rand(3)
        dist_x_rand = np.random.rand(1)
        dist_y_rand = np.random.rand(1)
        pedestrian.trajectory = [x_pos, y_pos, x_pos + dist_x_rand[0]*50 - 25.0, y_pos + dist_y_rand[0]*50 - 25.0]
        #pedestrian.current_position = [-7.0, 1.3, -13.0]
        #pedestrian.trajectory = [-7.0, -13.0, -7.0, -30.0]
        pedestrian_list.append(pedestrian)
        cur_ped_id += 1

    # Communicate:
    # First, try to connect once. If you can't connect, assume Webots is not already running and start Webots.
    print('Will Try connecting Webots')
    comm_interface = SimulationCommunicationInterface(sim_config.server_ip, sim_config.server_port, 3)
    simulator_instance = None  # simulator_instance will keep the pid of the Webots if we need to start it here.
    if comm_interface.comm_module is None:
        del comm_interface
        comm_interface = None
        if existing_simulator_instance is not None:
            print('Will Kill Webots by PID')
            sim_config_tools.kill_webots_pid(existing_simulator_instance.pid)
            existing_simulator_instance = None
            time.sleep(0.5)

    # try:
    while comm_interface is None or comm_interface.comm_module is None:
        print('Will Kill Webots by name')
        sim_config_tools.kill_webots_by_name()  # Will only work on Windows
        time.sleep(5.0)
        print('Will Start Webots')
        simulator_instance = sim_config_tools.start_webots(sim_config.world_file, False)
        time.sleep(5.0)
        comm_interface = None
        print('Will Connect Webots')
        comm_interface = SimulationCommunicationInterface(server_address=sim_config.server_ip,
                                                          server_port=sim_config.server_port,
                                                          max_connection_retry=20)
        time.sleep(0.2)

    if comm_interface is not None and comm_interface.comm_module is not None:
        for pedestrian in pedestrian_list:
            if not comm_interface.add_pedestrian_to_simulation(pedestrian, True):
                print('ADD PEDESTRIAN error')
            time.sleep(0.1)  # Give some time to Webots to mofiy simulation environment

        for vhc_obj in ego_vhc_list:
            if not comm_interface.add_vehicle_to_simulation(vhc_obj, False, True):
                print('ADD VEHICLE error')
            time.sleep(0.1)  # Give some time to Webots to mofiy simulation environment

        for vhc_obj in dummy_vhc_list:
            if not comm_interface.add_vehicle_to_simulation(vhc_obj, True, True):
                print('ADD VEHICLE error')
            time.sleep(0.1)  # Give some time to Webots to mofiy simulation environment

        if not comm_interface.set_heart_beat_config(HeartBeatConfig.WITHOUT_SYNC, 1000):
            print('SET HEART BEAT error')

        if not comm_interface.set_view_follow_point(ItemDescription.ITEM_TYPE_VEHICLE, 0):
            print('SET VIEW FOLLOW POINT error')

        if not comm_interface.set_view_point_position([ego_vhc_list[0].current_position[0], \
                                                       ego_vhc_list[0].current_position[1] + 5.0, \
                                                       ego_vhc_list[0].current_position[2] - 20.0]):
            print('SET VIEW POINT POSITION error')

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

        # if not comm_interface.set_robustness_type(RobustnessComputation.ROB_TYPE_DISTANCE_TO_PEDESTRIAN):
        #     print('SET ROBUSTNESS TYPE failed.')

        if not comm_interface.start_simulation(sim_config.sim_duration_ms,
                                               sim_config.sim_step_size,
                                               sim_config.run_config_arr[0].simulation_run_mode):
            print('START SIMULATION error')
        simulation_continues = True
        while simulation_continues:
            received_heart_beat = comm_interface.receive_heart_beat()
            if received_heart_beat is not None:
                if received_heart_beat.simulation_status == HeartBeat.SIMULATION_STOPPED:
                    simulation_continues = False
                #else:
                #    comm_interface.send_continue_sim_command()

        if not comm_interface.restart_simulation():
            print('RESTART SIMULATION error')
        time.sleep(0.1)
        comm_interface.disconnect_from_simulator()
        del comm_interface
        comm_interface = None
    return simulator_instance

def npArray2Matlab(x):
    return x.tolist()
    #return matlab.double(x.tolist())
    #return matlab.double(x)

def rotate_point_ccw(point, theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.dot(np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]), point)

def rotate_rectangle_ccw(rect_corners, theta):
    new_rectangle = rect_corners[:]
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    for (ind, corner) in rect_corners:
        new_rectangle[ind] = np.dot(np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]), corner)
    return new_rectangle

def get_rectangle_of_interest(rect_corners, vhc_pos, vhc_length, vhc_orientation):
    new_rectangle = rotate_rectangle_ccw(rect_corners, vhc_orientation)
    new_rectangle = new_rectangle + [vhc_pos[CoordinateSystem.X_AXIS], vhc_pos[CoordinateSystem.Y_AXIS] + vhc_length]
    return new_rectangle

def is_pedestrian_in_rectangle_of_interest(vhc_pos, vhc_orientation, ped_pos, vhc_length, rect_half_width, rect_depth):
    ped_rotated = rotate_point_ccw(ped_pos-vhc_pos, -vhc_orientation)
    if -rect_half_width < ped_rotated[0] < rect_half_width and \
        vhc_length < ped_rotated[1] < vhc_length + rect_depth:
        in_rectangle = True
        distance = ped_rotated[1] - vhc_length
    else:
        in_rectangle = False
        distance = abs(ped_rotated[1] - vhc_length) + RECT_OF_INTEREST_DEPTH

    return in_rectangle, distance

def extend_trajectory(trajectory):
    new_trajectory = np.append(trajectory, np.zeros([len(trajectory), 2]), 1)
    min_dist = 10000.0
    for (traj_ind, traj_point) in enumerate(new_trajectory):
        vhc_pos = traj_point[1:3]
        vhc_orientation = traj_point[3]
        ped_pos = traj_point[5:7]
        (new_trajectory[traj_ind][-2], new_trajectory[traj_ind][-1]) = \
            is_pedestrian_in_rectangle_of_interest(vhc_pos, \
                                                   vhc_orientation, \
                                                   ped_pos, \
                                                   VEHICLE_AXIS_TO_FRONT_LEN, \
                                                   RECT_OF_INTEREST_HALF_WIDTH, \
                                                   RECT_OF_INTEREST_DEPTH)
        if new_trajectory[traj_ind][-2] and new_trajectory[traj_ind][-1] < min_dist:
            min_dist = new_trajectory[traj_ind][-1]
    print('In trajectory: Minimum distance: {}'.format(min_dist))
    return new_trajectory

def run_test(sim_config):
    # Load the covering array experiments from a csv file.
    num_of_experiments = 1000
    print('Number of experiments to run: {}'.format(num_of_experiments))
    simulator_instance = None
    for exp_ind in range(367, num_of_experiments):
        success = False
        counter = 0
        while (not success) and (counter < 20):
            counter += 1
            try:
                print(' --- Running Experiment {} / {} ---'.format(exp_ind + 1, num_of_experiments))
                (new_sim_instance) = run_single_test(exp_ind, sim_config, simulator_instance)
                sys.stdout.flush()
                if new_sim_instance is not None:  # sun_single_test function started a webots instance
                    simulator_instance = new_sim_instance
                success = True
            except:
                success = False
                print('RETRYING !!!')
                sys.stdout.flush()
                sim_config_tools.kill_webots_by_name()
                simulator_instance = None
                time.sleep(5.0)

        time.sleep(4.0)  # Give some time to Webots to reload the simulation environment

    # If Webots was not already running (we started the process), then we kill it:
    if simulator_instance is not None:
        sim_config_tools.kill_webots_pid(simulator_instance.pid)

    return True

def main():
    """Do combinatorial testing for an intersection scenario."""
    sim_config = sim_config_tools.SimulationConfig(1)
    sim_config.run_config_arr.append(sim_config_tools.RunConfig())
    sim_config.run_config_arr[0].simulation_run_mode = SimData.SIM_TYPE_REAL_TIME
    sim_config.sim_duration_ms = 15000
    sim_config.sim_step_size = 10
    sim_config.world_file = FILE_PATH + '/../Webots_Projects/worlds/city_intersection_simple.wbt'
    data = run_test(sim_config)
    print('data: {}'.format(data))

if __name__ == "__main__":
    main()
