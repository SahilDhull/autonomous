"""Defines various classes and tool functions used for simulation configurations.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


import sys
import os
import subprocess
import pickle
import datetime
from Sim_ATAV.simulation_control.sim_data import SimData
from Sim_ATAV.simulation_control.webots_road import WebotsRoad
from Sim_ATAV.simulation_control.webots_vehicle import WebotsVehicle
from Sim_ATAV.simulation_control.webots_pedestrian import WebotsPedestrian


class RunConfig(object):
    """RunConfig class defines configuration for execution of a simulation."""
    def __init__(self):
        self.VUT_CONTROLLER = 'simple_controller'
        self.DUMMY_CONTROLLER = 'dummy_vhc_nn_control'
        self.DUMMY_POS_LAT = 3.5
        self.DUMMY_POS_LON = 0.0
        self.simulation_run_mode = SimData.SIM_TYPE_FAST_NO_GRAPHICS


class SimulationConfig(object):
    """SimulationConfig class defines configuration for a simulation.
    A simulation configuration also contains run configuration array."""
    def __init__(self, world_no=1):
        self.world_file = '../Webots_Projects/worlds/test_world_{}.wbt'.format(world_no)
        self.server_port = 10020 + world_no
        self.server_ip = '127.0.0.1'
        self.run_config_arr = []
        self.sim_duration_ms = 50000
        self.sim_step_size = 10
        self.simulator_instance_pid = None

    def insert_run_config(self, run_config):
        """Inserts a run configuration to the simulation configuration."""
        import copy
        self.run_config_arr.append(copy.deepcopy(run_config))


class ExperimentConfig(object):
    """ExperimentConfig class contains parallelization / log etc. information for running a series of simulations."""
    def __init__(self):
        self.NUM_WEIGHTS = 418
        self.WEIGHTS_MIN = -5.0
        self.WEIGHTS_MAX = 5.0
        self.MAX_ITERATIONS = 1
        self.INITIAL_SIGMA = 3.0
        self.NUM_OF_PARALLEL_WEBOTS = 1
        self.LOG_FILE_PREFIX = '.\\logs\\' + datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_")
        self.WEBOT_RESTART_THRESHOLD = 110
        self.SIMULATION_RUN_FUNC = None
        self.sim_config_arr = []
        self.cma_results = None
        self.POP_SIZE = None


class WithExtraArgs(object):
    """WithExtraArgs class is used to pass extra arguments to the parallelized function calls."""
    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def __call__(self, idx):
        return self.func(self.args[0][idx], self.args[1][idx])


def create_road_object():
    """Returns a WebotsRoad object."""
    road_obj = WebotsRoad()
    return road_obj


def create_vehicle_object():
    """Returns a WebotsVehicle object."""
    vehicle_obj = WebotsVehicle()
    return vehicle_obj

def create_pedestrian_object():
    """Returns a WebotsPedestrian object."""
    pedestrian_obj = WebotsPedestrian()
    return pedestrian_obj

def start_webots(world_file, minimized):
    if sys.platform == 'win32':  # Windows
        cmd = 'C:\\Program Files\\Webots\\msys64\\mingw64\\bin\\webots'
    elif sys.platform == 'darwin':  # Mac OS
        cmd = '/Applications/Webots.app/webots'
    else:  # Linux
        cmd = 'webots'
    simulator_instance = 0
    if minimized:
        params = ['--mode=fast', '--minimize', '--batch', world_file]
        simulator_instance = subprocess.Popen([cmd, params[0], params[1], params[2], params[3]])
    else:
        try:  # Because We get mkl dll could not be found error if Webots starts as a subprocess.
            os.startfile(cmd)
        except:
            print("Error while starting Webots: {}".format(sys.exc_info()[0]))

    return simulator_instance


def start_webots_from_path(world_file, minimized, path):
    cmd = path
    if minimized:
        params = ['--mode=fast', '--minimize', '--batch', world_file]
        simulator_instance = subprocess.Popen([cmd, params[0], params[1], params[2], params[3]])
    else:
        params = ['--mode=fast', '--batch', world_file]
        simulator_instance = subprocess.Popen([cmd, params[0], params[1], params[2]])

    return simulator_instance


def kill_process_id(pid):
    if sys.platform == 'win32':  # Windows
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(pid)])
    else:
        subprocess.call(['kill', str(pid)])


def kill_webots(simulator_instance):
    try:
        kill_process_id(simulator_instance.pid)
    except:
        print('Cannot kill webots')


def kill_webots_pid(pid):
    try:
        kill_process_id(pid)
    except:
        print('Cannot kill webots')

def kill_webots_by_name():
    try:
        if sys.platform == 'win32':  # Windows
            os.system("taskkill /im webots.exe")
        else:
            os.system("killall /-9 webots")
    except:
        print('Cannot kill webots')

def kill_process_by_name(process_name):
    try:
        if sys.platform == 'win32':  # Windows
            os.system("taskkill /im " + process_name)
        else:
            os.system("killall -9 " + process_name)
    except:
        print('Cannot kill process')

def get_weights_from_file(results_file_name):
    exp_conf = pickle.load(open(results_file_name, 'rb'))
    return exp_conf.cma_results[0]  # Best solution is at index 0. See Python CMA Documentation.


def get_experiment_config_from_file(file_name):
    exp_conf = pickle.load(open(file_name, 'rb'))
    return exp_conf


def save_results_to_file(res, results_file_name):
    pickle.dump(res, open(results_file_name, 'wb'))


def run_results(results_file_name):
    exp_config = get_experiment_config_from_file(results_file_name)
    for i in range(len(exp_config.sim_config_arr[0].run_config_arr)):
        exp_config.sim_config_arr[0].run_config_arr[i].simulation_run_mode = SimData.SIM_TYPE_REAL_TIME
    simulator_instance = start_webots(exp_config.sim_config_arr[0].world_file, False)
    weights = exp_config.cma_results[0]
    # print weights
    exp_config.SIMULATION_RUN_FUNC(weights, exp_config.sim_config_arr[0])
    kill_webots_pid(simulator_instance.pid)
