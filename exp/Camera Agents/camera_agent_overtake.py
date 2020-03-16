import os
import sys
import math
import time
import  csv

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
# from path import *
# from grid import *

import numpy as np
import copy
import dubins
import shapely.geometry as geom

inf = 1e9


def RadiusofCurvature(start_pt, end_pt, turn_radius=20.0, step_size=1.0):
    """Generate points along a Dubins path connecting start point to end point.
    Format for input / output points: (x, y, angle)"""
    min_turn_radius = min(0.1, turn_radius)
    satisfied = False
    configurations = [start_pt, end_pt]
    while not satisfied:
        dubins_path = dubins.shortest_path(start_pt, end_pt, turn_radius)
        configurations, _ = dubins_path.sample_many(step_size)
        cex_found = False
        for configuration in configurations:
            if not (min(start_pt[0], end_pt[0]) - 0.1 <= configuration[0] <= max(start_pt[0], end_pt[0]) + 0.1 and
                    min(start_pt[1], end_pt[1]) - 0.1 <= configuration[1] <= max(start_pt[1], end_pt[1]) + 0.1):
                cex_found = True
                break
        satisfied = not cex_found
        if cex_found:
            # Decrease radius until finding a satisfying result.
            # We could do a binary search but that requires a termination condition.
            turn_radius = turn_radius*0.9
            if turn_radius < min_turn_radius:
                break
    if not satisfied:
        return 0.1
    return turn_radius


def cost(c1, pt1,pt2, off=0.0):
    r = RadiusofCurvature(pt1,pt2)
    return c1 + math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2) + 10.0/r + 10.0*abs(off)


def computeTargetPath():
    grid_points = []

    x1 = 500.0
    x2 = 0

    y1 = 0.0
    y2 = -40.0

    w = 3.6

    x_step = 5.0
    y_step = 0.4

    r_curv = 20.0
    x_ctr1 = 0.0
    y_ctr1 = -20.0
    x_ctr2 = 500.0
    y_ctr2 = -20.0
    st = math.floor((math.pi*r_curv)/x_step)        # steps to take in the curved part

    # 1st part
    for i in np.arange(x1,x2,-x_step):
        gp = []
        for j in np.arange(y1+w,y1-w,-y_step):
            gp.append([i,y1+round(j,2),math.pi])
        grid_points.append(gp)


    # 2nd part
    for i in range(st):
        gp = []
        theta = i*x_step/r_curv
        x_cur = x_ctr1 - r_curv*math.sin(theta)
        y_cur = y_ctr1 + r_curv*math.cos(theta)
        for j in np.arange(y1+w,y1-w,-y_step):
            gp.append([round(x_cur+j*math.sin(theta),2),round(y_cur-j*math.cos(theta),2),math.pi+theta])
        grid_points.append(gp)


    # 3rd part
    for i in np.arange(x2,x1,x_step):
        gp = []
        for j in np.arange(y1+w,y1-w,-y_step):
            gp.append([i,y2+round(j,2),0.0])
        grid_points.append(gp)

    # 4th part
    for i in range(st):
        gp = []
        theta = i*x_step/r_curv
        x_cur = x_ctr2 + r_curv*math.sin(theta)
        y_cur = y_ctr2 - r_curv*math.cos(theta)
        for j in np.arange(y1+w,y1-w,-y_step):
            gp.append([round(x_cur+j*math.sin(theta),2),round(y_cur-j*math.cos(theta),2),theta])
        grid_points.append(gp)

    #-----------Solve the circularity problem with theta------------------------
    # print(grid_points[0][9])

    travel_path = []
    total_steps = 1000


    p = []
    c = []
    X = round(2*w/y_step)
    Y = len(grid_points)

    for j in range(Y):
        k = []
        f = []
        for i in range(X):
            k.append(inf)
            f.append((-1,-1))
        c.append(k)
        p.append(f)

    c[0][9] = 0.0


    for i in  range(Y-1):
        for j in range(X):
            m1 = max(0,j-3)
            m2 = min(X-1,j+3)
            for k in range(m1,m2+1):
                cur_cost = 0;
                cur_cost = cost(c[i][j],grid_points[i][j],grid_points[i+1][k],abs(k-9)*0.4)
                if(c[i+1][k] > cur_cost):
                    c[i+1][k] = cur_cost
                    p[i+1][k] = (i,j)

    i= Y-1
    j = 9
    # print(type(grid_points[0][0][0]))
    
    while(p[i][j]!=(-1,-1)):
        travel_path = [[float(grid_points[i][j][0]),float(grid_points[i][j][1])]] + travel_path
        (i,j) = p[i][j]

    return travel_path

def run_test(ego_init_speed_m_s=10.0, ego_x_pos=20.0, pedestrian_speed=3.0, sim_duration=120000, for_matlab=False):
    """Runs a test with the given arguments"""

    sim_environment = SimEnvironment()
    # --- Add road
    # road = WebotsRoad(number_of_lanes=3)
    # road.rotation = [0, 1, 0, -math.pi / 2]
    # road.position = [500, 0.02, 0]
    # road.length = 500.0
    # sim_environment.road_list.append(road)

    # ----- Define VEHICLES:
    
    # Ego vehicle
    vhc_obj = WebotsVehicle()
    vhc_obj.current_position = [80.0, 0.35, 0.0]
    vhc_obj.current_orientation = math.pi/2
    vhc_obj.rotation = [0.0, 1.0, 0.0, vhc_obj.current_orientation]
    vhc_obj.current_orientation = -math.pi/2.0
    vhc_obj.rotation = [0.0, 1.0, 0.0, -math.pi/2.0]
    
    vhc_obj.vhc_id = 1
    vhc_obj.color = [1.0, 0.0, 0.0]
    vhc_obj.set_vehicle_model('TeslaModel3')
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
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_type = 'Camera' # 'Radar' #'DelphiESR'
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.TOP
    vhc_obj.sensor_array[-1].add_sensor_field('name', '"camera"')
    # sim_environment.ego_vehicles_list.append(vhc_obj)
    

    #############################################
    

    
    # ----- Agent vehicles
    # Agent:
    vhc_obj = WebotsVehicle()
    vhc_obj.current_position = [450.0, 0.35, 0.0]
    vhc_obj.current_orientation = 0.0
    vhc_obj.rotation = [0.0, 1.0, 0.0, -math.pi/2]
    vhc_obj.vhc_id = 2
    vhc_obj.set_vehicle_model('TeslaModel3')
    vhc_obj.color = [1.0, 0.0,  0.0]
    vhc_obj.controller = 'path_and_speed_follower'
    vhc_obj.controller_arguments.append('25.0')
    vhc_obj.controller_arguments.append('True')
    vhc_obj.controller_arguments.append('3.5')
    vhc_obj.controller_arguments.append('2')#vhc_id
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
    vhc_obj.sensor_array.append(WebotsSensor())
    vhc_obj.sensor_array[-1].sensor_type = 'Radar' # 'Radar' #'DelphiESR'
    vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.FRONT
    vhc_obj.sensor_array[-1].add_sensor_field('name', '"radar"')
    sim_environment.agent_vehicles_list.append(vhc_obj)


    '''
    vhc_obj = WebotsVehicle()
    vhc_obj.current_position = [310.0, 0.35, 3.0]
    vhc_obj.current_orientation = 0.0
    vhc_obj.rotation = [0.0, 1.0, 0.0, math.pi/2]
    vhc_obj.vhc_id = 3
    vhc_obj.set_vehicle_model('TeslaModel3')
    vhc_obj.color = [1.0, 0.0,  0.0]
    vhc_obj.controller = 'path_and_speed_follower'
    vhc_obj.controller_arguments.append('25.0')
    vhc_obj.controller_arguments.append('True')
    vhc_obj.controller_arguments.append('3.5')
    vhc_obj.controller_arguments.append('3')
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
    # vhc_obj.sensor_array.append(WebotsSensor())
    # vhc_obj.sensor_array[-1].sensor_type = 'Camera' # 'Radar' #'DelphiESR'
    # vhc_obj.sensor_array[-1].sensor_location = WebotsSensor.TOP
    # vhc_obj.sensor_array[-1].add_sensor_field('name', '"camera"')
    # sim_environment.ego_vehicles_list.append(vhc_obj)
    sim_environment.agent_vehicles_list.append(vhc_obj)
    '''
    

    # ----- Define PEDESTRIANS:
    # Pedestrian 1
    # pedestrian = WebotsPedestrian()
    # pedestrian.ped_id = 1
    # pedestrian.current_position = [50.0, 1.3, 0.0]
    # pedestrian.shirt_color = [0.0, 0.0, 0.0]
    # pedestrian.pants_color = [0.0, 0.0, 1.0]
    # pedestrian.target_speed = pedestrian_speed
    # pedestrian.trajectory = [50.0, 0.0, 80.0, -3.0, 200.0, 0.0]
    # pedestrian.controller = 'pedestrian_control'
    # sim_environment.pedestrians_list.append(pedestrian)

    # ----- Fog:
    # sim_environment.fog = WebotsFog()
    # sim_environment.fog.visibility_range = 700.0

    # ----- Road Disturbances:
    # road_disturbance = WebotsRoadDisturbance()
    # road_disturbance.disturbance_type = WebotsRoadDisturbance.TRIANGLE_DOUBLE_SIDED
    # road_disturbance.rotation = [0, 1, 0, -math.pi / 2.0]
    # road_disturbance.position = [40, 0, 0]
    # road_disturbance.width = 3.5
    # road_disturbance.length = 3
    # road_disturbance.height = 0.04
    # road_disturbance.inter_object_spacing = 0.5
    # sim_environment.road_disturbances_list.append(road_disturbance)

    # ----- Stop sign:
    # sim_obj = WebotsSimObject()
    # sim_obj.object_name = 'StopSign'
    # sim_obj.object_parameters.append(('translation', '40 0 6'))
    # sim_obj.object_parameters.append(('rotation', '0 1 0 1.5708'))
    # sim_environment.generic_sim_objects_list.append(sim_obj)

    # ----- Initial State Configurations:
    sim_environment.initial_state_config_list.append(
        InitialStateConfig(item=ItemDescription(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                                                item_index=0,
                                                item_state_index=WebotsVehicle.STATE_ID_VELOCITY_X),
                           value=ego_init_speed_m_s))

    # ----- Controller Parameters:
    # Ego Target Path:
    target_pos_list = [[450.0, 0.0],
                       [-400.0, 0.0]]
    # target_pos_list = [[450.0, 0.0], [445.0, 0.0], [440.0, 0.0], [435.0, 0.0], [430.0, 0.0], [425.0, 0.0], [420.0, 0.0], [415.0, 0.0], [410.0, 0.0], [405.0, 0.0], [400.0, 0.9], [395.0, 3.6], [390.0, 3.6], [385.0, 3.6], [380.0, 0.9], [375.0, 0.0], [370.0, 0.0], [365.0, 0.0], [360.0, 0.0], [355.0, 0.0], [350.0, 0.0], [345.0, 0.0], [340.0, 0.0], [335.0, 0.0], [330.0, 0.0], [325.0, 0.0], [320.0, 0.0], [315.0, 0.0], [310.0, 0.0], [305.0, 0.0], [300.0, 0.0], [295.0, 0.0], [290.0, 0.0], [285.0, 0.0], [280.0, 0.0], [275.0, 0.0], [270.0, 0.0], [265.0, 0.0], [260.0, 0.0], [255.0, 0.0], [250.0, 0.0], [245.0, 0.0], [240.0, 0.0], [235.0, 0.0], [230.0, 0.0], [225.0, 0.0], [220.0, 0.0], [215.0, 0.0], [210.0, 0.0], [205.0, 0.0], [200.0, 0.0], [195.0, 0.0], [190.0, 0.0], [185.0, 0.0], [180.0, 0.0], [175.0, 0.0], [170.0, 0.0], [165.0, 0.0], [160.0, 0.0], [155.0, 0.0], [150.0, 0.0], [145.0, 0.0], [140.0, 0.0], [135.0, 0.0], [130.0, 0.0], [125.0, 0.0], [120.0, 0.0], [115.0, 0.0], [110.0, 0.0], [105.0, 0.0], [100.0, 0.0], [95.0, 0.0], [90.0, 0.0], [85.0, 0.0], [80.0, 0.0], [75.0, 0.0], [70.0, 0.0], [65.0, 0.0], [60.0, 0.0], [55.0, 0.0], [50.0, 0.0], [45.0, 0.0], [40.0, 0.0], [35.0, 0.0], [30.0, 0.0], [25.0, 0.0], [20.0, 0.0], [15.0, 0.0], [10.0, 0.0], [5.0, 0.0], [-0.0, 0.0], [-4.95, -0.62], [-9.59, -2.45], [-13.63, -5.37], [-16.83, -9.19], [-18.98, -13.69], [-19.95, -18.59], [-19.68, -23.56], [-18.19, -28.32], [-15.56, -32.56], [-11.97, -36.02], [-7.63, -38.49], [-2.82, -39.8], [0.0, -40.0], [5.0, -40.0], [10.0, -40.0], [15.0, -40.0], [20.0, -40.0], [25.0, -40.0], [30.0, -40.0], [35.0, -40.0], [40.0, -40.0], [45.0, -40.0], [50.0, -40.0], [55.0, -40.0], [60.0, -40.0], [65.0, -40.0], [70.0, -40.0], [75.0, -40.0], [80.0, -40.0], [85.0, -40.0], [90.0, -40.0], [95.0, -40.0], [100.0, -40.0], [105.0, -40.0], [110.0, -40.0], [115.0, -40.0], [120.0, -40.0], [125.0, -40.0], [130.0, -40.0], [135.0, -40.0], [140.0, -40.0], [145.0, -40.0], [150.0, -40.0], [155.0, -40.0], [160.0, -40.0], [165.0, -40.0], [170.0, -40.0], [175.0, -40.0], [180.0, -40.0], [185.0, -40.0], [190.0, -40.0], [195.0, -40.0], [200.0, -40.0], [205.0, -40.0], [210.0, -40.0], [215.0, -40.0], [220.0, -40.0], [225.0, -40.0], [230.0, -40.0], [235.0, -40.0], [240.0, -40.0], [245.0, -40.0], [250.0, -40.0], [255.0, -40.0], [260.0, -40.0], [265.0, -40.0], [270.0, -40.0], [275.0, -40.0], [280.0, -40.0], [285.0, -40.0], [290.0, -40.0], [295.0, -40.0], [300.0, -40.0], [305.0, -40.0], [310.0, -40.0], [315.0, -40.0], [320.0, -40.0], [325.0, -40.0], [330.0, -40.0], [335.0, -40.0], [340.0, -40.0], [345.0, -40.0], [350.0, -40.0], [355.0, -40.0], [360.0, -40.0], [365.0, -40.0], [370.0, -40.0], [375.0, -40.0], [380.0, -40.0], [385.0, -40.0], [390.0, -40.0], [395.0, -40.0], [400.0, -40.0], [405.0, -40.0], [410.0, -40.0], [415.0, -40.0], [420.0, -40.0], [425.0, -40.0], [430.0, -40.0], [435.0, -40.0], [440.0, -40.0], [445.0, -40.0], [450.0, -40.0], [455.0, -40.0], [460.0, -40.0], [465.0, -40.0], [470.0, -40.0], [475.0, -40.0], [480.0, -40.0], [485.0, -40.0], [490.0, -40.0], [495.0, -40.0], [500.0, -40.0], [504.95, -39.38], [509.59, -37.55], [513.63, -34.63], [516.83, -30.81], [518.98, -26.31], [519.95, -21.41], [519.68, -16.44], [518.19, -11.68], [515.56, -7.44], [511.97, -3.98], [507.63, -1.51], [502.82, -0.2], [500.0, 0.0], [495.0, 0.0], [490.0, 0.0], [485.0, 0.0], [480.0, 0.0], [475.0, 0.0], [470.0, 0.0], [465.0, 0.0], [460.0, 0.0], [455.0, 0.0], [450.0, 0.0], [445.0, 0.0], [440.0, 0.0], [435.0, 0.0], [430.0, 0.0], [425.0, 0.0], [420.0, 0.0], [415.0, 0.0], [410.0, 0.0], [405.0, 0.0], [400.0, 0.0], [395.0, 0.0], [390.0, 0.0], [385.0, 0.0], [380.0, 0.0], [375.0, 0.0], [370.0, 0.0], [365.0, 0.0], [360.0, 0.0], [355.0, 0.0], [350.0, 0.0], [345.0, 0.0], [340.0, 0.0], [335.0, 0.0], [330.0, 0.0], [325.0, 0.0], [320.0, 0.0], [315.0, 0.0], [310.0, 0.0], [305.0, 0.0], [300.0, 0.0], [295.0, 0.0], [290.0, 0.0], [285.0, 0.0], [280.0, 0.0], [275.0, 0.0], [270.0, 0.0], [265.0, 0.0], [260.0, 0.0], [255.0, 0.0], [250.0, 0.0], [245.0, 0.0], [240.0, 0.0], [235.0, 0.0], [230.0, 0.0], [225.0, 0.9], [220.0, 3.6], [215.0, 3.6], [210.0, 0.9], [205.0, 0.0], [200.0, 0.0], [195.0, 0.0], [190.0, 0.0], [185.0, 0.0], [180.0, 0.0], [175.0, 0.0], [170.0, 0.0]]
    
    '''
    target_pos_list = [[10.0, 0.0],
                       [0.0, 0.0],
                       [-20*math.sin(1.0/4), 20*math.cos(1.0/4)-20],
                       [-20*math.sin(2.0/4), 20*math.cos(2.0/4)-20],
                       [-20*math.sin(3.0/4), 20*math.cos(3.0/4)-20],
                       [-20*math.sin(4.0/4), 20*math.cos(4.0/4)-20],
                       [-20*math.sin(5.0/4), 20*math.cos(5.0/4)-20],
                       [-20*math.sin(6.0/4), 20*math.cos(6.0/4)-20],
                       [-20*math.sin(7.0/4), 20*math.cos(7.0/4)-20],
                       [-20*math.sin(8.0/4), 20*math.cos(8.0/4)-20],
                       [-20*math.sin(9.0/4), 20*math.cos(9.0/4)-20],
                       [-20*math.sin(10.0/4), 20*math.cos(10.0/4)-20],
                       [-20*math.sin(11.0/4), 20*math.cos(11.0/4)-20],
                       [-20*math.sin(12.0/4), 20*math.cos(12.0/4)-20],
                       [500.0,-40.0],
                       [500+20*math.sin(1.0/4), -20*math.cos(1.0/4)-20],
                       [500+20*math.sin(2.0/4), -20*math.cos(2.0/4)-20],
                       [500+20*math.sin(3.0/4), -20*math.cos(3.0/4)-20],
                       [500+20*math.sin(4.0/4), -20*math.cos(4.0/4)-20],
                       [500+20*math.sin(5.0/4), -20*math.cos(5.0/4)-20],
                       [500+20*math.sin(6.0/4), -20*math.cos(6.0/4)-20],
                       [500+20*math.sin(7.0/4), -20*math.cos(7.0/4)-20],
                       [500+20*math.sin(8.0/4), -20*math.cos(8.0/4)-20],
                       [500+20*math.sin(9.0/4), -20*math.cos(9.0/4)-20],
                       [500+20*math.sin(10.0/4), -20*math.cos(10.0/4)-20],
                       [500+20*math.sin(11.0/4), -20*math.cos(11.0/4)-20],
                       [500+20*math.sin(12.0/4), -20*math.cos(12.0/4)-20],
                       [500.0, 0.0],
                       [0.0,   0.0]]
    
    
    '''
    for target_pos in target_pos_list:
        sim_environment.controller_params_list.append(
            WebotsControllerParameter(vehicle_id=2,
                                      parameter_name='target_position',
                                      parameter_data=target_pos))

    '''
    target_pos_list = [[310.0, 3.0],
                       [1000.0, 3.0]]
    # target_pos_list = [[450.0, 0.0], [445.0, 0.0], [440.0, 0.0], [435.0, 0.0], [430.0, 0.0], [425.0, 0.0], [420.0, 0.0], [415.0, 0.0], [410.0, 0.0], [405.0, 0.0], [400.0, 0.9], [395.0, 3.6], [390.0, 3.6], [385.0, 3.6], [380.0, 0.9], [375.0, 0.0], [370.0, 0.0], [365.0, 0.0], [360.0, 0.0], [355.0, 0.0], [350.0, 0.0], [345.0, 0.0], [340.0, 0.0], [335.0, 0.0], [330.0, 0.0], [325.0, 0.0], [320.0, 0.0], [315.0, 0.0], [310.0, 0.0], [305.0, 0.0], [300.0, 0.0], [295.0, 0.0], [290.0, 0.0], [285.0, 0.0], [280.0, 0.0], [275.0, 0.0], [270.0, 0.0], [265.0, 0.0], [260.0, 0.0], [255.0, 0.0], [250.0, 0.0], [245.0, 0.0], [240.0, 0.0], [235.0, 0.0], [230.0, 0.0], [225.0, 0.0], [220.0, 0.0], [215.0, 0.0], [210.0, 0.0], [205.0, 0.0], [200.0, 0.0], [195.0, 0.0], [190.0, 0.0], [185.0, 0.0], [180.0, 0.0], [175.0, 0.0], [170.0, 0.0], [165.0, 0.0], [160.0, 0.0], [155.0, 0.0], [150.0, 0.0], [145.0, 0.0], [140.0, 0.0], [135.0, 0.0], [130.0, 0.0], [125.0, 0.0], [120.0, 0.0], [115.0, 0.0], [110.0, 0.0], [105.0, 0.0], [100.0, 0.0], [95.0, 0.0], [90.0, 0.0], [85.0, 0.0], [80.0, 0.0], [75.0, 0.0], [70.0, 0.0], [65.0, 0.0], [60.0, 0.0], [55.0, 0.0], [50.0, 0.0], [45.0, 0.0], [40.0, 0.0], [35.0, 0.0], [30.0, 0.0], [25.0, 0.0], [20.0, 0.0], [15.0, 0.0], [10.0, 0.0], [5.0, 0.0], [-0.0, 0.0], [-4.95, -0.62], [-9.59, -2.45], [-13.63, -5.37], [-16.83, -9.19], [-18.98, -13.69], [-19.95, -18.59], [-19.68, -23.56], [-18.19, -28.32], [-15.56, -32.56], [-11.97, -36.02], [-7.63, -38.49], [-2.82, -39.8], [0.0, -40.0], [5.0, -40.0], [10.0, -40.0], [15.0, -40.0], [20.0, -40.0], [25.0, -40.0], [30.0, -40.0], [35.0, -40.0], [40.0, -40.0], [45.0, -40.0], [50.0, -40.0], [55.0, -40.0], [60.0, -40.0], [65.0, -40.0], [70.0, -40.0], [75.0, -40.0], [80.0, -40.0], [85.0, -40.0], [90.0, -40.0], [95.0, -40.0], [100.0, -40.0], [105.0, -40.0], [110.0, -40.0], [115.0, -40.0], [120.0, -40.0], [125.0, -40.0], [130.0, -40.0], [135.0, -40.0], [140.0, -40.0], [145.0, -40.0], [150.0, -40.0], [155.0, -40.0], [160.0, -40.0], [165.0, -40.0], [170.0, -40.0], [175.0, -40.0], [180.0, -40.0], [185.0, -40.0], [190.0, -40.0], [195.0, -40.0], [200.0, -40.0], [205.0, -40.0], [210.0, -40.0], [215.0, -40.0], [220.0, -40.0], [225.0, -40.0], [230.0, -40.0], [235.0, -40.0], [240.0, -40.0], [245.0, -40.0], [250.0, -40.0], [255.0, -40.0], [260.0, -40.0], [265.0, -40.0], [270.0, -40.0], [275.0, -40.0], [280.0, -40.0], [285.0, -40.0], [290.0, -40.0], [295.0, -40.0], [300.0, -40.0], [305.0, -40.0], [310.0, -40.0], [315.0, -40.0], [320.0, -40.0], [325.0, -40.0], [330.0, -40.0], [335.0, -40.0], [340.0, -40.0], [345.0, -40.0], [350.0, -40.0], [355.0, -40.0], [360.0, -40.0], [365.0, -40.0], [370.0, -40.0], [375.0, -40.0], [380.0, -40.0], [385.0, -40.0], [390.0, -40.0], [395.0, -40.0], [400.0, -40.0], [405.0, -40.0], [410.0, -40.0], [415.0, -40.0], [420.0, -40.0], [425.0, -40.0], [430.0, -40.0], [435.0, -40.0], [440.0, -40.0], [445.0, -40.0], [450.0, -40.0], [455.0, -40.0], [460.0, -40.0], [465.0, -40.0], [470.0, -40.0], [475.0, -40.0], [480.0, -40.0], [485.0, -40.0], [490.0, -40.0], [495.0, -40.0], [500.0, -40.0], [504.95, -39.38], [509.59, -37.55], [513.63, -34.63], [516.83, -30.81], [518.98, -26.31], [519.95, -21.41], [519.68, -16.44], [518.19, -11.68], [515.56, -7.44], [511.97, -3.98], [507.63, -1.51], [502.82, -0.2], [500.0, 0.0], [495.0, 0.0], [490.0, 0.0], [485.0, 0.0], [480.0, 0.0], [475.0, 0.0], [470.0, 0.0], [465.0, 0.0], [460.0, 0.0], [455.0, 0.0], [450.0, 0.0], [445.0, 0.0], [440.0, 0.0], [435.0, 0.0], [430.0, 0.0], [425.0, 0.0], [420.0, 0.0], [415.0, 0.0], [410.0, 0.0], [405.0, 0.0], [400.0, 0.0], [395.0, 0.0], [390.0, 0.0], [385.0, 0.0], [380.0, 0.0], [375.0, 0.0], [370.0, 0.0], [365.0, 0.0], [360.0, 0.0], [355.0, 0.0], [350.0, 0.0], [345.0, 0.0], [340.0, 0.0], [335.0, 0.0], [330.0, 0.0], [325.0, 0.0], [320.0, 0.0], [315.0, 0.0], [310.0, 0.0], [305.0, 0.0], [300.0, 0.0], [295.0, 0.0], [290.0, 0.0], [285.0, 0.0], [280.0, 0.0], [275.0, 0.0], [270.0, 0.0], [265.0, 0.0], [260.0, 0.0], [255.0, 0.0], [250.0, 0.0], [245.0, 0.0], [240.0, 0.0], [235.0, 0.0], [230.0, 0.0], [225.0, 0.9], [220.0, 3.6], [215.0, 3.6], [210.0, 0.9], [205.0, 0.0], [200.0, 0.0], [195.0, 0.0], [190.0, 0.0], [185.0, 0.0], [180.0, 0.0], [175.0, 0.0], [170.0, 0.0]]


    
    for target_pos in target_pos_list:
        sim_environment.controller_params_list.append(
            WebotsControllerParameter(vehicle_id=3,
                                      parameter_name='target_position',
                                      parameter_data=target_pos))

    '''
    # target_pos_list = computeTargetPath()
    # target_pos_list = [[80.0, 0.0], [50.0, 0.0], [45.0, 0.1], [40.0, 0.1], [35.0, 0.1], [30.0, 0.1], [25.0, 0.1], [20.0, 0.1], [15.0, 0.1], [10.0, 0.1], [5.0, 0.1], [-0.0, -0.4], [-4.85, -1.01], [-9.4, -2.8], [-13.7, -5.29], [-16.91, -9.14], [-19.07, -13.66], [-19.55, -18.61], [-18.79, -23.4], [-17.82, -28.16], [-15.64, -32.63], [-11.73, -35.7], [-7.29, -37.65], [-2.77, -39.4], [0.0, -40.4], [5.0, -40.4], [10.0, -40.9], [15.0, -41.4], [20.0, -41.9], [25.0, -42.4], [30.0, -42.4], [35.0, -42.4], [40.0, -42.4], [45.0, -42.4], [50.0, -42.4], [55.0, -41.9], [60.0, -41.4], [65.0, -40.9], [70.0, -40.4], [75.0, -39.9], [80.0, -39.9], [85.0, -39.9], [90.0, -39.9], [95.0, -39.9], [100.0, -39.9], [105.0, -39.9], [110.0, -39.9], [115.0, -39.9], [120.0, -39.9], [125.0, -39.9], [130.0, -39.9], [135.0, -39.9]]

    # target_pos_list = [[500.0, 0.0], [495.0, 0.0], [490.0, 0.0], [485.0, 0.0], [480.0, 0.0], [475.0, 0.0], [470.0, 0.0], [465.0, 0.0], [460.0, 0.0], [455.0, 0.0], [450.0, 0.4], [445.0, 0.4], [440.0, 0.4], [435.0, 0.0], [430.0, 0.0], [425.0, 0.0], [420.0, 0.0], [415.0, 0.0], [410.0, 0.0], [405.0, 0.0], [400.0, 0.0], [395.0, 0.0], [390.0, 0.0], [385.0, 0.4], [380.0, 0.8], [375.0, 1.2], [370.0, 1.6], [365.0, 2.0], [360.0, 2.4], [355.0, 2.4], [350.0, 2.4], [345.0, 2.4], [340.0, 2.4], [335.0, 2.4], [330.0, 2.0], [325.0, 1.6], [320.0, 1.2], [315.0, 0.8], [310.0, 0.4], [305.0, 0.0], [300.0, 0.0], [295.0, 0.0], [290.0, 0.0], [285.0, 0.0], [280.0, 0.0], [275.0, 0.0], [270.0, 0.0], [265.0, 0.0], [260.0, 0.0], [255.0, 0.0], [250.0, 0.0], [245.0, 0.0], [240.0, 0.0], [235.0, 0.0], [230.0, 0.0], [225.0, 0.0], [220.0, 0.0], [215.0, 0.0], [210.0, 0.0], [205.0, 0.0], [200.0, 0.0], [195.0, 0.0], [190.0, 0.0], [185.0, 0.0], [180.0, 0.0], [175.0, 0.0], [170.0, 0.0], [165.0, 0.0], [160.0, 0.0], [155.0, 0.0], [150.0, 0.0], [145.0, 0.0], [140.0, 0.0], [135.0, 0.0], [130.0, 0.0], [125.0, 0.0], [120.0, 0.0], [115.0, 0.0], [110.0, 0.0], [105.0, 0.0], [100.0, 0.0]]
    target_pos_list = [[500.0, 0.0], [495.0, 0.0], [490.0, 0.0], [485.0, 0.0], [480.0, 0.0], [475.0, 0.0], [470.0, 0.0], [465.0, 0.0], [460.0, 0.0], [455.0, 0.0], [450.0, 0.0], [445.0, 0.4], [440.0, 0.8], [435.0, 1.2], [430.0, 1.6], [425.0, 2.0], [420.0, 2.4], [415.0, 2.8], [410.0, 2.8], [405.0, 2.8], [400.0, 2.8], [395.0, 2.8], [390.0, 2.4], [385.0, 2.0], [380.0, 1.6], [375.0, 1.2], [370.0, 0.8], [365.0, 0.4], [360.0, 0.0], [355.0, -0.4], [350.0, -0.8], [345.25, -1.4], [340.6, -2.8], [336.37, -5.37], [333.17, -9.19], [331.02, -13.69], [330.45, -18.61], [331.11, -23.42], [332.18, -28.16], [334.44, -32.56], [338.03, -36.02], [342.52, -38.12], [347.23, -39.4], [350.0, -40.0], [355.0, -40.0], [360.0, -40.0], [365.0, -40.0], [370.0, -40.0], [375.0, -40.0], [380.0, -40.0], [385.0, -40.0], [390.0, -40.0], [395.0, -40.0], [400.0, -40.0], [405.0, -40.0], [410.0, -40.0], [415.0, -40.0], [420.0, -40.0], [425.0, -40.0], [430.0, -40.0], [435.0, -40.0]]



















    for target_pos in target_pos_list:
        sim_environment.controller_params_list.append(
            WebotsControllerParameter(vehicle_id=1,
                                      parameter_name='target_position',
                                      parameter_data=target_pos))

    # ----- Heart Beat Configuration:
    sim_environment.heart_beat_config = HeartBeatConfig(sync_type=HeartBeatConfig.WITHOUT_SYNC,
                                                        period_ms=2000)

    # ----- View Follow configuration:
    sim_environment.view_follow_config = \
        ViewFollowConfig(item_type=ItemDescription.ITEM_TYPE_VEHICLE,
                         item_index=0,
                         # position=[sim_environment.agent_vehicles_list[0].current_position[0] + 15.0,
                         #           sim_environment.agent_vehicles_list[0].current_position[1] + 2.0,
                         #           sim_environment.agent_vehicles_list[0].current_position[2]],
                         position=[515.0, 2.35, 0.0],
                         rotation=[0.0, -1.0, 0.0, -math.pi/2.0])

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
    sim_config.world_file = '../Webots_Projects/worlds/our_world.wbt'

    sim_env_configurator = SimEnvironmentConfigurator(sim_config=sim_config)
    (is_connected, simulator_instance) = sim_env_configurator.connect(max_connection_retry=3)
    if not is_connected:
        raise ValueError('Could not connect!')
    sim_env_configurator.setup_sim_environment(sim_environment)
    trajectory = sim_env_configurator.run_simulation_get_trace()
    if for_matlab:
        trajectory = experiment_tools.npArray2Matlab(trajectory)
    time.sleep(1)  # Wait for Webots to reload the world.
    # print(trajectory)
    
    # with open("out.csv","w") as f:
    #     wr = csv.writer(f)
    #     wr.writerows(trajectory)
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

run_test()
# run_covering_array_tests()
