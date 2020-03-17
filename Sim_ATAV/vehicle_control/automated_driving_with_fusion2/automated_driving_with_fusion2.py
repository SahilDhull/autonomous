"""Defines SimpleSensorFusionControl class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import math
import numpy as np
import threading


from Sim_ATAV.common.controller_communication_interface import ControllerCommunicationInterface
from Sim_ATAV.vehicle_control.base_controller.base_controller import BaseCarController
from Sim_ATAV.vehicle_control.controller_commons import controller_commons
from Sim_ATAV.vehicle_control.controller_commons.path_following_tools import PathFollowingTools
from Sim_ATAV.vehicle_control.controller_commons.perception.sensor_fusion.sensor_fusion_tracker \
    import SensorFusionTracker
from Sim_ATAV.vehicle_control.controller_commons.planning.target_speed_planner import TargetSpeedPlanner,\
    TargetSpeedData
from Sim_ATAV.vehicle_control.generic_stanley_controller.generic_stanley_controller \
    import GenericStanleyController
from Sim_ATAV.vehicle_control.generic_pid_controller.generic_pid_controller import GenericPIDController

from Sim_ATAV.vehicle_control.controller_commons.visualization.camera_info_display import CameraInfoDisplay

WORLD_TIME_STEP_MS = 10
HAS_DEBUG_DISPLAY = True
SENSOR_TYPE = 'Actual'  # 'Actual', 'Perfect'
DEBUG_MODE = False



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import csv
import pickle
import random
# import dill




exp_out = [[]]
time_index = 0
#change heading when changing left to right/straight
folder_cnt = 4
img_cnt = 1850
data_dict = {}
inf = 1e9
save = True
# save = False

# file_path = '../../../correction/Scenario'+ str(folder_cnt) + '/'
file_path = '../../../images/correction/left/'
image_path = file_path
pkl_file = file_path + 'control_throttle.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


class NetworkLight(nn.Module):
    def __init__(self):
        super(NetworkLight, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 5, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=48*18*36 + 1, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=2)
        )
        
    def forward(self, input, vel):
        input = input.view(input.size(0), 3, 310, 600)
        output = self.conv_layers(input)

        # Append velocity in the output vector
        output = output.view(output.size(0), -1)
        vel = vel.view(vel.size(0),-1)
        # print(vel.shape)
        # print(output.shape)
        output = torch.cat((output,vel),dim = 1)
        output = self.linear_layers(output)
        return output

class Dataset2(data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        batch_samples = self.samples[index]
        img_name = image_path + batch_samples[0]
        center_img = read(img_name)
        center_img = self.transform(center_img)
        return (center_img, batch_samples[1])
      
    def __len__(self):
        return len(self.samples)

def read(name):
    current_image = cv2.imread(name)
    current_image = current_image[65:-25, :, :]
    return current_image

def toDevice(datas, device):
    imgs, vel = datas
    return imgs.float().to(device), vel.float().to(device)

def testing(model, test_generator):
    model.eval()
    with torch.set_grad_enabled(False):
        for local_batch, data in enumerate(test_generator):
            data = toDevice(data, device)
            # print(data)
            imgs, vel = data
            with torch.no_grad():
                outputs = model(imgs,vel)
    return outputs

def MLmodel(sample_test):
    


    eval_model = NetworkLight()
    eval_state = torch.load(file_path + 'model.h5')
    eval_model = eval_state['model']
    eval_model.float()
    eval_model.eval()



    #--------Remove this later on----------------------------------------
    # with open(pkl_file, 'rb') as handle:
    #     samples = pickle.load(handle)

    # samples_list = [ [k, v[0], v[1], v[2]] for k, v in samples.items() ]
    #--------------------------------------------------



    transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0) - 0.5)])

    params = {'batch_size': 1,
              'shuffle': True}


      
    # print(samples_list[0])
    test_set = Dataset2([sample_test], transformations)
    test_generator = DataLoader(test_set, **params)



    # print('device is: ', device)


    # print(samples_list[0])

    # dat = test_set.__getitem__(0)
    # img = toDevice(dat , device)
    # print(img)
    # print(vel)

    # print(len(test_generator))



    # for local_batch, data in enumerate(test_generator):
    # data = toDevice(data, device)
    # print(data)       
            
    Result = testing(eval_model, test_generator)
    # print(Result)
    return float(Result[0][0]),float(Result[0][1])

# Our global variables
target_throttle = [0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 0.65, 0.7, 0.7, 0.7, 0.75, 0.6, 0.6, 0.6, 0.35, 0.35, -0.3, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.2, -0.2, -0.2, -0.2, -0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, -0.1, 0.15, 0.3, 0.55, 0.65, 0.75, 0.85, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 0.65, 0.65, 0.7, 0.7, 0.7, 0.75, 0.6, 0.6, 0.6, 0.35, 0.35, -0.3, -0.3, -0.3, -0.4, -0.4, -0.4, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1, 0.05, 0.05, 0.05, 0.05, 0.05, -0.1]







target_t = [1.58, 2.23, 2.73, 3.15, 3.52, 3.86, 4.17, 4.47, 4.75, 5.02, 5.28, 5.53, 5.77, 6.01, 6.24, 6.47, 6.7, 6.92, 7.14, 7.36, 7.58, 7.8, 8.02, 8.26, 8.51, 8.78, 9.07, 9.39, 9.74, 10.14, 10.58, 11.08, 11.68, 12.39, 13.14, 13.89, 14.64, 15.39, 16.14, 16.89, 17.68, 18.47, 19.15, 19.7, 20.15, 20.54, 20.89, 21.21, 21.51, 21.8, 22.08, 22.34, 22.59, 22.84, 23.08, 23.32, 23.55, 23.78, 24.01, 24.23, 24.45, 24.67, 24.89, 25.11, 25.33, 25.57, 25.82, 26.09, 26.38, 26.7, 27.05, 27.42, 27.83, 28.29, 28.82, 29.47, 30.26, 31.11, 31.96, 32.81, 33.66, 34.51, 35.43]









def debug_print(print_str):
    if DEBUG_MODE:
        print(print_str)
        sys.stdout.flush()


class PathAndSpeedFollower(BaseCarController):
    """PathAndSpeedFollower class is a car controller class for Webots."""

    CAMERA_TO_FRONT_DISTANCE = 2.3  # 2.3 m is the distance from Prius top sensor location to the very front of the car
    LIDAR_TO_FRONT_DISTANCE = 2.3
    CAMERA_MAIN_RELATIVE_POSITION = [0.0, 1.3]
    LIDAR_MAIN_RELATIVE_POSITION = [0.0, 1.3]
    RADAR_FRONT_RELATIVE_POSITION = [0.0, 3.6]
    FRONT_TO_REAR_WHEELS_DISTANCE = 3.6  # Approximate (this is intentially longer than the actual wheel base
    # for smoother operation)

    CAMERA_LOCAL_COORDINATES = [0.0, 1.3, 1.1]
    CAMERA_X_ROT_ANGLE = -0.01
    CAMERA_LOCAL_ROTATION = np.array([[1.0, 0.0, 0.0],
                                      [0.0, math.cos(CAMERA_X_ROT_ANGLE), -math.sin(CAMERA_X_ROT_ANGLE)],
                                      [0.0, math.sin(CAMERA_X_ROT_ANGLE), math.cos(CAMERA_X_ROT_ANGLE)]])

    CAR_FRONT_TRIANGLE_LINE1_M = -192/126  # old value: -0.6  # Line 1 m for front triangle.
    CAR_FRONT_TRIANGLE_LINE1_B = 1142.9  # Old value: 526  # Line 1 b for front triangle.
    CAR_FRONT_TRIANGLE_LINE2_M = 192/126  # old value: 0.6  # Line 2 m for front triangle.
    CAR_FRONT_TRIANGLE_LINE2_B = -758.9  # Old value: -202  # Line 2 b for front triangle.

    PED_FRONT_TRIANGLE_LINE1_M = -192/204  # old value: -0.6  # Line 1 m for front triangle.
    PED_FRONT_TRIANGLE_LINE1_B = 779.3  # Old value: 526  # Line 1 b for front triangle.
    PED_FRONT_TRIANGLE_LINE2_M = 192/204  # old value: 0.6  # Line 2 m for front triangle.
    PED_FRONT_TRIANGLE_LINE2_B = -395.3  # Old value: -202  # Line 2 b for front triangle.

    CLASSIFIER_PERIOD_MS = 100
    LIDAR_PERIOD_MS = 200
    RADAR_PERIOD_MS = 200
    MIN_EMERGENCY_BRAKE_DURATION_MS = 100.0
    MEASURE_EXEC_TIME = False
    LANE_WIDTH = 3.5
    MIN_STEERING_MANEUVER_MS = 2000.0
    EMERGENCY_STEERING_TTC = 1.0

    OBJECT_TRACKER_MAX_DISTANCE = 70.0

    def __init__(self, controller_parameters):
        (car_model, target_speed_m_s, is_direct_speed_control, target_lat_pos, self_vhc_id, slow_at_intersection,
         use_fusion) = controller_parameters
        BaseCarController.__init__(self, car_model)
        self.slow_at_intersection = slow_at_intersection in ('True', 'true', 'yes', 'Yes')
        self.is_direct_speed_control = is_direct_speed_control in ('True', 'true', 'yes', 'Yes')
        self.use_fusion = use_fusion in ('True', 'true', 'yes', 'Yes')
        self.camera_device_name = 'camera'
        self.camera = None
        self.compass_device_name = 'compass'
        self.compass = None
        self.display_device_name = 'display'
        self.display = None
        self.camera_info_display = None
        self.sensor_display_device_name = 'sensor_display'
        self.sensor_display = None
        self.sensor_info_display = None
        self.gps_device_name = 'gps'
        self.gps = None
        self.receiver_device_name = 'receiver'
        self.receiver = None
        self.emitter_device_name = 'emitter'
        self.emitter = None
        self.lidar_main_device_name = 'velodyne'  # ibeo',  'velodyne'
        self.lidar_main = None
        self.radar_front_device_name = 'radar'
        self.radar_front = None
        self.target_speed_m_s = float(target_speed_m_s)
        self.classifier = None
        self.classification_client = None
        self.obj_tracker = None
        self.ground_truth_generator = None
        self.contr_comm = ControllerCommunicationInterface()
        self.target_lat_pos = float(target_lat_pos)
        self.target_bearing = 0.0
        self.lateral_controller = GenericStanleyController()
        self.lateral_controller.k = 0.5
        self.lateral_controller.k2 = 0.4
        self.lateral_controller.k3 = 1.1
        self.lateral_controller.set_output_range(-0.8, 0.8)
        self.longitudinal_controller = GenericPIDController(0.15, 0.01, 0.0)
        self.longitudinal_controller.set_integrator_value_range(-20.0, 20.0)
        self.self_vhc_id = int(self_vhc_id)
        self.path_following_tools = PathFollowingTools()
        self.self_sensor_fusion_tracker = None
        self.last_segment_ind = 0
        self.self_current_state = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_segment_ind = 0
        self.detour_start_time = None
        self.target_speed_planner = TargetSpeedPlanner(default_speed=self.target_speed_m_s)
        print('AutomatedDrivingControl Initialized: {}, {}'.format(car_model, self.target_speed_m_s))

    def start_devices(self):
        """Start the devices on the car and initialize objects like classifier."""
        # Start camera and the sensors:
        self.camera = self.getCamera(self.camera_device_name)
        if self.camera is not None:
            self.camera.enable(self.CLASSIFIER_PERIOD_MS)
        self.camera_info_display = CameraInfoDisplay(self.display)
          
        self.gps = self.getGPS(self.gps_device_name)
        if self.gps is not None:
            self.gps.enable(WORLD_TIME_STEP_MS)
        self.compass = self.getCompass(self.compass_device_name)
        if self.compass is not None:
            self.compass.enable(WORLD_TIME_STEP_MS)
        self.receiver = self.getReceiver(self.receiver_device_name)
        if self.receiver is not None:
            self.receiver.enable(WORLD_TIME_STEP_MS)
        self.emitter = self.getEmitter(self.emitter_device_name)
        # Start the car engine
        self.start_car()

    def run(self):
        """Runs the controller."""
        self.start_devices()
        print("Devices Started.")
        sys.stdout.flush()

        def get_self_position():
            """Returns current self position."""
            return self.self_current_state[0:2]

        def get_self_speed_ms():
            """Returns current speed in m/s."""
            return self.self_current_state[2]

        def get_self_yaw_angle():
            """Returns self yaw angle in radians."""
            return self.self_current_state[3]

        # Internal functions to keep the code more readable:
        def read_gps_sensor(gps_device):
            """Reads GPS sensor."""
            if gps_device is not None:
                sensor_gps_speed_m_s = gps_device.getSpeed()
                sensor_gps_position_m = gps_device.getValues()
            else:
                sensor_gps_speed_m_s = 0.0
                sensor_gps_position_m = [0.0, 0.0, 0.0]
            return sensor_gps_position_m, sensor_gps_speed_m_s

        def read_compass_sensor(compass_device):
            """Reads Compass Sensor."""
            if compass_device is not None:
                sensor_compass_bearing_rad = controller_commons.get_bearing(compass_device)
            else:
                sensor_compass_bearing_rad = 0.0
            return sensor_compass_bearing_rad

        def compute_and_apply_control():
            """Computes control output using the detected objects from sensor suite."""
            cur_position = get_self_position()
            cur_speed_ms = get_self_speed_ms()
            cur_yaw_angle = get_self_yaw_angle()

            # Compute control
            if self.path_following_tools.target_path is not None:
                # Compute distance from front wheels for smoother turns:
                temp_cur_pos = [cur_position[0] - (self.FRONT_TO_REAR_WHEELS_DISTANCE * math.sin(cur_yaw_angle) +
                                                   cur_speed_ms * 0.2 * math.sin(cur_yaw_angle)),
                                cur_position[1] + (self.FRONT_TO_REAR_WHEELS_DISTANCE * math.cos(cur_yaw_angle) +
                                                   cur_speed_ms * 0.2 * math.cos(cur_yaw_angle))]
                (current_segment_ind, line_segment_as_list, nearest_pos_on_path, dist_to_seg_end) = \
                    self.path_following_tools.get_current_segment(temp_cur_pos, self.last_segment_ind)
                (distance_err, angle_err) = \
                    self.path_following_tools.get_distance_and_angle_error(temp_cur_pos,
                                                                           cur_yaw_angle,
                                                                           last_segment_ind=self.last_segment_ind,
                                                                           is_detouring=False)
                self.last_segment_ind = current_segment_ind
                if len(self.path_following_tools.path_details) > current_segment_ind:
                    (next_turn_angle, travel_distance) = self.path_following_tools.path_details[current_segment_ind]
                    travel_distance += dist_to_seg_end
                else:
                    (next_turn_angle, travel_distance) = (0.0, 0.0)
            else:
                current_segment_ind = -1
                angle_err = self.target_bearing - cur_yaw_angle
                while angle_err > math.pi:
                    angle_err -= 2*math.pi
                while angle_err < -math.pi:
                    angle_err += 2*math.pi
                distance_err = -(self.target_lat_pos - cur_position[0])
                (next_turn_angle, travel_distance) = (0.0, 0.0)

            current_target_speed = \
                self.target_speed_planner.get_current_target_speed(cur_time_ms=cur_time_ms,
                                                                   cur_segment_ind=current_segment_ind)
            if self.slow_at_intersection and abs(next_turn_angle) > math.pi/60 and travel_distance < 100.0:
                turn_ratio = min(1.0, abs(next_turn_angle)/(math.pi/4.0))
                max_speed_limit = 10.0 + ((1.0 - turn_ratio)*30.0)
                # decrease speed limit as we approach to the intersection.
                max_speed_limit = (max_speed_limit + (current_target_speed - max_speed_limit) *
                                   ((max(travel_distance, 10.0)-10.0)/80.0))
            else:
                max_speed_limit = current_target_speed

            control_steering = self.lateral_controller.compute(angle_err,
                                                               distance_err,
                                                               cur_speed_ms)
            speed_ratio = min(1.0, self.self_current_state[2]/22.0)
            max_steering = 0.1 + (1.0 - speed_ratio)*0.7
            control_steering = min(max(-max_steering, control_steering), max_steering)

            if self.is_direct_speed_control:

                # self.set_target_speed_and_angle(speed=controller_commons.speed_ms_to_kmh(10.0), angle=control_steering)
                
                '''
                v = 0.1
                t = 0.3
                global t1, v1, flag

                if cur_time_ms==100:
                    self.set_target_speed_and_angle(speed=controller_commons.speed_ms_to_kmh(v), angle=control_steering)
                elif cur_time_ms>=5000:
                    self.set_throttle(t)
                # if cur_time_ms%200==0:
                #     print("time: "+str(cur_time_ms)+" vel: "+str(cur_speed_ms))
                if abs(round(cur_speed_ms,0)-cur_speed_ms)<0.01:
                    t1 = cur_time_ms
                    v1 = cur_speed_ms
                    # print ("--> "+str(t1))
                if cur_time_ms-t1 in (100,200,300,400,500,600,700,800,900,1000):
                    a = ((cur_speed_ms-v1)/(cur_time_ms-t1))*1000
                    # print("time: "+str(cur_time_ms)+" diff: "+str(cur_time_ms-t1)+" speed: "+str(round(v1,2)) + " acc: "+str(round(a,2)))
                '''

                # if cur_time_ms-t1 == 1000:
                #     a = ((cur_speed_ms-v1)/(cur_time_ms-t1))*1000
                #     print("time: "+str(cur_time_ms)+" diff: "+str(cur_time_ms-t1)+" speed: "+str(round(v1,2)) + " acc: "+str(round(a,2)))


                if cur_time_ms<3010:
                    x = 0.0
                    self.set_target_speed_and_angle(speed= controller_commons.speed_ms_to_kmh(x) ,angle=control_steering)
                else:
                    # global time_index
                    # if(target_t[time_index] < ((cur_time_ms/1000.0) -4) ):
                    #     time_index = time_index + 1
                    # x2 = exp_out[time_index][0]
                    # y2 = exp_out[time_index][1]
                    # inc = 0.0
                    # if(time_index>0):
                    #     t1 = exp_out[time_index-1][4]
                    #     dt = cur_time_ms/1000.0 - 3 - t1
                    #     x1 = exp_out[time_index-1][0]
                    #     u1 = exp_out[time_index-1][3]
                    #     a2 = exp_out[time_index][2]
                    #     dx = u1*dt + 0.5*a2*dt*dt
                    #     if(abs(x2-x1)==5.0):
                    #         if( (dx-0.5)/abs(x2-x1)>(cur_position[1]-x1)/(x2-x1) ):
                    #             inc = 0.05
                    #         elif( (dx+0.5)/abs(x2-x1)<(cur_position[1]-x1)/(x2-x1) ):
                    #             inc = -0.05
                    #         else:
                    #             inc = 0.0
                    x = 3.0
                    self.set_target_speed_and_angle(speed= controller_commons.speed_ms_to_kmh(x) ,angle=control_steering)
                
                    # if(target_throttle[time_index])
                    heading = 0
                    # if cur_position[0]>=290.0 and cur_position[0]<=340.0 and cur_position[1]<=20:
                    #     heading = 1
                    if cur_time_ms%100==0:
                        global img_cnt
                        img_name = "img_"+str(img_cnt)+".png"
                        if save:
                            self.camera.saveImage(image_path+img_name,1)
                        img_cnt = img_cnt + 1
                        data_dict[img_name] = [cur_speed_ms,target_throttle[time_index],control_steering, heading]
                        # print(heading)
                        # throttle, angle = MLmodel([img_name,cur_speed_ms])
                        # print("throttle: "+str(throttle)+" angle: "+str(angle))
                        # data_dict[img_name] = [cur_speed_ms,target_throttle[time_index],control_steering]
                        # self.set_throttle_and_steering_angle(throttle, angle)
                # self.set_target_speed_and_angle(speed=controller_commons.speed_ms_to_kmh(min(max_speed_limit,
                #                                                                              current_target_speed)),
                #                                 angle=control_steering)
                if cur_time_ms%500==0:
                    print("Time: "+str(cur_time_ms)+" Agent vehicle speed: "+str(cur_speed_ms) + " pos: "+str(cur_position))
            else:
                control_throttle = self.longitudinal_controller.compute(min(max_speed_limit, current_target_speed)
                                                                        - cur_speed_ms)
                self.set_throttle_and_steering_angle(control_throttle, control_steering)
            if current_target_speed < 0.0:
                # Emergency / sudden braking
                self.set_brake(1.0)
                self.set_throttle(0.0)

        while self.step() >= 0:
            sim_time = self.get_sim_time()
            cur_time_ms = int(round(1000 * sim_time))
            # -------------- Read Sensors----------------
            # ************ Read GPS ************
            (sensor_gps_position_m, sensor_gps_speed_m_s) = read_gps_sensor(self.gps)
            # ************ Read Compass ************
            sensor_compass_bearing_rad = read_compass_sensor(self.compass)

            # -------------- Sensor Fusion ----------------
            # ************ Sensor Fusion for own states (GPS + Compass) ************
            if self.self_sensor_fusion_tracker is None:
                self.self_current_state = [sensor_gps_position_m[0], sensor_gps_position_m[2], sensor_gps_speed_m_s,
                                           sensor_compass_bearing_rad, 0.0]
                if sensor_gps_speed_m_s > 50.0 or sensor_gps_speed_m_s < -20.0:  # Filter out errors in read gps speed
                    sensor_gps_speed_m_s = 0.0
                    self.self_current_state[2] = sensor_gps_speed_m_s
                if self.use_fusion:
                    # Initiate self sensor fusion tracker
                    self.self_sensor_fusion_tracker = SensorFusionTracker(initial_state_mean=self.self_current_state,
                                                                          filter_type='ukf')
            else:
                if self.gps is not None and self.compass is not None:
                    measurement = [sensor_gps_position_m[0], sensor_gps_position_m[2], sensor_gps_speed_m_s,
                                   sensor_compass_bearing_rad]
                    (self.self_current_state, state_cov) = self.self_sensor_fusion_tracker.get_estimates(
                        measurements=measurement, sensor_type=SensorFusionTracker.SENSOR_TYPE_GPS_COMPASS)
                elif self.gps is not None:
                    measurement = [sensor_gps_position_m[0], sensor_gps_position_m[2], sensor_gps_speed_m_s]
                    (self.self_current_state, state_cov) = self.self_sensor_fusion_tracker.get_estimates(
                        measurements=measurement, sensor_type=SensorFusionTracker.SENSOR_TYPE_GPS)
                elif self.compass is not None:
                    measurement = [sensor_compass_bearing_rad]
                    (self.self_current_state, state_cov) = self.self_sensor_fusion_tracker.get_estimates(
                        measurements=measurement, sensor_type=SensorFusionTracker.SENSOR_TYPE_COMPASS)
                else:
                    self.self_current_state = [0.0, 0.0, 0.0, 0.0, 0.0]

            # Read sensor-like information from Simulation Supervisor
            
            if self.receiver is not None:
                messages = self.contr_comm.receive_all_communication(self.receiver)
                command_list = self.contr_comm.extract_all_commands_from_message(messages)
                path_modified = False
                for command_item in command_list:
                    command = command_item[0]
                    if command == ControllerCommunicationInterface.SET_CONTROLLER_PARAMETERS_MESSAGE:
                        parameter = command_item[1]
                        if parameter.get_vehicle_id() == self.self_vhc_id:
                            if parameter.get_parameter_name() == 'target_position':
                                parameter_data = parameter.get_parameter_data()
                                # print(parameter_data)
                                self.path_following_tools.add_point_to_path(parameter_data)
                                path_modified = True
                            elif parameter.get_parameter_name() == 'target_speed_at_time':
                                # 1st parameter is the start time for the target speed in seconds as float.
                                # 2nd: how long will the target speed be active in seconds -1 for infinite/until next.
                                # 3rd parameter is the target speed.
                                parameter_data = parameter.get_parameter_data()
                                if parameter_data[1] < 0:
                                    target_length = math.inf
                                else:
                                    target_length = int(round(1000 * parameter_data[1]))
                                self.target_speed_planner.add_target_speed_data(
                                    TargetSpeedData(event_type='time',
                                                    start_time=int(round(1000 * parameter_data[0])),
                                                    length=target_length,
                                                    target_speed=parameter_data[2]))
                            elif parameter.get_parameter_name() == 'target_speed_at_segment':
                                # 1st parameter is the start segment index for the target speed.
                                # 2nd: how long will the target speed be active in seconds:
                                #  -1 for infinite/until next, 0 for during the segment
                                # 3rd parameter is the target speed.
                                parameter_data = parameter.get_parameter_data()
                                if parameter_data[1] < 0:
                                    target_length = -1
                                else:
                                    target_length = int(round(1000 * parameter_data[1]))
                                self.target_speed_planner.add_target_speed_data(
                                    TargetSpeedData(event_type='segment',
                                                    start_time=int(round(parameter_data[0])),
                                                    length=target_length,
                                                    target_speed=parameter_data[2]))

                if path_modified:
                    self.path_following_tools.smoothen_the_path()
                    self.path_following_tools.populate_the_path_with_details()
                    # print(self.path_following_tools.target_path)

            #----------Dynamic Path computation starts-------------------------
            '''
            if(cur_time_ms == 10):
                cur_position = get_self_position()
                t1 = threading.Thread(target=self.computeTargetPath, args=(cur_position,))
                t1.start() 

            
            global suboptimalPath
            if (cur_time_ms == 8000):
                t1.join()
                self.path_following_tools.target_path = None
                self.path_following_tools.path_details = None
                for pt in suboptimalPath:
                    self.path_following_tools.add_point_to_path(pt)

                self.path_following_tools.smoothen_the_path()
                self.path_following_tools.populate_the_path_with_details()
                 
                cur_position = suboptimalPath[-1]
                t1 = threading.Thread(target=self.computeTargetPath, args=(cur_position,)) 
                t1.start()

            elif (cur_time_ms % 8000 == 0):
                t1.join()
            
                # print(suboptimalPath)
                # cur_position = get_self_position()
                # (cur_seg,line_seg,nearest_pos,dis) = self.path_following_tools.get_current_segment(cur_position,0,self.path_following_tools.target_path)
                
                self.path_following_tools.target_path = self.path_following_tools.future_target_path
                self.path_following_tools.path_details = self.path_following_tools.future_path_details

                cur_position = suboptimalPath[-1]
                t1 = threading.Thread(target=self.computeTargetPath, args=(cur_position,)) 
                t1.start()
            '''
            #---------Dynamic Path computation end--------------------
            compute_and_apply_control()

        out_file = pkl_file

        

        if save:
            try:
                prevdict = pickle.load(open(out_file, "rb"))
            except (OSError, IOError) as e:
                prevdict = {}
                pickle.dump(prevdict, open(out_file, "wb"))

            # with open(out_file, 'rb') as handle:
            #     prevdict = pickle.load(handle)
            
            # print(prevdict)
            prevdict.update(data_dict)

            # print(prevdict)
            # with open(out_file, 'wb') as handle:
            #     pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


            with open(out_file, 'wb') as handle:
                pickle.dump(prevdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Clean up
        del self.classifier
        del self.obj_tracker
        print("Bye!")
        sys.stdout.flush()
