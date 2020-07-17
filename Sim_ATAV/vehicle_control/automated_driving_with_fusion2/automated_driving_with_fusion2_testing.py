# To run this file, rename it to automated_driving_with_fusion2.py



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
import pandas as pd
import json
from PIL import Image

# import dill

file_path = '../../../'
image_path = file_path + 'images/'
pkl_file = file_path + 'control_throttle.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
config = json.load(open('../../../config.json'))
test = True


def generate_input(index, config, direction):
    inputs = {}
    labels = {}
    phase = 'test'
    # sequence_length = 2
    # history_frequency = 4
    # end = index - sequence_length
    # skip = int(-1 * history_frequency)
    
    # rows = []
    # for i in range(index,end,skip):
    #     rows.append(i)


    front_transforms = {
                'test': transforms.Compose([
                    transforms.Resize((240, 135)),
                    transforms.ToTensor()
                    # ,
                    # transforms.Normalize(mean=config['image']['norm']['mean'],
                    #                      std=config['image']['norm']['std'])
                ])}
    sides_transforms = {
        'test': transforms.Compose([
            transforms.Resize((240, 135)),
            transforms.ToTensor()
            # ,
            # transforms.Normalize(mean=config['image']['norm']['mean'],
            #                      std=config['image']['norm']['std'])
        ])}
    
    imageFront_transform = front_transforms[phase]
    imageSides_transform = sides_transforms[phase]

    data_dir = '../../../Conv_LSTM/test'
    x = 'cameraFront'
    inputs[x] =  imageFront_transform(Image.open(data_dir + '/front/' + str(index) + '.jpg')).unsqueeze(0)

    x = 'cameraRight'
    inputs[x] =  imageFront_transform(Image.open(data_dir + '/right/' + str(index) + '.jpg')).unsqueeze(0)
    
    x = 'cameraLeft'
    inputs[x] =  imageFront_transform(Image.open(data_dir + '/left/' + str(index) + '.jpg')).unsqueeze(0)

    inputs['cameraRear'] = imageFront_transform(Image.open(data_dir + '/rear/' + str(index) + '.jpg')).unsqueeze(0)

    x = 'direction'
    # rows = {}
    # rows['direction'] = 0
    # row = pd.DataFrame(rows)
    # inputs['direction'] = (rows['direction'].iloc[0])
    inputs['direction'] = torch.tensor([direction])
    return inputs

def toDevice(datas, device):
    imgs, vel, one_hot = datas
    return imgs.float().to(device), vel.float().to(device), one_hot.float().to(device)

def testing(model, test_generator):
    model.eval()
    with torch.set_grad_enabled(False):
        outputs = model(test_generator)
    return outputs

class CONV_LSTM(nn.Module):
    def __init__(self):
        super(CONV_LSTM, self).__init__()
        final_concat_size = 0
        self.split_gpus = False
        
        in_chan_1 = 3
        out_chan_1 = 12
        out_chan_2 = 24
        kernel_size = 5
        stride_len = 2
        
        # CNN_output_size = 2688
        CNN_output_size = 2352
        
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(in_chan_1, out_chan_1, kernel_size, stride = stride_len),
            nn.ELU(),
            nn.Conv2d(out_chan_1, out_chan_2, kernel_size, stride = stride_len),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.3)
        )

        self.conv_layers3 = nn.Sequential(
            nn.Conv2d(in_chan_1, out_chan_1, kernel_size, stride = stride_len),
            nn.ELU(),
            nn.Conv2d(out_chan_1, out_chan_2, kernel_size, stride = stride_len),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.3)
        )

        self.conv_layers4 = nn.Sequential(
            nn.Conv2d(in_chan_1, out_chan_1, kernel_size, stride = stride_len),
            nn.ELU(),
            nn.Conv2d(out_chan_1, out_chan_2, kernel_size, stride = stride_len),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.3)
        )
        
        # Fully Connected Layers
        # self.dir_fc = nn.Sequential(
        #                 nn.Linear(1, 4),
        #                 nn.ReLU())
        # final_concat_size += 4

        self.front_fc = nn.Sequential(
                          nn.Linear(CNN_output_size, 512),
                          nn.ReLU(),
                          nn.Linear(512, 32),
                          nn.ReLU())
        final_concat_size += 32

        self.left_fc = nn.Sequential(
                          nn.Linear(CNN_output_size, 256),
                          nn.ReLU(),
                          nn.Linear(256, 16),
                          nn.ReLU())
        final_concat_size += 16

        self.right_fc = nn.Sequential(
                          nn.Linear(CNN_output_size, 256),
                          nn.ReLU(),
                          nn.Linear(256, 16),
                          nn.ReLU())
        final_concat_size += 16

        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(final_concat_size+1, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1)
        )
        
            
    def forward(self, data):
        module_outputs = []

        # x = data['direction']
        # x = x.view(x.size(0), -1)
        # x = self.dir_fc(x)
        # x = x.to(device)
        # module_outputs.append(x)

        v =  data['cameraFront']
        x = self.conv_layers1(v)
        x = x.view(x.size(0), -1)
        x = self.front_fc(x)
        module_outputs.append(x)

        v = data['cameraLeft']
        x = self.conv_layers3(v)
        x = x.view(x.size(0), -1)
        x = self.left_fc(x)
        module_outputs.append(x)

        v = data['cameraRight']
        x = self.conv_layers4(v)
        x = x.view(x.size(0), -1)
        x = self.right_fc(x)
        module_outputs.append(x)

        # x = torch.FloatTensor([1.0])
        # x = x.view(x.size(0), -1)
        # x = x.to(device)
        # x = torch.ones(2,1)
        # print(x.size())

        temp_dim = config['data_loader']['train']['batch_size']

        if test:
            temp_dim = 1

        module_outputs.append(torch.ones(temp_dim, 1).to(device))

        # print(module_outputs[0].shape)
        x_cat = torch.cat(module_outputs, dim=-1)

        # Feed concatenated outputs into the 
        # regession networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat))}
        return prediction
model_val = 1
gpu_avail = True

############ Model 1 ###############

if model_val == 1:
    eval_model = CONV_LSTM()
    if gpu_avail:
        eval_state = torch.load('../../../Conv_LSTM/models/'+config['model']['category']+'/'+config['model']['type']+'/epoch'+str(config['model']['test_epoch'])+'.h5')
    else:
        eval_state = torch.load('../../../Conv_LSTM/epoch20.h5', map_location = 'cpu')
    eval_model.load_state_dict(eval_state['state_dict'])
    eval_model.eval()
    eval_model = eval_model.to(device)


############ Model 2 ###############

elif model_val == 2:
    left_model = NetworkLight()
    left_state = torch.load(file_path + 'models/left_model2.h5')
    left_model = left_state['model']
    left_model.float().to(device)
    left_model.eval()

    # right_model = NetworkLight()
    # right_state = torch.load(file_path + 'models/right_model2.h5')
    # right_model = right_state['model']
    # right_model.float().to(device)
    # right_model.eval()

    st_model = NetworkLight()
    st_state = torch.load(file_path + 'models/st_model2.h5')
    st_model = st_state['model']
    st_model.float().to(device)
    st_model.eval()


# Our global variables
target_throttle = [0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 0.65, 0.7, 0.7, 0.7, 0.75, 0.6, 0.6, 0.6, 0.35, 0.35, -0.3, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.2, -0.2, -0.2, -0.2, -0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, -0.1, 0.15, 0.3, 0.55, 0.65, 0.75, 0.85, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 0.65, 0.65, 0.7, 0.7, 0.7, 0.75, 0.6, 0.6, 0.6, 0.35, 0.35, -0.3, -0.3, -0.3, -0.4, -0.4, -0.4, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1, 0.05, 0.05, 0.05, 0.05, 0.05, -0.1]

target_t = [1.58, 2.23, 2.73, 3.15, 3.52, 3.86, 4.17, 4.47, 4.75, 5.02, 5.28, 5.53, 5.77, 6.01, 6.24, 6.47, 6.7, 6.92, 7.14, 7.36, 7.58, 7.8, 8.02, 8.26, 8.51, 8.78, 9.07, 9.39, 9.74, 10.14, 10.58, 11.08, 11.68, 12.39, 13.14, 13.89, 14.64, 15.39, 16.14, 16.89, 17.68, 18.47, 19.15, 19.7, 20.15, 20.54, 20.89, 21.21, 21.51, 21.8, 22.08, 22.34, 22.59, 22.84, 23.08, 23.32, 23.55, 23.78, 24.01, 24.23, 24.45, 24.67, 24.89, 25.11, 25.33, 25.57, 25.82, 26.09, 26.38, 26.7, 27.05, 27.42, 27.83, 28.29, 28.82, 29.47, 30.26, 31.11, 31.96, 32.81, 33.66, 34.51, 35.43]

exp_out = [[]]

time_index = 0
img_cnt = 0
data_dict = {}
str_angle = 0.0

inf = 1e9

def convert_to_CUDA(x):
    for k, v in x.items():
        x[k] = v.to(device)

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
        
        self.camera_device_name_front = 'camera_front'
        self.camera_front = None
        self.camera_device_name_left = 'camera_left'
        self.camera_left = None
        self.camera_device_name_right = 'camera_right'
        self.camera_right = None
        self.camera_device_name_rear = 'camera_rear'
        self.camera_rear = None


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
        # self.camera = self.getCamera(self.camera_device_name)
        # if self.camera is not None:
        #     self.camera.enable(self.CLASSIFIER_PERIOD_MS)
        
        self.camera_front = self.getCamera(self.camera_device_name_front)
        self.camera_rear = self.getCamera(self.camera_device_name_rear)
        self.camera_left = self.getCamera(self.camera_device_name_left)
        self.camera_right = self.getCamera(self.camera_device_name_right)
        
        
        if self.camera_front is not None:
            self.camera_front.enable(self.CLASSIFIER_PERIOD_MS)
        if self.camera_rear is not None:
            self.camera_rear.enable(self.CLASSIFIER_PERIOD_MS)
        if self.camera_left is not None:
            self.camera_left.enable(self.CLASSIFIER_PERIOD_MS)
        if self.camera_right is not None:
            self.camera_right.enable(self.CLASSIFIER_PERIOD_MS)
        
        
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

                global str_angle

                if cur_time_ms%100==0:
                    global img_cnt
                    img_name = str(img_cnt) + ".jpg"
                    self.camera_front.saveImage("../../../Conv_LSTM/test/front/"+img_name,1)
                    self.camera_rear.saveImage("../../../Conv_LSTM/test/rear/"+img_name,1)
                    self.camera_left.saveImage("../../../Conv_LSTM/test/left/"+img_name,1)
                    self.camera_right.saveImage("../../../Conv_LSTM/test/right/"+img_name,1)
                    img_cnt = img_cnt + 1

                if cur_time_ms<3010:
                    x = 0.0
                    self.set_target_speed_and_angle(speed= controller_commons.speed_ms_to_kmh(x) ,angle=control_steering)
                else:
                    x = 3.0
                    direction = 1.0
                    # if cur_position[0]<=320.0 and cur_position[1]<=15:
                    #     direction = -1

                    data = generate_input(img_cnt-1,config, direction)
                    convert_to_CUDA(data)
                    Result = testing(eval_model, data)
                    speed = x
                    steering = float(Result['canSteering'])
                    # steering = ((config['target']['std']['canSteering']*steering)+config['target']['mean']['canSteering'])
                    # speed = ((config['target']['std']['canSpeed']*speed)+config['target']['mean']['canSpeed'])
                    
                    print("speed = "+ str(speed) + " steering = " + str(steering) + " cur_speed = " + str(cur_speed_ms))
                    # control_throttle = self.longitudinal_controller.compute(speed
                    #                                                     - cur_speed_ms)
                    # self.set_throttle_and_steering_angle(control_throttle, steering)
            
                    self.set_target_speed_and_angle(speed = controller_commons.speed_ms_to_kmh(x), angle = steering)
                if cur_time_ms%500==0:
                    print("Time: "+str(cur_time_ms)+" Agent vehicle speed: "+str(cur_speed_ms) + " pos: "+str(cur_position))
            
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

            compute_and_apply_control()

        # out_file = "../../../control_throttle.pkl"
        
        # with open(out_file, 'rb') as handle:
        #     prevdict = pickle.load(handle)
        
        # # print(prevdict)
        # prevdict.update(data_dict)

        # # print(prevdict)
        # with open(out_file, 'wb') as handle:
        #     pickle.dump(prevdict, handle)

        # Clean up
        del self.classifier
        del self.obj_tracker
        print("Bye!")
        sys.stdout.flush()
