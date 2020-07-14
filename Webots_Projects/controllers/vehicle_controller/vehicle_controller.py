"""vehicle_controller is the vehicle controller known to Webots.
It calls the actual controller which is given to it as a parameter.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from importlib import import_module
import os
import sys
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_PATH + "/../../../")
#LIBRARY_PATH = os.environ.get("WEBOTS_HOME") + "/projects/automobile/libraries/python"
try:
    LIBRARY_PATH = os.environ.get("WEBOTS_HOME") + "/lib/python36"
except:
    LIBRARY_PATH = 'C:/Program Files/Webots/lib/python36'
LIBRARY_PATH.replace('/', os.sep)
sys.path.append(LIBRARY_PATH)

# --------------------------------------------
#Added for ML part

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
# import numpy as np
import csv
import pickle
import random
# import dill

# class NetworkLight(nn.Module):
#     def __init__(self):
#         super(NetworkLight, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 24, 5, stride=2),
#             nn.ELU(),
#             nn.Conv2d(24, 48, 5, stride=2),
#             nn.MaxPool2d(4, stride=4),
#             nn.Dropout(p=0.3)
#         )
#         self.linear_layers1 = nn.Sequential(
#             nn.Linear(in_features=48*18*36, out_features=90),
#             nn.ELU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(in_features=90, out_features=10),
#             nn.Dropout(p=0.3)
#         )
#         self.final_layer = nn.Linear(in_features=13, out_features=1)
        

#     def forward(self, inp, direction):
#         inp = inp.view(inp.size(0), 3, 310, 600)

#         output = self.conv_layers(inp)
#         output = output.view(output.size(0), -1)
#         # print(output.size(1))
#         output = self.linear_layers1(output)

#         direction = direction.view(direction.size(0),-1)
#         output = torch.cat((output,direction),dim = 1)
#         output = self.final_layer(output)

#         return output

test = True

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
        
# ---------------------------------------------
print("Vehicle controller will load.")

if len(sys.argv) > 1:
    controller_name = sys.argv[1]

    if controller_name is None or type(controller_name) is not str:
        print("Controller name is not given")
    else:
        if sys.platform == 'win32':  # Windows
            path_to_controller = FILE_PATH + "\\..\\" + controller_name
        else:  # Linux / Mac OS
            path_to_controller = FILE_PATH + "/../" + controller_name
        print("Controller: {}, Path:{}".format(controller_name, path_to_controller))
        sys.path.append(path_to_controller)
        #controller = __import__(controller_name)
        controller = import_module(controller_name)
        print(controller)
        methodToCall = getattr(controller, 'run')
        print(methodToCall)
        print(sys.argv[2:])
        robot = methodToCall(sys.argv[2:])
        robot()
