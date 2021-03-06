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

class NetworkLight(nn.Module):
    def __init__(self):
        super(NetworkLight, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 5, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.3)
        )
        self.linear_layers1 = nn.Sequential(
            nn.Linear(in_features=1152, out_features=90),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=90, out_features=10),
            nn.Dropout(p=0.3)
        )
        self.final_layer = nn.Linear(in_features=13, out_features=1)
        

    def forward(self, inp, direction):
        inp = inp.view(inp.size(0), 3, 60, 150)

        output = self.conv_layers(inp)
        output = output.view(output.size(0), -1)
        # print(output.size(1))
        output = self.linear_layers1(output)

        direction = direction.view(direction.size(0),-1)
        output = torch.cat((output,direction),dim = 1)
        output = self.final_layer(output)

        return output


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
