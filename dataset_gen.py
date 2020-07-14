# Combines control throttle into our the csv

import torch
import torch.nn as nn
import torch.optim as optim
# Used to generate csv for dataset

from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import csv
import pickle
import random
import pandas as pd
import json
from PIL import Image
import math
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
config = json.load(open('./config.json'))

with open(config['data_loader']['train']['csv_name'], 'w', newline = '') as csvfile:
    file = csv.writer(csvfile)
    file.writerow(['folder_no','img_no','Speed','Steering_angle','Cur_position','Cur_yawangle','direction','weight'])
    data_type = 'straight'
    folder = 'straight'
    direc = 0.0
    w = 1
    for folder_no in range(0,6):
        pkl_file = './CNN/control_throttle/'+data_type+'/control_throttle_'+str(folder_no)+'.pkl'
        with open(pkl_file, 'rb') as handle:
            samples = pickle.load(handle)
            for k, v in samples.items():
                file.writerow([data_type+'/'+str(folder_no), k, v[0], v[1], v[2], v[3], direc, w])
    data_type = 'correction'
    for folder_no in range(0,27):
        pkl_file = './CNN/control_throttle/'+data_type+'/control_throttle_'+str(folder_no)+'.pkl'
        with open(pkl_file, 'rb') as handle:
            samples = pickle.load(handle)
            for k, v in samples.items():
                file.writerow([data_type+'/'+str(folder_no), k, v[0], v[1], v[2], v[3], direc, w])

    
    folder = 'right_turn'
    direc = 1.0
    data_type = 'right_new'
    for folder_no in range(0,36):
        pkl_file = './CNN/control_throttle/'+data_type+'/control_throttle_'+str(folder_no)+'.pkl'
        with open(pkl_file, 'rb') as handle:
            samples = pickle.load(handle)
            for k, v in samples.items():
                file.writerow([data_type+'/'+str(folder_no), k, v[0], v[1], v[2], v[3], direc, w])

    data_type = 'correction_right'
    for folder_no in range(0,34):
        pkl_file = './CNN/control_throttle/'+data_type+'/control_throttle_'+str(folder_no)+'.pkl'
        with open(pkl_file, 'rb') as handle:
            samples = pickle.load(handle)
            for k, v in samples.items():
                file.writerow([data_type+'/'+str(folder_no), k, v[0], v[1], v[2], v[3], direc, w])

    

