import os
from PIL import Image
import pandas as pd
import numpy as np
from random import shuffle
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import math

class SubsetSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class Drive360Loader(DataLoader):

    def __init__(self, config, phase):

        self.drive360 = Drive360(config, phase)
        batch_size = config['data_loader'][phase]['batch_size']
        sampler = SubsetSampler(self.drive360.indices)
        num_workers = config['data_loader'][phase]['num_workers']

        super().__init__(dataset=self.drive360,
                         batch_size=batch_size,
                         sampler=sampler
                         )

class Drive360(object):
    ## takes a config json object that specifies training parameters and a
    ## phase (string) to specifiy either 'train', 'test', 'validation'
    def __init__(self, config, phase):
        self.config = config
        self.data_dir = config['data_loader']['data_dir']
        self.csv_name = config['data_loader'][phase]['csv_name']
        self.shuffle = config['data_loader'][phase]['shuffle']
        self.history_number = config['data_loader']['historic']['number']
        self.history_frequency = config['data_loader']['historic']['frequency']
        self.normalize_targets = config['target']['normalize']
        self.target_mean = {}
        target_mean = config['target']['mean']
        for k, v in target_mean.items():
            self.target_mean[k] = np.asarray(v, dtype=np.float32)
        self.target_std = {}
        target_std = config['target']['std']
        for k, v in target_std.items():
            self.target_std[k] = np.asarray(v, dtype=np.float32)

        self.front = self.config['front']
        self.direction = config['direction']
        self.right_left = config['multi_camera']['right_left']
        self.rear = config['multi_camera']['rear']
        
        if config['cuda']['use']:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        else:
            self.device = 'cpu'

        #### reading in dataframe from csv #####
        self.dataframe = pd.read_csv(os.path.join(self.data_dir, self.csv_name),
                                     dtype={'folder_no': object,
                                            'img_no': np.int32,
                                            'Speed': np.float32,
                                            'Steering_angle': np.float32,
                                            'Cur_position': object,
                                            'Cur_yawangle': np.float32,
                                            'direction': np.float32,
                                            'weight': np.float32
                                            })

        # self.dataframe['Steering_angle'] = -1*self.dataframe['Steering_angle']*(180.0/math.pi)
        # self.dataframe['Speed'] = self.dataframe['Speed']+1
        self.sequence_length = self.history_number*self.history_frequency
        # self.here_sequence_length = self.here_frequency*self.here_number
        max_temporal_history = 1

        self.indices = self.dataframe.groupby('folder_no').apply(lambda x: x.iloc[max_temporal_history:]).index.droplevel(level=0).tolist()

        if self.normalize_targets and not phase == 'test':
            self.dataframe['Steering_angle'] = (self.dataframe['Steering_angle'].values -
                                            self.target_mean['canSteering']) / self.target_std['canSteering']
            self.dataframe['Speed'] = (self.dataframe['Speed'].values -
                                            self.target_mean['canSpeed']) / self.target_std['canSpeed']

        if self.shuffle:
            shuffle(self.indices)



        print('Phase:', phase, '# of data:', len(self.indices))

        front_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor()
            ]),
            'validation': transforms.Compose([
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.ToTensor()
            ])}
        sides_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor()
            ]),
            'validation': transforms.Compose([
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.ToTensor()
            ])}

        self.imageFront_transform = front_transforms[phase]
        self.imageSides_transform = sides_transforms[phase]


    def __getitem__(self, index):
        inputs = {}
        labels = {}
        # end = index - self.sequence_length
        # skip = int(-1 * self.history_frequency)
        row = self.dataframe.iloc[index]

        if self.front:
            x = 'cameraFront'
            inputs[x] = (self.imageFront_transform(Image.open(self.data_dir +'CNN/front/'+str(row['folder_no'])+'/'+str(row['img_no'])+'.png' ).convert("RGB")))

        if self.right_left:
            x = 'cameraRight'
            inputs[x] = (self.imageFront_transform(Image.open(self.data_dir +'CNN/right/'+str(row['folder_no'])+'/'+str(row['img_no'])+'.png' ).convert("RGB")))
            
            x = 'cameraLeft'
            inputs[x] = (self.imageFront_transform(Image.open(self.data_dir +'CNN/left/'+str(row['folder_no'])+'/'+str(row['img_no'])+'.png' ).convert("RGB")))
            
            '''
            x= 'cameraRear'
            # inputs[x] = {}
            for row_idx, (_, row) in enumerate(rows.iterrows()):
                if row['folder_no'].isdigit():
                    inputs[x] = (self.imageFront_transform(Image.open(self.data_dir +'Conv_LSTM/test/rear/straight/'+str(row['folder_no'])+'/'+str(row['img_no'])+'.png' ).convert("RGB")))
                else:
                    inputs[x] = (self.imageFront_transform(Image.open(self.data_dir +'Conv_LSTM/test/rear/'+str(row['folder_no'])+'/'+str(row['img_no'])+'.png' ).convert("RGB")))
            ''' 
        
        if self.direction:
            x = 'direction'
            inputs['direction'] = (row['direction'])

        inputs['weight'] = (row['weight'])
        labels['canSteering'] = self.dataframe['Steering_angle'].iloc[index]
        labels['canSpeed'] = self.dataframe['Speed'].iloc[index]

        return inputs, labels

