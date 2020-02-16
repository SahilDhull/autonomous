import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import csv
import pickle
import random

file_path = './'
image_path = file_path + 'images/'
pkl_file = file_path + 'control_throttle.pkl'



with open(pkl_file, 'rb') as handle:
    samples = pickle.load(handle)

samples_list = [ [k, v[0], v[1], v[2]] for k, v in samples.items() ]

# print (samples_list[0])
# print(type(samples))

# Step2: Divide the data into training set and validation set
l = len(samples_list)
print(l)
train_len = (int((0.8*l)/32))*32
valid_len = (int((l - train_len)/32))*32

random.shuffle(samples_list)

train_samples = samples_list[:train_len]
validation_samples = samples_list[l-valid_len:]


# Step3a: Define the augmentation, transformation processes, parameters and dataset for dataloader
def augment(imgName, angle):
	name = image_path + imgName
	current_image = cv2.imread(name)
	# print(type(current_image))
	# print(current_image.shape)
	# h = 400, w = 600, colors = 3
	current_image = current_image[65:-25, :, :]
	# h = 310, w = 600, colors = 3
	# print(type(current_image))
	# print(current_image.shape)
	if np.random.rand() < 0.5:
		current_image = cv2.flip(current_image, 1)
		angle = angle * -1.0  
	return current_image, angle

class Dataset(data.Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        batch_samples = self.samples[index]
        steering_angle = float(batch_samples[3])
        center_img, steering_angle_center = augment(batch_samples[0], steering_angle)
        # left_img, steering_angle_left = augment(batch_samples[1], steering_angle + 0.4)
        # right_img, steering_angle_right = augment(batch_samples[2], steering_angle - 0.4)
        center_img = self.transform(center_img)
        # left_img = self.transform(left_img)
        # right_img = self.transform(right_img)
        return (center_img, batch_samples[1] , batch_samples[2] , steering_angle_center)
      
    def __len__(self):
        return len(self.samples)
                                  

# Step3b: Creating generator using the dataloader to parallasize the process
transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0) - 0.5)])

params = {'batch_size': 32,
          'shuffle': True}

training_set = Dataset(train_samples, transformations)
training_generator = DataLoader(training_set, **params)


# validation_samples[0:32] is written to prevent error, since batch_size is 32
# CHANGE it later on when having more dataset
validation_set = Dataset(validation_samples, transformations)
validation_generator = DataLoader(validation_set, **params)






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


model = NetworkLight().float()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

criterion = nn.MSELoss()

# Step6: Check the device and define function to move tensors to that device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print('device is: ', device)

def toDevice(datas, device):
  
  imgs, vel, throttle, angles = datas
  return imgs.float().to(device), vel.float().to(device), throttle.float().to(device), angles.float().to(device)




def validation(model, validation_generator):
	print("")
	print("==========  Validation  ==========")
	print("")
	model.eval()
	valid_loss = 0
	with torch.set_grad_enabled(False):
		for local_batch, data in enumerate(validation_generator):
		# Transfer to GPU
			data = toDevice(data, device)

			# Model computations
			# optimizer.zero_grad()
			imgs, vel, throttle, angles = data
			with torch.no_grad():
				outputs = model(imgs,vel)

			throttle = torch.unsqueeze(throttle, 1)
			angles = torch.unsqueeze(angles,1)

			out_label = torch.cat([throttle,angles],dim=1)

			loss = criterion(output, out_label)

			valid_loss += loss.data.item()

			if local_batch % 16 == 0:
				print('Valid Loss: %.5f '
				% (valid_loss/(local_batch+1)))
	print("")




print(len(training_generator))

max_epochs = 0
for epoch in range(max_epochs):
    print("")
    print("==========  Epoch %d  ==========" %epoch)
    model.to(device)
    
    # Training
    train_loss = 0
    model.train()
    for local_batch, data in enumerate(training_generator):
        # Transfer to GPU
        data = toDevice(data, device)
        
        # Model computations
        optimizer.zero_grad()
        model.zero_grad()
        model.train()
        imgs, vel, throttle, angles = data

        output = model(imgs,vel)

        throttle = torch.unsqueeze(throttle, 1)
        angles = torch.unsqueeze(angles,1)

        out_label = torch.cat([throttle,angles],dim=1)

        loss = criterion(output, out_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.data.item()
            
        if local_batch % 40 == 0:
            print('Loss: %.5f '
                % (train_loss/(local_batch+1)))
    validation(model, validation_generator)
 


'''
 # Step8: Define state and save the model wrt to state
state = {
        'model': model.module if device == 'cuda' else model,
        }

torch.save(state, file_path + 'model.h5')

#Model saved
'''