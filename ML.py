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
import dill

file_path = './'
image_path = file_path + 'images/'
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

ind = 87
def MLmodel(sample_test):
	


	eval_model = NetworkLight()
	eval_state = torch.load(file_path + 'model.h5')
	eval_model = eval_state['model']
	eval_model.float()
	eval_model.eval()



	#--------Remove this later on----------------------------------------
	with open(pkl_file, 'rb') as handle:
	    samples = pickle.load(handle)

	samples_list = [ [k, v[0], v[1], v[2]] for k, v in samples.items() ]

	# print(samples_list[ind])
	# print(samples_list[0][:])
	for ind,k in enumerate(samples_list):
		if k[3]>0.2:
			print("ind="+str(ind)+"value="+str(k[3]))
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

ind = 98
print(MLmodel(["img_"+str(ind+1)+".png",20.0]))