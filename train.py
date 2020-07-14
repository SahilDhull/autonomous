from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import json
import numpy as np
from dataset import Drive360Loader
import sys
import math

test = False

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = json.load(open('./config.json'))

log_file = config['model']['save_path'] + 'training.txt'
logs = open(log_file, 'w')

model = CONV_LSTM()
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params)
# exit()
# device = 'cpu'
model = model.to(device)
criterion = nn.SmoothL1Loss()
criterion = criterion.to(device)
learning_rate = 1e-4
# optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.7)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def convert_to_CUDA(x, y):
    for k, v in x.items():
        x[k] = v.to(device)
    for k, v in y.items():
        y[k] = v.to(device)
    

# state = torch.load('./'+config['model']['path']+config['model']['category']+'/'+config['model']['type']+'/epoch49.h5')
# model.load_state_dict(state['state_dict'])
# optimizer.load_state_dict(state['optimizer'])


train_loader = Drive360Loader(config, 'train')
# print("check")

total_epochs = 100
print("total_epochs = "+str(total_epochs)+"\nlearning rate = "+str(learning_rate))
print("batch size = "+str(config['data_loader']['train']['batch_size']))

print("total_epochs = "+str(total_epochs)+"\nlearning rate = "+str(learning_rate), file = logs)
print("batch size = "+str(config['data_loader']['train']['batch_size']), file = logs)

logs.close()
    
for epoch in range(0, total_epochs):
    model = model.train()
    running_loss = 0.0
    train_loss = 0.0
    # print(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx)
        model.zero_grad()
        optimizer.zero_grad()
        convert_to_CUDA(data, target)
        prediction = model(data)
        w = data['weight']
        # w = w.view(w.size(0), -1)
        loss = criterion(w*prediction['canSteering'], w*target['canSteering'])
        loss = loss.to(device)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # if batch_idx % batch_skip == 0:
        #     print(batch_idx)
            
    print('epoch: %d, train loss: %.5f' %(epoch, train_loss/(1.0*len(train_loader))))
    # print_to_logs('epoch: %d, train loss: %.5f' %(epoch, train_loss/(1.0*len(train_loader))))
    
    if (epoch+1)%10==0:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state,config['model']['save_path']+'epoch'+str(epoch)+'.h5')
    # val_loss = validation()
    # if val_loss < min_val_loss:
    #     min_val_loss = val_loss
    #     print("Min val loss = %.5f at epoch %d" %(min_val_loss, epoch))
    #     print_to_logs("Min val loss = %.5f at epoch %d" %(min_val_loss, epoch))
