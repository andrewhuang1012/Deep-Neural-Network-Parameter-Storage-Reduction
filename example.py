#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import os
import math
# os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1"
transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
trainset = torchvision.datasets.ImageNet(root='/home/mitlab26/ImageNet', split='train', transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2)
testset = torchvision.datasets.ImageNet(root='/home/mitlab26/ImageNet', split='val', transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=2)
# training set: 1,281,167, validation set: 50,000 image


# In[2]:


import torchvision.models as models
Resnet50=models.resnet50(pretrained=True)
compressed_Resnet50=models.resnet50(pretrained=True)
encoding_Resnet50=models.resnet50()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# compressed_Resnet152 = nn.parallel.DataParallel(compressed_Resnet152)
Resnet50.to(device)
compressed_Resnet50.to(device)
encoding_Resnet50.to(device)


# In[3]:


# Find all convolutional layers and fully-connected layers
conv_layers=[]
for name,layer in compressed_Resnet50.named_modules():
  if isinstance(layer, torch.nn.Conv2d):
    conv_layers.append(layer)
fc_layers=[]
for name,layer in compressed_Resnet50.named_modules():
  if isinstance(layer, torch.nn.Linear):
    fc_layers.append(layer)

encoding_conv_layers=[]
for name,layer in encoding_Resnet50.named_modules():
  if isinstance(layer, torch.nn.Conv2d):
    encoding_conv_layers.append(layer)
encoding_fc_layers=[]
for name,layer in encoding_Resnet50.named_modules():
  if isinstance(layer, torch.nn.Linear):
    encoding_fc_layers.append(layer)


# In[4]:


import torch.optim as optim
def train(epochs, learning_rate, model, stop):    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total = 0
        top1 = 0
        topk = 0
        k = 5
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device),labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()
            # print('epoch: %d iteration: %d loss: %.3f' % (epoch + 1, i + 1, loss))
            _,maxk = torch.topk(outputs,k,dim=-1)
            total += labels.size(0)
            test_labels = labels.view(-1,1) # reshape labels from [n] to [n,1] to compare [n,k]

            top1 += (test_labels == maxk[:,0:1]).sum().item()
            topk += (test_labels == maxk).sum().item()
            print('Epoch: %d iteration: %d top1-accuracy: %.3f top5-accuracy: %.3f loss: %.3f' % (epoch + 1, i + 1,100*top1/total,100*topk/total ,loss))
            if i==stop:
                break
            # print('\n')
            # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
        learning_rate = learning_rate /100
#     print('Finish training model. Total training images:{}'%(total))


# In[5]:


import datetime
def evaluate_accuracy(testloader, model):
  starttime = datetime.datetime.now()
  with torch.no_grad():
    total = 0
    top1 = 0
    topk = 0
    k = 5
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # output
        outputs = model(images)
        _,maxk = torch.topk(outputs,k,dim=-1)
        total += labels.size(0)
        test_labels = labels.view(-1,1) # reshape labels from [n] to [n,1] to compare [n,k]

        top1 += (test_labels == maxk[:,0:1]).sum().item()
        topk += (test_labels == maxk).sum().item()
    endtime = datetime.datetime.now()
    print(str(endtime - starttime)+'seconds')
    print('Accuracy of the network on total {} test images: top1={}% ; top{}={}%'.format(total,100 * top1 / total,k,100*topk/total))


# In[6]:


# Linearization L(x)=f(a)+f'(a)(x-a)
def Linearization (model,p):
  model = model.clone().view(1,-1)
  sorted, index = torch.sort(model[0])
  nonzero_value = sorted[sorted.nonzero().squeeze().detach()]  # sorted value
  nonzero_index = index[sorted.nonzero().squeeze().detach()] # equal to index, nonzero value original position
  slope = []
  interval = []
  interval_index = 0
  new_value = []
  delta_x = [] # x-a
  all_fixed_points = []
  all_derivatives = []
  distance = 0
  approximation_error = 0 # True error
  average_error = 0 # Estimated error
  # Evaluate slopes at every points
    # left end-point
  m = float(nonzero_value[1]-nonzero_value[0])
  slope.append(m)

    #interior points
  for i in range(1, nonzero_value.size()[0]-1):
    m = 0.5 * float(nonzero_value[i+1]-nonzero_value[i-1])
    slope.append(m)

    # right end-point
  m = float(nonzero_value[nonzero_value.size()[0]-1]-nonzero_value[nonzero_value.size()[0]-2])
  slope.append(m)

  derivative = slope[0]
  fixed_point = nonzero_value[0]
  all_derivatives.append(derivative) # f'(0)
  all_fixed_points.append(fixed_point) # f(0)

  # Partition weights to intervals
  for j in range(0,len(slope)):
    if (j+1<len(slope) and (abs(derivative-slope[j]) > p*derivative)):  # The change of slope should be less then slope
      interval.append(interval_index)
      interval_index = interval_index + 1
      new_value.append(fixed_point + distance * derivative) # Linearization to approximate values on the same interval. distance is (x-a).
      approximation_error = approximation_error+abs((nonzero_value[j] - new_value[j]).item())
      average_error = average_error + distance*p*derivative
      delta_x.append(distance)
      derivative = float(slope[j+1])  # Assign the next slope as f'(a)
      fixed_point = float(nonzero_value[j+1]) # Assign the next point as f(a)
      all_derivatives.append(derivative)
      all_fixed_points.append(fixed_point)
      distance=0
    else:
      interval.append(interval_index)
      new_value.append(fixed_point + distance * derivative) # Linearization to approximate values on the same interval. distance is (x-a).
      approximation_error = approximation_error+abs((nonzero_value[j] - new_value[j]).item())
      average_error = average_error + distance*p*derivative
      delta_x.append(distance)
      distance = distance + 1
  approximation_error = approximation_error / len(nonzero_value)
  average_error = average_error / len(nonzero_value)
  interval = torch.tensor(interval) # Convert list to tensor

  # Convert list to tensor
  new_value = torch.tensor(new_value).to(device)
  delta_x = torch.tensor(delta_x).to(device)
  all_fixed_points = torch.tensor(all_fixed_points).to(device)
  all_derivatives = torch.tensor(all_derivatives).to(device)

  # print(alexnet.features[0].weight[0][0][0])
#   print(new_value[2000:2020])
#   print('slope:'+str(slope[2000:2020]))
#   print('distance:'+str(delta_x[2000:2020]))
#   print('approximation error:'+str(approximation_error))
#   print('average error:'+str(average_error))
  # Assign new value to model
  model[0][index]=new_value
  delta_x[index]=delta_x
  interval[index]=interval
  return model, interval, delta_x, all_fixed_points, all_derivatives


# In[7]:


# compress convolutional layers
import datetime
compressed_Resnet50.train()
starttime = datetime.datetime.now()
for l in range(1,len(conv_layers)): 
  model, interval, distance, all_fixed_points, all_derivatives = Linearization(conv_layers[l].weight,0.96) 
  model = model.view(conv_layers[l].weight.size())
  del conv_layers[l].weight
  del encoding_conv_layers[l].weight
  conv_layers[l].register_parameter('weight', nn.Parameter(model))
  encoding_conv_layers[l].register_parameter('fixed_points',nn.Parameter(all_fixed_points))
  encoding_conv_layers[l].register_parameter('derivatives',nn.Parameter(all_derivatives))
  encoding_conv_layers[l].register_parameter('interval',nn.Parameter(interval.type(torch.uint8),False))
  encoding_conv_layers[l].register_parameter('distance',nn.Parameter(distance.type(torch.uint8),False))
  conv_layers[l].weight.requires_grad=False
endtime = datetime.datetime.now()
print(str(endtime - starttime)+'seconds')


# In[8]:


# compress fully-connected layers
import datetime
starttime = datetime.datetime.now()
for l in range(len(fc_layers)): 
  model, interval, distance, all_fixed_points, all_derivatives = Linearization(fc_layers[l].weight,0.95) 
  model = model.view(fc_layers[l].weight.size())
  del fc_layers[l].weight
  del encoding_fc_layers[l].weight
  fc_layers[l].register_parameter('weight', nn.Parameter(model))
  encoding_fc_layers[l].register_parameter('fixed_points',nn.Parameter(all_fixed_points))
  encoding_fc_layers[l].register_parameter('derivatives',nn.Parameter(all_derivatives))
  encoding_fc_layers[l].register_parameter('interval',nn.Parameter(interval.type(torch.uint8),False))
  encoding_fc_layers[l].register_parameter('distance',nn.Parameter(distance.type(torch.uint8),False))
  fc_layers[l].weight.requires_grad=False
endtime = datetime.datetime.now()
print(str(endtime - starttime)+'seconds')


# In[11]:


# compressed_Resnet50.eval()
# evaluate_accuracy(testloader, compressed_Resnet50)
Resnet50.eval()
evaluate_accuracy(testloader, Resnet50)


# In[14]:


PATH = './96compressed_Resnet50.pth' 
torch.save(compressed_Resnet50.state_dict(), PATH)
PATH2 = './96encoding_Resnet50.pth' 
torch.save(encoding_Resnet50.state_dict(), PATH2)


# In[12]:


for name, para in encoding_Resnet50.named_parameters():
  print(name+":"+str(para.numel()))


# In[13]:


for name, para in compressed_Resnet50.named_parameters():
  print(name+":"+str(para.numel()))





