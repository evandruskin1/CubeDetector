import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from CubeDetector1 import CubeDetector1

def RightHalfImage(pillow_image):
    #print("pillow image shape: ", pillow_image)
    r_img = pillow_image.crop((160,0,320,240))
    r_img = r_img.convert("L")
   # n_img = np.array(r_img)
    #print("np array shape ",n_img.shape)
    # subscript [:, 160:, 1] pulls the right half of the image
    #t_img = torch.tensor(n_img[:, 160:, 1], dtype=torch.float).view(1,240,160)
    # print('shapes', n_img.shape, t_img.shape)
    return r_img

trainset = torchvision.datasets.folder.ImageFolder(
  root="./dataset1",
  transform=transforms.Compose([
    RightHalfImage,
    torchvision.transforms.RandomCrop([235,155]),
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize(mean=[0.450], std=[0.230], inplace=True)
    ]
    )    
)

batchSize = 250
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batchSize,shuffle=True)
# For debugging...
pats = trainloader.__iter__().__next__()

# Uncomment just one of the two lines below:
device = torch.device(0)        # GPU board
#device = torch.device('cpu')    # regular CPU

in_dim = (155,235)  # right half of 320x240 Cozmo camera image
model = CubeDetector1(in_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

nepochs = 15
epoch = 0

def train_model():
  global epoch
  npats = len(trainset.samples)
  print('Epoch  Error    Time         Correct')
  for i in range(nepochs):
    epoch += 1
    runningLoss = 0.0
    now = time.time()
    correct = 0.0 
    for (images,labels) in trainloader:
      #print("original image shape: ",images.shape)
      #images = images.reshape(-1,240,160)
      #toPIL = torchvision.transforms.ToPILImage()
      #crop = torchvision.transforms.RandomCrop([235,155])
      #toTensor = torchvision.transforms.ToTensor()
      #images = toPIL(images)
      #images = images.convert("L")
      #print("after Pil: ", images.size)
      #images = crop(images)
      #print("after crop: ",images.size)
      #images = toTensor(crop(toPIL(images)))
      #images = toTensor(images)
      #print("after tensor: ", images.shape)
      #images = np.array(images)
      #print(np.asarray(images) is images)
      images = images.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      #print(images.shape)
      #images = images.reshape(-1,1,235,155)
      outputs = model(images)
      #print(outputs)
      #print(outputs.shape)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      runningLoss += loss.item() * batchSize
      correct += np.array(outputs.cpu()[:,1].sign() == (labels.cpu()*2-1), dtype='int').sum()
      print('%03d   %8.4f  time=%5.4f %5d %8.4f' % (epoch, runningLoss,  time.time()-now, correct,correct/npats * 100))
def show_pattern(n=0):
  patterns = trainloader.__iter__().__next__()
  img = np.array(patterns[0][n].view(-1,155))
  plt.imshow(img,cmap='gray')
  plt.xlabel('Class: %s' % ['Cube','No Cube'][patterns[1][n]])
  plt.pause(0.5)

