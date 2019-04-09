'''
Deform-Covnet
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)
'''
import os
import random
import time
import numpy as np
import torch
import math
import torch.nn as nn
from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset import sonar,sonartest
from transform import Relabel, ToLabel, Colorize
#from visualize import Dashboard

import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile
#from erfnet import ERFNet
from Edanet import EDANet

NUM_CHANNELS = 3
NUM_CLASSES = 2 #pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

#Augmentations - different function implemented to perform random augments on both image and target

image_transform = ToPILImage()
input_transform = Compose([
    Resize((512,512)),
    #CenterCrop(256),
    ToTensor(),
    #Normalize([112.65,112.65,112.65],[32.43,32.43,32.43])
    #Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform = Compose([
    Resize((512,512)),
    #CenterCrop(324),
    ToLabel(),
    #Relabel(255, 1),
])

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def train( model):
    best_acc = 0
    num_epochs=60
     
    loader = DataLoader(train(input_transform, target_transform),num_workers=1, batch_size=4, shuffle=True)
   
    criterion = nn.CrossEntropyLoss()
    #criterion = CrossEntropyLoss2d(weight)

        
    #print(type(criterion))

    savedir = '/home/uvl/tk1/EDA/Deform-save'


    automated_log_path = savedir + "/log.txt"
    modeltxtpath = savedir + "/model.txt"

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), 1e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    #optimizer = Adam(model.parameters(), 1e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)      ## scheduler 2

    start_epoch = 1

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lr_updater = lr_scheduler.StepLR(optimizer, 100,
                                     0.1)                             ## scheduler 2
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model
    #state='/home/uvl/tk1/EDA/save1/model/main-erfnet_my-step-998-epoch-20.pth'
    #print('LODA MODEL!')
    #model = load_my_state_dict(model, torch.load(state))


    for epoch in range(start_epoch, num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        lr_updater.step()

        epoch_loss = []
        time_train = []

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):

            start_time = time.time()
            
            images = images.cuda()
            labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            #print inputs.size(),targets.size()
            outputs = model(inputs)
 
            #print outputs.size()
            optimizer.zero_grad()
            loss = criterion(outputs, targets[:,0])
            loss.backward()
            optimizer.step()
            #print loss.item()
            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)
     

            if step % 100 == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print('epoch:%f'%epoch,'step:%f'%step,'loss:%f'%average)

            with open(automated_log_path, "a") as myfile:
                myfile.write("\n%d\t\t%d\t\t%.4f" % (epoch, step,average ))
        if epoch % 1 == 0 and epoch != 0:

            filename = 'main-'+'Deform-eda'+'-step-'+str(step)+'-epoch-'+str(epoch)+'.pth'
            torch.save(model.state_dict(), '/home/uvl/tk1/EDA/Deform-save/model/'+filename)
       # print "SAVE MODEL!!"
        #filename = 'main-'+'erfnet_my'+'-epoch-'+str(epoch)+'.pth'
        #torch.save(model.state_dict(), '/home/uvl/tk1/save/model/'+filename)
            
        #average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        #with open(automated_log_path, "a") as myfile:
            #myfile.write("\n%d\t\t%.4f" % (epoch, average_epoch_loss_train ))
    
    return(model)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main():
    savedir = '/home/uvl/tk1/EDA/Deform-save'

    Net = EDANet(NUM_CLASSES)


    Net = Net.cuda()


    train(Net)

if __name__ == '__main__':
    main()
