# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:14:01 2021

@author: zhang
"""
import torch
import torch.nn as nn
import pywt




DIAG = 0   #use diag pooling
SFP = 1    #use scalable frequency pooling



class CoordPooling(nn.Module):
    def __init__(self,h,w):
        super(CoordPooling, self).__init__()       
        
        self.avgpool_h = nn.AvgPool2d(kernel_size=(1,h), stride=1, padding=0)    
        self.avgpool_w = nn.AvgPool2d(kernel_size=(w,1), stride=1, padding=0)    

    def forward(self,x):
        x_h = self.avgpool_h(x)
        x_w = self.avgpool_w(x)
        x_w = x_w.permute(0,1,3,2)
        return x_h,x_w


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)        








class SFP(nn.Module):
    def __init__(self,gate_channel,reduction_ratio=16, num_layers=1):
        super(SFP, self).__init__()        
        
        self.levels = 3   #set your level of SFP here
        
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]))
            self.gate_c.add_module('gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]))
            self.gate_c.add_module('gate_c_relu_%d'%(i+1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))        
        



    
    def wavelet(self,y):
       #print(y.shape)
        y_level = y.chunk(self.levels,dim=1)
        y_level = y_level
        for i in range(1,self.levels+1):
            k = y_level[i-1]
            k = k.cpu().detach().numpy()
            t = pywt.wavedec(k,wavelet = 'haar',mode='symmetric', level=i, axis=2)[0]             # wavelet = 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'
            t = torch.Tensor(t)
            t = t.cuda()
            if i == 1:
                z = torch.mean(t,dim=2) 
               #print(z.shape)
            else:
                z = torch.cat([z,torch.mean(t,dim=2)],dim=1)
               #print(z.shape)
        z=z.squeeze()        
        z=z.unsqueeze(2).unsqueeze(3)
        return z
    
    
    def forward(self,y):        
        z = self.wavelet(y)
        
        z = self.gate_c(z)
        
        return z   









class Attention(nn.Module):
    
    
    def __init__(self,h,w,c):
        super(Attention, self).__init__()      

        
        
        num_channel = c
        
        
        self.conv_y = nn.Sequential(nn.Conv2d(num_channel, num_channel, kernel_size=1, stride = 1, padding = 0, bias=False), #卷积核为1，步长为1，补足为0
                                   nn.BatchNorm2d(num_channel),
                                   nn.ReLU(inplace=True),)
        self.conv_diag = nn.Sequential(nn.Conv2d(num_channel, num_channel, kernel_size=1, stride = 1, padding = 0, bias=False), #卷积核为1，步长为1，补足为0
                                   nn.BatchNorm2d(num_channel),
                                   nn.ReLU(inplace=True),)
        
        self.conv_x_h = nn.Sequential(nn.Conv2d(num_channel, num_channel, kernel_size=1, stride = 1, padding = 0, bias=False), #卷积核为1，步长为1，补足为0
                                   nn.BatchNorm2d(num_channel),
                                   nn.ReLU(inplace=True),)
        self.conv_x_w = nn.Sequential(nn.Conv2d(num_channel, num_channel, kernel_size=1, stride = 1, padding = 0, bias=False), #卷积核为1，步长为1，补足为0
                                   nn.BatchNorm2d(num_channel),
                                   nn.ReLU(inplace=True),)
        self.conv_diag1 = nn.Sequential(nn.Conv2d(num_channel, num_channel, kernel_size=1, stride = 1, padding = 0, bias=False), #卷积核为1，步长为1，补足为0
                                   nn.BatchNorm2d(num_channel),
                                   nn.ReLU(inplace=True),)
        self.conv_diag2 = nn.Sequential(nn.Conv2d(num_channel, num_channel, kernel_size=1, stride = 1, padding = 0, bias=False), #卷积核为1，步长为1，补足为0
                                   nn.BatchNorm2d(num_channel),
                                   nn.ReLU(inplace=True),)
        
        
        self.coord = CoordPooling(h,w)
        
        self.sfp = SFP(c)
        
        
        
    
    def diag(self,x):
        x2 = torch.flip(x,dim=2)
        
        diag1=torch.ones((x.shape[0],x.shape[1],2*x.shape[2]-1,1))
        diag1=diag1.cuda()
        diag2=torch.ones((x.shape[0],x.shape[1],2*x.shape[2]-1,1))
        diag2=diag2.cuda()

        for i in range(-x.shape[2]+1,x.shape[2]):
            d = torch.diagonal(x,i)
            m = torch.mean(d,dim=2)
            diag1[:,:,i+x.shape[2]-1,1]=m
            
            d = torch.diagonal(x2,i)
            m = torch.mean(d,dim=2)
            diag2[:,:,i+x2.shape[2]-1,1]=m
        return diag1,diag2
            
 
    
    def diagback(self,x,diag1,diag2):
        for i in range(-x.shape[2]+1,x.shape[2]):
            d = torch.diagonal(x,i)
            m = torch.mean(d,dim=2)
            diag1[:,:,i+x.shape[2]-1,1]=m
            
            d = torch.diagonal(x,i)
            m = torch.mean(d,dim=2)
            diag2[:,:,i+x.shape[2]-1,1]=m
        return diag1,diag2
        
    

    
    
    def forward(self,x):
        
        x_h,x_w = self.coord(x)
        y = torch.cat([x_h,x_w],dim=2)
        y = self.conv_y(y)
        x_h,x_w = y.chunk(2,dim=2)
        x_h = self.conv_x_h(x_h)
        x_w = self.conv_x_w(x_w)
        x_w = x_w.permute(0,1,3,2)

        if SFP == 0:
            return x*torch.sigmoid(x_h.squeeze().unsqueeze(3).expand_as(x)*x_w.squeeze().unsqueeze(2).expand_as(x))

        z = self.sfp(y)
        
        return x*torch.sigmoid(x_h.squeeze().unsqueeze(3).expand_as(x)*x_w.squeeze().unsqueeze(2).expand_as(x)) + x*z.unsqueeze(2).unsqueeze(3).expand_as(x)
        





        if DIAG == 0:
            return x*torch.sigmoid(x_h.squeeze().unsqueeze(3).expand_as(x)*x_w.squeeze().unsqueeze(2).expand_as(x))
        
        diag1,diag2 = self.diag(x)       
        diag = torch.cat([diag1,diag2],dim=2)
        diag = diag.cuda()
        diag = self.conv_diag(diag)
        diag1,diag2 = diag.chunk(2,dim=2)     
        diag1 = self.conv_diag1(diag1)
        diag2 = self.conv_diag2(diag2)       
        weight_diag = self.diagback(x,diag1,diag2)
        
        return x*torch.sigmoid(weight_diag*x_h.squeeze().unsqueeze(3).expand_as(x)*x_w.squeeze().unsqueeze(2).expand_as(x))

        





        
        
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        