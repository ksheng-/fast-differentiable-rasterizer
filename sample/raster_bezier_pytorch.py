#!/bin/python3

import torch
import numpy as np
from torch.autograd import Variable
from time import time

class Bezier(torch.nn.Module):
    def __init__(self, res=512):
        super(Bezier, self).__init__()
        self.res = res

        C, D = np.meshgrid(range(self.res), range(self.res))
        C_e = C[np.newaxis, :, :]
        D_e = D[np.newaxis, :, :]
        
        c = Variable(torch.Tensor(C_e / res)).expand(100, 512, 512)
        d = Variable(torch.Tensor(D_e / res)).expand(100, 512, 512)

        self.c = torch.transpose(c, 0, 2)
        self.d = torch.transpose(d, 0, 2)        
        
    @staticmethod
    def lin_interp(point1, point2, num_steps):
          a = point1[0].expand(num_steps)
          b = point1[1].expand(num_steps)
          a_= point2[0].expand(num_steps)
          b_ = point2[1].expand(num_steps)
          
          t = Variable(torch.linspace(0, 1, num_steps))
        
          interp1 = a + (a_ - a) * t
          interp2 = b + (b_ - b) * t
        
          return torch.stack([interp1, interp2])

    def raster(self, curve):
        raster = np.zeros((self.res, self.res))
        x = curve[0]
        y = curve[1]
        x_ = x.expand(self.res, self.res, 100)
        y_ = y.expand(self.res, self.res, 100)
        tic = time()
        raster = torch.exp(-(x_ - self.c)**2 / 2e-4 - (y_ - self.d) ** 2 / 2e-4)
        raster = torch.mean(raster, dim=2)
        print(time() - tic)
        

        return torch.squeeze(raster)
      
    def forward(self, control_points):
        n_steps = 100
        steps = Variable(torch.arange(0, n_steps).expand(2, n_steps))
        a = self.quadforward(control_points[0:3,:],steps)
        if control_points.size()[0] == 4:
            b = self.quadforward(control_points[1:4,:],steps)
            curve = a + (steps.float()/float(n_steps))*(b-a)
            return self.raster(curve)
        return self.raster(a)

    def quadforward(self,control_points,steps):
        n_steps = 100
        a = self.lin_interp(control_points[0], control_points[1], n_steps)
        b = self.lin_interp(control_points[1], control_points[2], n_steps)
        return a + (steps.float() / float(n_steps)) * (b - a)

net = Bezier()

control_points_l = [
    [1.0,0.0],
    [0.21,0.12],
    [0.72,0.83],
    [0.0,1.0]
    ]

control_points_t = Variable(torch.Tensor(np.array(control_points_l)), requires_grad=True)

tic = time()
curve = net.forward(control_points_t)
print(time() - tic)

crit = torch.nn.L1Loss()
loss = crit(curve, Variable(torch.Tensor(curve.data)))
loss.backward()

curve_ = curve.data.numpy()

import matplotlib.pyplot as plt
plt.matshow(curve_)

plt.show()
