#!/bin/python3

import torch
import numpy as np
from torch.autograd import Variable
from time import time

class Bezier(torch.nn.Module):
    def __init__(self, res=512, steps=100):
        super(Bezier, self).__init__()
        self.res = res
        self.steps = steps

        C, D = np.meshgrid(range(self.res), range(self.res))
        C_e = C[np.newaxis, :, :]
        D_e = D[np.newaxis, :, :]
        
        c = Variable(torch.Tensor(C_e / res)).expand(self.steps, self.res, self.res)
        d = Variable(torch.Tensor(D_e / res)).expand(self.steps, self.res, self.res)

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

    def raster(self, curve, sigma=1e-2):
        raster = np.zeros((self.res, self.res))
        x = curve[0]
        y = curve[1]
        xmax, ymax = [(self.res * (i.max() + 3*sigma)).ceil().int().item() for i in (x, y)]
        xmin, ymin = [(self.res * (i.min() - 3*sigma)).floor().int().item() for i in (x, y)]
        print(xmin, ymin, xmax, ymax)
        w = xmax-xmin
        h = ymax-ymin
        print(w, h)
        x_ = x.expand(w, h, self.steps)
        y_ = y.expand(w, h, self.steps)
        #  x_ = x.expand(self.res, self.res, self.steps)
        #  y_ = y.expand(self.res, self.res, self.steps)
        tic = time()
        # this is the slow part
        c = self.c[xmin:xmax, ymin:ymax]
        d = self.d[xmin:xmax, ymin:ymax]
        # raster_ = (x_ - c)**2 + (y_ - d) ** 2 < 1e-3
        # print(np.amax(raster_))
        # raster_ = torch.max(raster_, dim=2)[0]
        raster_ = torch.exp((-(x_ - c)**2 - (y_ - d) ** 2) / (2*sigma**2))
        raster_ = torch.mean(raster_, dim=2)
        raster = torch.zeros([self.res, self.res], dtype=torch.float)
        raster[xmin:xmax, ymin:ymax] = raster_
        print(x_)
        print(self.c)
        print(y_)
        print(self.d)
        print(raster)
        print(time() - tic)
        
        return torch.squeeze(raster)
      
    def forward(self, control_points):
        a = self.lin_interp(control_points[0], control_points[1], self.steps)
        b = self.lin_interp(control_points[1], control_points[2], self.steps)
        steps = Variable(torch.arange(0, self.steps).expand(2, self.steps))
        curve = a + (steps.float() / float(self.steps)) * (b - a)

        return self.raster(curve)

net = Bezier()

control_points_l = [
    [0.1, 0.1],
    [0.9, 0.9],
    [0.5, 0.9]
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
