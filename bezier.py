#!/bin/python3

import argparse
import torch
import numpy as np
from torch.autograd import Variable
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='')
parser.add_argument('--display', action='store_true', help='')

args = parser.parse_args()

use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device "{}".'.format(device))
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

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

    def raster(self, curve, sigma=1e-2, n=None):
        raster = np.zeros((self.res, self.res))
        x = curve[0]
        y = curve[1]
        
        tic = time()
        
        spread = 2 * sigma
        # nextpow2 above 2 standard deviations in both x and y
        w = 2*int(2**np.ceil(np.log2(self.res*spread)))
        w = 32
        print(w)
        # lower left corner of a w*w block centered on each point of the curve
        blocks = torch.clamp((self.res * curve).floor().int() - w // 2, 0,  self.res - w)

        #  blocks = []
        #  mask = torch.zeros([self.res, self.res]).byte()
        #  # non overlapping blocks instead
        #  for point in torch.t(curve):
            #  x, y = torch.clamp((self.res * point).floor().int() - w // 2, 0,  self.res - w)
            
            #  mask[x:x+w, y:y+w] = 1
            #  blocks.append([x, y])
        
        # chunked
        # xmax, ymax = (self.res * (curve + spread)).ceil().int()
        # xlim = torch.stack([xmin, xmax], 1)
        # ylim = torch.stack([ymin, ymax], 1)
        # print(x.size())
        #  for point in torch.t(curve)[::]:
            #  xmax, ymax = (self.res * (point + spread)).ceil().int().tolist()
            #  xmin, ymin = (self.res * (point - spread)).floor().int().tolist()
            #  chunks.append([xmin, xmax, ymin, ymax])
        #  xmax, ymax = [(self.res * (i.max() + 3*sigma)).ceil().int().item() for i in (x, y)]
        #  xmin, ymin = [(self.res * (i.min() - 3*sigma)).floor().int().item() for i in (x, y)]
        
        

        c = torch.stack([self.c[x:x+w, y:y+w, t] for t, (x, y) in enumerate(torch.t(blocks))], dim=2)
        d = torch.stack([self.d[x:x+w, y:y+w, t] for t, (x, y) in enumerate(torch.t(blocks))], dim=2)
        print(time() - tic)
        x_ = x.expand(w, w, self.steps)
        y_ = y.expand(w, w, self.steps)
        raster = torch.zeros([self.res, self.res, self.steps])
        raster_ = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2*sigma**2))
        print(time() - tic)
        for t, (x, y) in enumerate(torch.t(blocks)):
            raster[x:x+w, y:y+w, t] = raster_[:,:,t]
        raster = torch.mean(raster, dim=2)
        
        #  for xmin, xmax, ymin, ymax in segments:
            #  w = xmax-xmin
            #  h = ymax-ymin
            #  print(w, h)
            #  x_ = x.expand(w, h, self.steps)
            #  y_ = y.expand(w, h, self.steps)
            #  #  x_ = x.expand(self.res, self.res, self.steps)
            #  #  y_ = y.expand(self.res, self.res, self.steps)
            #  # this is the slow part
            #  c = self.c[xmin:xmax, ymin:ymax]
            #  d = self.d[xmin:xmax, ymin:ymax]
            #  raster_ = torch.exp((-(x_ - c)**2 - (y_ - d) ** 2) / (2*sigma**2))
            #  raster_ = torch.mean(raster_, dim=2)
            #  raster[xmin:xmax, ymin:ymax] = raster_
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

control_points_t = Variable(torch.Tensor(np.array(control_points_l), device=device), requires_grad=True)

tic = time()
curve = net.forward(control_points_t)
print(time() - tic)

crit = torch.nn.L1Loss()
loss = crit(curve, Variable(torch.Tensor(curve.data)))
loss.backward()

curve_ = curve.data.numpy()

if args.display:
    import matplotlib.pyplot as plt
    plt.matshow(curve_)
    plt.show()
