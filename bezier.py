#!/bin/python3

import argparse
import torch
import numpy as np
from torch.autograd import Variable
from time import time

class Bezier(torch.nn.Module):
    def __init__(self, res=512, steps=128, method='base'):
        super(Bezier, self).__init__()
        self.res = res
        self.steps = steps
        if method == 'base':
            self.raster = self._raster_base
            self.lin_interp = self.lin_interp_base
        elif method == 'shrunk':
            self.raster = self._raster_shrunk
            self.lin_interp = self.lin_interp_fast

        C, D = np.meshgrid(range(self.res), range(self.res))
        C_e = C[np.newaxis, :, :]
        D_e = D[np.newaxis, :, :]
        
        c = Variable(torch.Tensor(C_e / res)).expand(self.steps, self.res, self.res)
        d = Variable(torch.Tensor(D_e / res)).expand(self.steps, self.res, self.res)

        self.c = torch.transpose(c, 0, 2)
        self.d = torch.transpose(d, 0, 2)        
        #  if use_cuda:
            #  torch.cuda.synchronize()
        
    @staticmethod
    def lin_interp_base(point1, point2, num_steps, steps):
          a = point1[0].expand(num_steps)
          b = point1[1].expand(num_steps)
          a_= point2[0].expand(num_steps)
          b_ = point2[1].expand(num_steps)
          
          t = Variable(torch.linspace(0, 1, num_steps))
        
          interp1 = a + (a_ - a) * t
          interp2 = b + (b_ - b) * t
        
          return torch.stack([interp1, interp2])

    @staticmethod
    def lin_interp_fast(point1, point2, num_steps,steps):
        #t = Variable(torch.linspace(0, 1, num_steps))
        interp1 = point1[0] + (point2[0] - point1[0]) * steps
        interp2 = point1[1] + (point2[1] - point1[1]) * steps
        return torch.stack([interp1, interp2])

    def forward(self, control_points):
#        steps = Variable(torch.arange(0, self.steps).expand(2, self.steps))
#        steps_ = steps.float()/float(self.steps)
        steps_ = Variable(torch.linspace(0, 1, self.steps))
        a = self.quadforward(control_points[0:3,:],steps_)
        if control_points.size()[0] == 4:
            b = self.quadforward(control_points[1:4,:],steps_)
            return self.raster(a + steps_ * (b-a))
        return self.raster(a)

    def quadforward(self,control_points,steps):    
        a = self.lin_interp(control_points[0], control_points[1], self.steps,steps)
        b = self.lin_interp(control_points[1], control_points[2], self.steps,steps)
        return a + steps * (b - a)
    
    def _raster_base(self, curve, sigma=1e-3):
        x = curve[0]
        y = curve[1]
        x_ = x.expand(self.res, self.res, 100)
        y_ = y.expand(self.res, self.res, 100)
        tic = time()
        raster = torch.exp(-(x_ - self.c)**2 / 2e-4 - (y_ - self.d) ** 2 / 2e-4)
        raster = torch.mean(raster, dim=2)
        if args.debug:
            print(time() - tic)
        
        return torch.squeeze(raster)
    
    def _raster_sparse(self, curve, sigma=1e-3):
        raster = np.zeros((self.res, self.res))
        x = curve[0]
        y = curve[1]
        
        tic = time()
        
        return torch.squeeze(raster)

    def _raster_wu(self, curve, sigma=1e-2):
        x = curve[0]
        y = curve[1]

        tic = time()
        
        spread = 2 * sigma
        # nextpow2 above 2 standard deviations in both x and y
        w = 2*int(2**np.ceil(np.log2(self.res*spread)))
        print(w)
        
        raster = torch.zeros([self.res, self.res])
        # raster_ = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2*sigma**2))
        # raster_ = (x_ - c)**2 + (y_ - d)**2
        for (x, y) in enumerate((self.res * curve).long().t()):
            print(x, y)
            raster[x, y] = 1
        
        print('{}: Rasterized.'.format(time() - tic))
        
        return torch.squeeze(raster)
    
    def _raster_smear(self, curve, sigma=1e-2):
        x = curve[0]
        y = curve[1]

        tic = time()
        
        spread = 2 * sigma
        # nextpow2 above 2 standard deviations in both x and y
        w = 2*int(2**np.ceil(np.log2(self.res*spread)))
        print(w)
        
        raster = torch.zeros([self.res, self.res])
        # raster_ = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2*sigma**2))
        # raster_ = (x_ - c)**2 + (y_ - d)**2
        for (x, y) in enumerate((self.res * curve).long().t()):
            print(x, y)
            raster[x, y] = 1
        
        print('{}: Rasterized.'.format(time() - tic))
        
        return torch.squeeze(raster)

    def _raster_shrunk(self, curve, sigma=1e-2):
        x = curve[0]
        y = curve[1]
        
        # torch.cuda.synchronize()

        tic = time()
        
        spread = 2 * sigma
        # nextpow2 above 2 standard deviations in both x and y
        w = 2*int(2**np.ceil(np.log2(self.res*spread)))
        if args.debug:
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
        
        # w * w * self.steps
        #  c = torch.zeros([w, w, self.steps])
        #  d = torch.zeros([w, w, self.steps])
        #  for t, (px, py) in enumerate(torch.t(blocks)):
            #  c[:,:,t] = self.c[px:px+w, py:py+w, t]
            #  d[:,:,t] = self.d[px:px+w, py:py+w, t]
        if args.debug:
            print('{}: Bounding rectangles found.'.format(time() - tic))
        c = torch.stack([self.c[px:px+w, py:py+w, t] for t, (px, py) in enumerate(torch.t(blocks))], dim=2)
        d = torch.stack([self.d[px:px+w, py:py+w, t] for t, (px, py) in enumerate(torch.t(blocks))], dim=2)
        if args.debug:
            print('{}: Bounding rectangles found.'.format(time() - tic))
        x_ = x.expand(w, w, self.steps)
        y_ = y.expand(w, w, self.steps)
        if args.debug:
            print('{}: Dims expanded.'.format(time() - tic))
        raster_ = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2*sigma**2))
        # raster_ = (x_ - c)**2 + (y_ - d)**2
        if args.debug:
            print('{}: Gradient generated.'.format(time() - tic))
        #  idx = torch.LongTensor
        #  self.r.scatter_(2, raster_)
        raster = torch.zeros([self.res, self.res], requires_grad=False)
        for t, (x, y) in enumerate(torch.t(blocks)):
            raster[x:x+w, y:y+w] += raster_[:,:,t]
        # raster = torch.mean(self.r, dim=2)
        
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
        if args.debug:
            print('{}: Rasterized.'.format(time() - tic))
        
        return torch.squeeze(raster)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='')
parser.add_argument('--display', action='store_true', help='')
parser.add_argument('--debug', action='store_true', help='')
parser.add_argument('--steps', default=100, type=int, help='')
parser.add_argument('--res', default=512, type=int, help='')
parser.add_argument('--method', default='base', help='')
parser.add_argument('--passes', default=1, type=int, help='')

args = parser.parse_args()

use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device "{}"'.format(device))
torch.set_default_tensor_type(torch.cuda.FloatTensor if use_cuda else torch.FloatTensor)

net = Bezier(res=args.res, steps=args.steps, method=args.method)

control_points_l = [
    [1, 0],
    [0.21, 0.12],
    [0.72, 0.83],
    [0,1]
    ]

control_points_t = Variable(torch.Tensor(np.array(control_points_l), device=device), requires_grad=True)

tic_total = time()
elapsed_fw, elapsed_bw = 0, 0

passes = args.passes
for i in range(passes):
    tic = time()
    curve = net.forward(control_points_t)
    elapsed_fw += time() - tic
    # print('{}: Total.'.format(time() - tic))

    tic = time()
    crit = torch.nn.L1Loss()
    loss = crit(curve, Variable(torch.Tensor(curve.data)))
    #  print('{}: Loss.'.format(time() - tic))
    loss.backward()
    elapsed_bw += time() - tic

    #  print('{}: Backwards.'.format(time() - tic))

elapsed = time() - tic_total
print('forwards:  {:4d} passes in {:7.3f} seconds [{:8.3f} iter/s {:>5.1f} ms/iter].'
        .format(passes, elapsed_fw, passes/elapsed_fw, elapsed_fw/passes * 1e3))
print('backwards: {:4d} passes in {:7.3f} seconds [{:8.3f} iter/s {:>5.1f} ms/iter].'
        .format(passes, elapsed_bw, passes/elapsed_bw, elapsed_bw/passes * 1e3))
print('total:     {:4d} passes in {:7.3f} seconds [{:8.3f} iter/s {:>5.1f} ms/iter].'
        .format(passes, elapsed, passes/elapsed, elapsed/passes * 1e3))

curve_ = curve.data.cpu().numpy()

if args.display:
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt
    plt.matshow(curve_)
    plt.show()
