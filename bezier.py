#!/bin/python3

import argparse
import torch
import numpy as np
from torch.autograd import Variable
from torch.multiprocessing import Pool
from time import time

class Bezier(torch.nn.Module):
    def __init__(self, res=512, steps=128, method='base'):
        super(Bezier, self).__init__()
        self.res = res
        self.steps = steps

        C, D = np.meshgrid(range(self.res), range(self.res))
        C_e = C[np.newaxis, :, :]
        D_e = D[np.newaxis, :, :]
        
        c = torch.Tensor(C_e / res).to(device).expand(self.steps, self.res, self.res)
        d = torch.Tensor(D_e / res).to(device).expand(self.steps, self.res, self.res)

        self.c = torch.transpose(c, 0, 2)
        self.d = torch.transpose(d, 0, 2)
        self.steps_t = (torch.arange(0, self.steps).float() / float(self.steps)).expand(2, self.steps).to(device)
        
        if method == 'base':
            self.raster = self._raster_base
        elif method == 'bounded':
            self.raster = self._raster_bounded
        elif method == 'bounded_tight':
            self.raster = self._raster_bounded_tight
        elif method == 'shrunk':
            self.raster = self._raster_shrunk
        elif method == 'tiled':
            # break in to NxN tiles
            self.tiles = 4
            self.chunksize = self.res // self.tiles

            #  print(self.c)
            #  self.c = self.c.reshape(-1, self.chunksize, self.chunksize, self.steps)
            #  self.d = self.d.reshape(-1, self.chunksize, self.chunksize, self.steps)
            # V100 has 80 sms
            self.streams = [torch.cuda.Stream() for i in range(self.tiles**2)]
            self.raster = self._raster_tiled
        elif method == 'reindex_tiled':
            # break in to NxN tiles
            self.tiles = 4
            self.chunksize = self.res // self.tiles

            self.c = self.c.reshape(-1, self.chunksize, self.chunksize, self.steps)
            self.d = self.d.reshape(-1, self.chunksize, self.chunksize, self.steps)
            self.streams = [torch.cuda.Stream() for i in range(self.tiles**2)]
            self.raster = self._raster_reindex_tiled

        #  if use_cuda:
            #  torch.cuda.synchronize()
        
    @staticmethod
    def lin_interp(point1, point2, num_steps):
          a = point1[0].expand(num_steps)
          b = point1[1].expand(num_steps)
          a_= point2[0].expand(num_steps)
          b_ = point2[1].expand(num_steps)
          
          t = torch.linspace(0, 1, num_steps)
        
          interp1 = a + (a_ - a) * t
          interp2 = b + (b_ - b) * t
        
          return torch.stack([interp1, interp2])

    def forward(self, control_points):
        # NCHW
        a = self.lin_interp(control_points[0], control_points[1], self.steps)
        b = self.lin_interp(control_points[1], control_points[2], self.steps)
        #  curve = a + (self.steps_t) * (b - a)
        steps = Variable(torch.arange(0, self.steps).expand(2, self.steps))
        curve = a + (steps.float() / float(self.steps)) * (b - a)
        return self.raster(curve)
    
    def _raster_base(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]

        x_ = x.to(device).expand(self.res, self.res, self.steps)
        y_ = y.to(device).expand(self.res, self.res, self.steps)
        if args.debug:
            print(time() - tic)
        
        raster = torch.exp((-(x_ - self.c)**2 - (y_ - self.d)**2) / (2*sigma**2))
        raster = torch.mean(raster, dim=2)
        if args.debug:
            print(time() - tic)
        
        return torch.transpose(torch.squeeze(raster), 0, 1)
    
    def _raster_linear(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]

        x_ = x.expand(self.res, self.res, self.steps).to(device)
        y_ = y.expand(self.res, self.res, self.steps).to(device)
        
        raster = torch.exp((-(x_ - self.c)**2 - (y_ - self.d) ** 2) / (2*sigma**2))
        raster = torch.mean(raster, dim=2)
        if args.debug:
            print(time() - tic)
        
        return torch.transpose(torch.squeeze(raster), 0, 1)

    def _raster_bounded(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]

        xmax = (self.res * (x.max() + 3*sigma)).ceil().int().item()
        ymax = (self.res * (y.max() + 3*sigma)).ceil().int().item()
        xmin = (self.res * (x.min() - 3*sigma)).floor().int().item()
        ymin = (self.res * (y.min() - 3*sigma)).floor().int().item()

        #  x_ind = torch.arange((self.res * (x.min() - 3*sigma)).floor(), (self.res * (x.max() + 3*sigma)).ceil()).long()
        #  y_ind = torch.arange((self.res * (y.min() - 3*sigma)).floor(), (self.res * (y.max() + 3*sigma)).ceil()).long()
        if args.debug:
            print(time() - tic)

        
        w = xmax-xmin
        h = ymax-ymin
        
        x_ = x.to(device).expand(w, h, self.steps)
        y_ = y.to(device).expand(w, h, self.steps)
        
        
        if args.debug:
            print(time() - tic)
        c = self.c[xmin:xmax, ymin:ymax]
        d = self.d[xmin:xmax, ymin:ymax]

        if args.debug:
            print(time() - tic)
        raster_ = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2*sigma**2))
        raster_ = torch.mean(raster_, dim=2)
        if args.debug:
            print(time() - tic)
        raster = torch.zeros([self.res, self.res]).to(device)
        raster[xmin:xmax, ymin:ymax] = raster_
        if args.debug:
            print(time() - tic)

        return torch.transpose(torch.squeeze(raster), 0, 1)
    
    def _raster_tiled(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]

        raster = torch.zeros([self.res, self.res]).to(device)
        
        r = x + 3*sigma
        l = x - 3*sigma
        b = y - 3*sigma
        t = y - 3*sigma
        
        x_ = x.to(device).expand(self.res, self.res, self.steps)
        y_ = y.to(device).expand(self.res, self.res, self.steps)
        #  x_ = x.to(device).expand(self.res, self.res, self.steps).reshape(-1, self.chunksize, self.chunksize, self.steps)
        #  y_ = y.to(device).expand(self.res, self.res, self.steps).reshape(-1, self.chunksize, self.chunksize, self.steps)
        
        torch.cuda.synchronize()
        for tile, stream in enumerate(self.streams):
            with torch.cuda.stream(stream):
                y_tile, x_tile = divmod(tile, self.tiles)
                x_idx = self.chunksize * x_tile
                y_idx = self.chunksize * y_tile
                xi = x_[x_idx:x_idx+self.chunksize, y_idx:y_idx+self.chunksize, :]
                yi = y_[x_idx:x_idx+self.chunksize, y_idx:y_idx+self.chunksize, :]
                ci = self.c[x_idx:x_idx+self.chunksize, y_idx:y_idx+self.chunksize, :]
                di = self.d[x_idx:x_idx+self.chunksize, y_idx:y_idx+self.chunksize, :]
                raster_ = torch.exp((-(xi - ci)**2 - (yi - di)**2) / (2*sigma**2))
                raster_ = torch.mean(raster_, dim=2)
                raster[x_idx:x_idx+self.chunksize, y_idx:y_idx+self.chunksize] = raster_
        torch.cuda.synchronize()

        #  torch.multiprocessing.set_start_method('spawn')
        #  pool = Pool(N**2)
        #  pool.map(stream_fn, range(N**2))
        #  print(c_.size())
        
        if args.debug:
            print(time() - tic)
        
        #  x_ = x.to(device).expand(self.res, self.res, self.steps)
        #  y_ = y.to(device).expand(self.res, self.res, self.steps)
        #  raster = torch.exp((-(x_ - self.c)**2 - (y_ - self.d)**2) / (2*sigma**2))
        #  raster = torch.mean(raster, dim=2)
        
        return torch.transpose(torch.squeeze(raster), 0, 1)
    
    def _raster_reindex_tiled(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]

        raster = torch.zeros([self.res, self.res]).to(device)
        
        r = x + 3*sigma
        l = x - 3*sigma
        b = y - 3*sigma
        t = y - 3*sigma
        
        x_ = x.to(device).expand(self.res, self.res, self.steps).reshape(-1, self.chunksize, self.chunksize, self.steps)
        y_ = y.to(device).expand(self.res, self.res, self.steps).reshape(-1, self.chunksize, self.chunksize, self.steps)
        
        torch.cuda.synchronize()
        for tile, stream in enumerate(self.streams[:2]):
            with torch.cuda.stream(stream):
                y_tile, x_tile = divmod(tile, self.tiles)
                x_idx = self.chunksize * x_tile
                y_idx = self.chunksize * y_tile
                raster_ = torch.exp((-(x_[tile] - self.c[tile])**2 - (y_[tile] - self.d[tile])**2) / (2*sigma**2))
                raster_ = torch.mean(raster_, dim=2)
                raster[x_idx:x_idx+self.chunksize, y_idx:y_idx+self.chunksize] = raster_
        torch.cuda.synchronize()
        
        return torch.transpose(torch.squeeze(raster), 0, 1)
        
    def _raster_bounded_tight(self, curve, sigma=1e-2):
        tic = time()
        print(curve) 
        # align start and end points
        theta = torch.atan(curve[1, -1] / curve[0, -1])
        print(theta)
        R = torch.Tensor([[theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]]).to(device)
        
        T = curve[:, 0].expand(self.steps, 2).transpose(0, 1)
        curve -= T
        print(curve)
        curve = R.matmul(curve)
        print(R)
        print(curve)
        x = curve[0]
        y = curve[1]
        xmax, ymax = [(self.res * (i.max() + 3*sigma)).ceil().int().item() for i in (x, y)]
        xmin, ymin = [(self.res * (i.min() - 3*sigma)).floor().int().item() for i in (x, y)]
        w = xmax-xmin
        h = ymax-ymin
        print(xmin, xmax)
        print(ymin, ymax)
        x_ = x.expand(w, h, self.steps)
        y_ = y.expand(w, h, self.steps)
        c = self.c[self.res+xmin:self.res+xmax, self.res+ymin:self.res+ymax]
        d = self.d[self.res+xmin:self.res+xmax, self.res+ymin:self.res+ymax]
        print(c)
        print(d)
        raster_ = torch.exp((-(x_ - c)**2 - (y_ - d) ** 2) / (2*sigma**2))
        raster_ = torch.mean(raster_, dim=2)
        raster = torch.zeros([2*self.res, 2*self.res])
        raster[self.res+xmin:self.res+xmax, self.res+ymin:self.res+ymax] = raster_

        if args.debug:
            print(time() - tic)

        return torch.transpose(torch.squeeze(raster), 0, 1)


    def _raster_shrunk(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]
        
        raster = torch.zeros([self.res, self.res], requires_grad=False).to(device)
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
        c = torch.stack([self.c[px:px+w, py:py+w, t] for t, (px, py) in enumerate(torch.t(blocks))], dim=2).to(device)
        d = torch.stack([self.d[px:px+w, py:py+w, t] for t, (px, py) in enumerate(torch.t(blocks))], dim=2).to(device)
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
parser.add_argument('--disable-cuda', action='store_true', help='')
parser.add_argument('--display', action='store_true', help='')
parser.add_argument('--debug', action='store_true', help='')
parser.add_argument('--steps', default=128, type=int, help='')
parser.add_argument('--res', default=512, type=int, help='')
parser.add_argument('--method', default='base', help='')
parser.add_argument('--passes', default=1, type=int, help='')

args = parser.parse_args()

use_cuda = not args.disable_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device "{}"'.format(device))
# torch.set_default_tensor_type(torch.cuda.FloatTensor if use_cuda else torch.FloatTensor)

net = Bezier(res=args.res, steps=args.steps, method=args.method)

control_points_l = [
    [0.1, 0.1],
    [0.9, 0.9],
    [0.5, 0.9]
    ]

batch_size = 32
control_points_batch = [control_points_l for i in range(batch_size)]
#  control_points_batch = control_points_l * batch_size
# print(torch.Tensor(np.array(control_points_batch)).size())

control_points_t = Variable(torch.Tensor(np.array(control_points_l)), requires_grad=True)

elapsed_fw, elapsed_bw = 0, 0

passes = args.passes
crit = torch.nn.L1Loss().cuda()
tic_total = time()
for i in range(passes):
    tic = time()
    curve = net.forward(control_points_t)
    elapsed_fw += time() - tic
    #  print('{}: Total.'.format(time() - tic))
    loss = crit(curve, torch.tensor(curve))
    tic = time()
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
