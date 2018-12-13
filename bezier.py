#!/bin/python3

import argparse
import torch
import numpy as np
from torch.autograd import Variable
from torch.multiprocessing import Pool
from time import time

class Bezier(torch.nn.Module):
    def __init__(self, res=512, steps=128, method='base', device='cpu', debug=False):
        super(Bezier, self).__init__()
        self.res = res
        self.steps = steps
        self.method = method
        self.device = device
        self.debug = debug

        self.cpu = torch.empty(2)
        self.gpu = torch.empty(2, device=device)
        self.gpu_half = torch.empty(2, device=device).half()
        C, D = np.meshgrid(range(self.res), range(self.res))
        C_e = C[np.newaxis, :, :]
        D_e = D[np.newaxis, :, :]
         
        self.c = torch.Tensor(C_e / res).to(self.device)
        self.d = torch.Tensor(D_e / res).to(self.device)

        if method == 'base':
            self.raster = self._raster_base
        if method == 'half':
            self.raster = self._raster_half
        elif method == 'bounded':
            self.raster = self._raster_bounded
        elif method == 'bounded_tight':
            self.raster = self._raster_bounded_tight
        elif method == 'shrunk':
            self.raster = self._raster_shrunk
        elif method == 'tiled':
            # break in to NxN tiles
            self.tiles = 4
            self.tiles_t = torch.Tensor([self.tiles]).long().to(device)
            self.chunksize = (self.res // self.tiles)

            # V100 has 80 sms
            self.streams = [torch.cuda.Stream() for i in range(self.tiles**2)]
            self.raster = self._raster_tiled

        if device == 'cuda':
            torch.cuda.synchronize()
        
    @staticmethod
    def _lin_interp_base(point1, point2, num_steps):
        a = point1[0].expand(num_steps)
        b = point1[1].expand(num_steps)
        a_= point2[0].expand(num_steps)
        b_ = point2[1].expand(num_steps)

        t = torch.linspace(0, 1, num_steps)
        interp1 = a + (a_ - a) * t
        interp2 = b + (b_ - b) * t

        return torch.stack([interp1, interp2])

    @staticmethod
    def _lin_interp_broadcast(point1, point2, num_steps):
        t = torch.linspace(0, 1, num_steps)
        interp1 = point1[0] + (point2[0] - point1[0]) * t
        interp2 = point1[1] + (point2[1] - point1[1]) * t
        return torch.stack([interp1, interp2])
    
    def lerp2d(self, a, b, steps):
        x = a[0] + (b[0] - a[0]) * steps
        y = a[1] + (b[1] - a[1]) * steps
        return torch.stack([x, y])

    def quadratic_interp(self, control_points, steps):  
        a = self.lerp2d(control_points[0], control_points[1], steps)
        b = self.lerp2d(control_points[1], control_points[2], steps)
        return a + steps * (b - a)

    def sample_curve(self, control_points, steps):
        if control_points.size()[0] == 3:
            curve = self.quadratic_interp(control_points[:3], steps)
        if control_points.size()[0] == 4:
            a = self.quadratic_interp(control_points[:3], steps)
            b = self.quadratic_interp(control_points[1:4], steps)
            curve = a + steps * (b - a)
        return curve
    
    def forward(self, curves):
        # NCHW
        steps = torch.linspace(0, 1, self.steps)
        curve = torch.cat([self.sample_curve(c, steps) for c in curves], dim=1)
        #  if curves.size()[1] == 3:
            #  curve = torch.cat([self.quadratic_interp(control_points[:3], steps) for c in curves]
        #  if curves.size()[1] == 4:
            #  a = self.quadratic_interp(control_points[:3], steps)
            #  b = self.quadratic_interp(control_points[1:4], steps)
            #  curve = a + steps * (b - a)
        return self.raster(curve)
    
    def _raster_base(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]
        
        steps = curve.size()[1]
        x_ = x.to(self.device).expand(self.res, self.res, steps)
        y_ = y.to(self.device).expand(self.res, self.res, steps)
        c = torch.transpose(self.c.expand(steps, self.res, self.res), 0, 2)
        d = torch.transpose(self.d.expand(steps, self.res, self.res), 0, 2)
        
        if self.debug:
            print(time() - tic)
        
        raster = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2*sigma**2))
        #  raster = torch.mean(raster, dim=2)
        # raster = torch.min(torch.sum(raster, dim=2), torch.Tensor([1]).to(self.device))
        raster = torch.sum(raster, dim=2)
        #  raster = torch.max(raster, dim=2)[0]
        if self.debug:
            print(time() - tic)
        
        return torch.transpose(torch.squeeze(raster), 0, 1)
    
    def _raster_half(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]

        steps = curve.size()[1]
        x_ = x.to(self.device).half()
        y_ = y.to(self.device).half()
        c = torch.transpose(self.c.expand(steps, self.res, self.res), 0, 2).half()
        d = torch.transpose(self.d.expand(steps, self.res, self.res), 0, 2).half()
        #  c = torch.transpose(self.c.cpu().repeat(steps, 1, 1), 0, 2).half().to(self.device)
        #  d = torch.transpose(self.d.cpu().repeat(steps, 1, 1), 0, 2).half().to(self.device)
        
        if self.debug:
            print(time() - tic)
        
        raster = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2*sigma**2))
        #  raster = torch.mean(raster, dim=2)
        raster = torch.sum(raster, dim=2)
        if self.debug:
            print(time() - tic)
        
        return torch.transpose(torch.squeeze(raster.float()), 0, 1)
    
    def _raster_linear(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]

        x_ = x.expand(self.res, self.res, self.steps).to(self.device)
        y_ = y.expand(self.res, self.res, self.steps).to(self.device)
        
        raster = torch.exp((-(x_ - c)**2 - (y_ - self.d) ** 2) / (2*sigma**2))
        #  raster = torch.mean(raster, dim=2)
        raster = torch.sum(raster, dim=2)
        if self.debug:
            print(time() - tic)
        
        return torch.transpose(torch.squeeze(raster), 0, 1)

    def _raster_bounded(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]

        steps = curve.size()[1]
        
        xmax = torch.clamp((self.res * (x.max() + 3*sigma)).ceil(), 0, self.res).int().item()
        ymax = torch.clamp((self.res * (y.max() + 3*sigma)).ceil(), 0, self.res).int().item()
        xmin = torch.clamp((self.res * (x.min() - 3*sigma)).floor(), 0, self.res).int().item()
        ymin = torch.clamp((self.res * (y.min() - 3*sigma)).floor(), 0, self.res).int().item()

        #  x_ind = torch.arange((self.res * (x.min() - 3*sigma)).floor(), (self.res * (x.max() + 3*sigma)).ceil()).long()
        #  y_ind = torch.arange((self.res * (y.min() - 3*sigma)).floor(), (self.res * (y.max() + 3*sigma)).ceil()).long()
        if self.debug:
            print(time() - tic)

        
        w = xmax-xmin
        h = ymax-ymin
        
        x_ = x.to(self.device).half()
        y_ = y.to(self.device).half()
        
        if self.debug:
            print(time() - tic)
        c = torch.transpose(self.c.half().expand(steps, self.res, self.res), 0, 2)[xmin:xmax, ymin:ymax]
        d = torch.transpose(self.d.half().expand(steps, self.res, self.res), 0, 2)[xmin:xmax, ymin:ymax]
        if self.debug:
            print(time() - tic)
        raster_ = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2*sigma**2))
        #  raster_ = torch.mean(raster_, dim=2)
        raster_ = torch.sum(raster_, dim=2)
        if self.debug:
            print(time() - tic)
        raster = torch.zeros([self.res, self.res]).to(self.device)
        raster[xmin:xmax, ymin:ymax] = raster_
        if self.debug:
            print(time() - tic)

        return torch.transpose(torch.squeeze(raster.float()), 0, 1)
    
    def _raster_tiled(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]

        #  raster = torch.zeros([self.res, self.res]).half().to(self.device)
        raster = self.gpu.new_zeros(self.res, self.res) 
        steps = curve.size()[1]
        curve.to(self.device)
        if self.debug:
            print('tiles 2 {}'.format(time() - tic))
        tiles = self.gpu.new_zeros(steps, steps)
        #  tiles = torch.cuda.FloatTensor(steps, steps).fill_(0)
        #  tiles = torch.zeros([steps, steps]).to(self.device, non_blocking=True)
        if self.debug:
            print('tiles eye {}'.format(time() - tic))
        steps_ = torch.eye(steps, device=self.device)
        if self.debug:
            print('tiles eye {}'.format(time() - tic))
        x_ = x.to(self.device, non_blocking=True)
        y_ = y.to(self.device, non_blocking=True)
        if self.debug:
            print('tiles 1 {}'.format(time() - tic))
        c = self.c.expand(steps, self.res, self.res)
        if self.debug:
            print('tiles 2 {}'.format(time() - tic))
        c = torch.transpose(c, 0, 2)
        if self.debug:
            print('tiles 3 {}'.format(time() - tic))
        d = torch.transpose(self.d.expand(steps, self.res, self.res), 0, 2)
        if self.debug:
            print('tiles 1 {}'.format(time() - tic))
        
        #  x_ = x.to(self.device).expand(self.res, self.res, steps)
        #  y_ = y.to(self.device).expand(self.res, self.res, steps)
        #  #  x_ = x.to(self.device).expand(self.res, self.res, steps).reshape(-1, self.chunksize, self.chunksize, steps)
        #  #  y_ = y.to(self.device).expand(self.res, self.res, steps).reshape(-1, self.chunksize, self.chunksize, steps)
        bound = int(self.res * 3 * sigma)

        if self.debug:
            print('tiles 1 {}'.format(time() - tic))
        if self.debug:
            print('tiles 2 {}'.format(time() - tic))
        curve_px = (curve * self.res).long().to(self.device)
        x_px, y_px = curve_px[0], curve_px[1]
        curve_tile = torch.min((curve_px / self.chunksize), self.tiles_t-1)
        x_tile, y_tile = curve_tile[0], curve_tile[1]


        if self.debug:
            print('tiles 2 {}'.format(time() - tic))
        center_tiles = (self.tiles * y_tile + x_tile).long()
        right_tiles = ((x_tile < self.tiles - 1) & (x_px + bound >= (x_tile + 1) * self.chunksize)).long()
        left_tiles = ((x_tile > 0) & (x_px - bound < x_tile * self.chunksize)).long()
        bottom_tiles = ((y_tile < self.tiles - 1) & (y_px + bound >= (y_tile + 1) * self.chunksize)).long() * self.tiles
        top_tiles = ((y_tile > 0) & (y_px - bound < y_tile * self.chunksize)).long() * self.tiles
        
        if self.debug:
            #  torch.cuda.synchronize()
            print('tiles 3 {}'.format(time() - tic))
        # the index is a 1d tensors of length steps containing the index of the tile the step falls into
        tiles = tiles.index_add_(0, center_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles + right_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles - left_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles + bottom_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles - top_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles + right_tiles + bottom_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles - left_tiles + bottom_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles + right_tiles - top_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles - left_tiles - top_tiles, steps_)

        if self.debug:
            #  torch.cuda.synchronize()
            print('tiles 4 {}'.format(time() - tic))
        # these are arrays of length steps that define another tile that the point at that step overlaps into 
        #  right_tiles = tiles + ((x_tile < self.tiles - 1) & (x_px + bound >= (x_tile + 1) * self.chunksize)).long()
        #  left_tiles = tiles - ((x_tile > 0) & (x_px - bound < x_tile * self.chunksize)).long()
        #  bottom_tiles = tiles + ((y_tile < self.tiles - 1) & (y_px + bound >= (y_tile + 1) * self.chunksize)).long() * self.tiles
        #  top_tiles = tiles - ((y_tile > 0) & (y_px - bound < y_tile * self.chunksize)).long() * self.tiles
        #  br = tiles + ((x3_tile < self.tiles - 1) & (x_px + bound >= (x_tile + 1) * self.chunksize)).long() + ((y_tile < self.tiles - 1) & (y_px + bound >= (y_tile + 1) * self.chunksize)).long() * self.tiles
        #  x_tile_boundaries = torch.arange(self.tiles).repeat(self.tiles, 0)

        #  tiles = self.tiles * y_tile + x_tile - 1
        #  tiles = self.tiles * y_tile + x_tile - 1
        #  tiles = self.tiles * y_tile + x_tile + 1
        #  tiles = self.tiles * y_tile + x_tile + 1

        if self.debug:
            #  torch.cuda.synchronize()
            print('tiles {}'.format(time() - tic))
        #  for i, (x, y) in enumerate(torch.t(curve)):
            #  xp = int(x * self.res)
            #  yp = int(y * self.res)
            #  xt = min(self.tiles-1, xp // self.chunksize)
            #  yt = min(self.tiles-1, yp // self.chunksize)
            #  tiles[self.tiles * yt + xt].append(i)
            #  if xt < self.tiles - 1 and xp + bound >= (xt + 1) * self.chunksize:
                #  tiles[self.tiles * yt + xt + 1].append(i)
            #  if xt > 0 and xp - bound < xt * self.chunksize:
                #  tiles[self.tiles * yt + xt - 1].append(i)
            #  if yt < self.tiles - 1 and yp + bound >= (yt + 1) * self.chunksize:
                #  tiles[self.tiles * yt + self.tiles + xt].append(i)
            #  if yt > 0 and yp - bound < yt * self.chunksize:
                #  tiles[self.tiles * yt - self.tiles + xt].append(i)
        #  tile = 0     
        #  steps_in_tile = ((tiles == tile) | (right_tiles == tile) | (left_tiles == tile) | (top_tiles == tile) | (bottom_tiles == tile) | (br == tile)).nonzero().reshape(-1)

        for tile, stream in enumerate(self.streams):
            with torch.cuda.stream(stream):
                steps_in_tile = tiles[tile].nonzero().reshape(-1) 
                if self.debug:
                    #  torch.cuda.synchronize()
                    print('steps in tile {}'.format(time() - tic))
                #  steps_in_tile = ((tiles == tile) | (right_tiles == tile) | (left_tiles == tile) | (top_tiles == tile) | (bottom_tiles == tile) | (br == tile)).nonzero().reshape(-1)
                if steps_in_tile.size()[0] > 0:
                        y_tile, x_tile = divmod(tile, self.tiles)
                        if self.debug:
                            print('1 {}'.format(time() - tic))
                        x_idx = self.chunksize * x_tile
                        y_idx = self.chunksize * y_tile
                        if self.debug:
                            print('2 {}'.format(time() - tic))
                        ci = c[x_idx:x_idx+self.chunksize, y_idx:y_idx+self.chunksize, steps_in_tile]
                        di = d[x_idx:x_idx+self.chunksize, y_idx:y_idx+self.chunksize, steps_in_tile]
                        if self.debug:
                            print('3 {}'.format(time() - tic))
                        raster_ = torch.exp((-(x_[steps_in_tile] - ci)**2 - (y_[steps_in_tile] - di)**2) / (2*sigma**2))
                        if self.debug:
                            print('4 {}'.format(time() - tic))
                        raster_ = torch.sum(raster_, dim=2)
                        raster[x_idx:x_idx+self.chunksize, y_idx:y_idx+self.chunksize] = raster_
                        if self.debug:
                            print('5 {}'.format(time() - tic))
                if self.debug:
                    print('send off {}'.format(time() - tic))

        torch.cuda.synchronize()
        #  raster = torch.min(raster, torch.Tensor([1]).to(self.device).half())
        #  torch.multiprocessing.set_start_method('spawn')
        #  pool = Pool(N**2)
        #  pool.map(stream_fn, range(N**2))
        #  print(c_.size())
        
        if self.debug:
            print(time() - tic)
        
        #  x_ = x.to(self.device).expand(self.res, self.res, steps)
        #  y_ = y.to(self.device).expand(self.res, self.res, steps)
        #  raster = torch.exp((-(x_ - self.c)**2 - (y_ - self.d)**2) / (2*sigma**2))
        #  raster = torch.mean(raster, dim=2)
        
        return torch.transpose(torch.squeeze(raster.float()), 0, 1)
    
    def _raster_bounded_tight(self, curve, sigma=1e-2):
        tic = time()
        print(curve) 
        # align start and end points
        theta = torch.atan(curve[1, -1] / curve[0, -1])
        print(theta)
        R = torch.Tensor([[theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]]).to(self.device)
        
        T = curve[:, 0].expand(steps, 2).transpose(0, 1)
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
        x_ = x.expand(w, h, steps)
        y_ = y.expand(w, h, steps)
        c = self.c[self.res+xmin:self.res+xmax, self.res+ymin:self.res+ymax]
        d = self.d[self.res+xmin:self.res+xmax, self.res+ymin:self.res+ymax]
        print(c)
        print(d)
        raster_ = torch.exp((-(x_ - c)**2 - (y_ - d) ** 2) / (2*sigma**2))
        #  raster_ = torch.mean(raster_, dim=2)
        raster_ = torch.sum(raster_, dim=2)
        raster = torch.zeros([2*self.res, 2*self.res])
        raster[self.res+xmin:self.res+xmax, self.res+ymin:self.res+ymax] = raster_

        if self.debug:
            print(time() - tic)

        return torch.transpose(torch.squeeze(raster), 0, 1)


    def _raster_shrunk(self, curve, sigma=1e-2):
        tic = time()
        
        x = curve[0]
        y = curve[1]
        
        steps = curve.size()[1]
        
        raster = torch.zeros([self.res, self.res], requires_grad=False).to(self.device)
        spread = 2 * sigma
        # nextpow2 above 2 standard deviations in both x and y
        w = 2*int(2**np.ceil(np.log2(self.res*spread)))
        if self.debug:
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
        
        # w * w * steps
        #  c = torch.zeros([w, w, steps])
        #  d = torch.zeros([w, w, steps])
        #  for t, (px, py) in enumerate(torch.t(blocks)):
            #  c[:,:,t] = self.c[px:px+w, py:py+w, t]
            #  d[:,:,t] = self.d[px:px+w, py:py+w, t]
        if self.debug:
            print('{}: Bounding rectangles found.'.format(time() - tic))
        x_ = x.to(self.device).expand(w, w, steps)
        y_ = y.to(self.device).expand(w, w, steps)
        c_ = torch.transpose(self.c.expand(steps, self.res, self.res), 0, 2)
        d_ = torch.transpose(self.d.expand(steps, self.res, self.res), 0, 2)
        c = torch.stack([c_[px:px+w, py:py+w, t] for t, (px, py) in enumerate(torch.t(blocks))], dim=2).to(self.device)
        d = torch.stack([d_[px:px+w, py:py+w, t] for t, (px, py) in enumerate(torch.t(blocks))], dim=2).to(self.device)
        if self.debug:
            print('{}: Bounding rectangles found.'.format(time() - tic))
        if self.debug:
            print('{}: Dims expanded.'.format(time() - tic))
        raster_ = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2*sigma**2))
        # raster_ = (x_ - c)**2 + (y_ - d)**2
        if self.debug:
            print('{}: Gradient generated.'.format(time() - tic))
        #  idx = torch.LongTensor
        #  self.r.scatter_(2, raster_)
        for t, (x, y) in enumerate(torch.t(blocks)):
            raster[x:x+w, y:y+w] += raster_[:,:,t]
        # raster = torch.mean(self.r, dim=2)
        
        raster = torch.min(raster, torch.Tensor([1]).to(self.device))
        #  for xmin, xmax, ymin, ymax in segments:
            #  w = xmax-xmin
            #  h = ymax-ymin
            #  print(w, h)
            #  x_ = x.expand(w, h, steps)
            #  y_ = y.expand(w, h, steps)
            #  #  x_ = x.expand(self.res, self.res, steps)
            #  #  y_ = y.expand(self.res, self.res, steps)
            #  # this is the slow part
            #  c = self.c[xmin:xmax, ymin:ymax]
            #  d = self.d[xmin:xmax, ymin:ymax]
            #  raster_ = torch.exp((-(x_ - c)**2 - (y_ - d) ** 2) / (2*sigma**2))
            #  raster_ = torch.mean(raster_, dim=2)
            #  raster[xmin:xmax, ymin:ymax] = raster_
        if self.debug:
            print('{}: Rasterized.'.format(time() - tic))
        
        return torch.transpose(torch.squeeze(raster), 0, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-cuda', action='store_true', help='')
    parser.add_argument('--display', action='store_true', help='')
    parser.add_argument('--debug', action='store_true', help='')
    parser.add_argument('--cubic', action='store_true', help='')
    parser.add_argument('--steps', default=128, type=int, help='')
    parser.add_argument('--res', default=512, type=int, help='')
    parser.add_argument('--method', default='base', help='')
    parser.add_argument('--draw', default='quadratic', help='')
    parser.add_argument('--batch', default=1, type=int, help='')
    #  parser.add_argument('--batches', nargs='*', default=[16], type=int, help='')
    parser.add_argument('--passes', default=1, type=int, help='')

    args = parser.parse_args()

    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device "{}"'.format(device))
    #  torch.set_default_tensor_type(torch.cuda.FloatTensor if use_cuda else torch.FloatTensor)

    net = Bezier(res=args.res, steps=args.steps, method=args.method, device=device)

    if args.draw == 'quadratic':
        control_points_l = [[
            [0.1, 0.1],
            [0.9, 0.9],
            [0.5, 0.9]
        ]]
    elif args.draw == 'cubic':
        control_points_l = [[
            [1.0, 0.0],
            [0.21, 0.12],
            [0.72, 0.83],
            [0.0, 1.0]
        ]]

    elif args.draw == 'char':
        #  <point x="166" y="1456" type="line"/>
        #  <point x="166" y="0" type="line"/>
        #  <point x="374" y="0" type="line"/>
        #  <point x="374" y="1289" type="line"/>
        #  <point x="650" y="1289" type="line" smooth="yes"/>
        #  <point x="868" y="1289"/>
        #  <point x="956" y="1180"/>
        #  <point x="956" y="1017" type="curve" smooth="yes"/>
        #  <point x="956" y="869"/>
        #  <point x="854" y="753"/>
        #  <point x="651" y="753" type="curve" smooth="yes"/>
        #  <point x="327" y="753" type="line"/>
        #  <point x="329" y="587" type="line"/>
        #  <point x="770" y="587" type="line"/>
        #  <point x="827" y="609" type="line"/>
        #  <point x="1039" y="666"/>
        #  <point x="1164" y="818"/>
        #  <point x="1164" y="1017" type="curve" smooth="yes"/>
        #  <point x="1164" y="1303"/>
        #  <point x="983" y="1456"/>
        #  <point x="650" y="1456" type="curve" smooth="yes"/>
        from fontTools.ttLib import TTFont
        font = TTFont('fonts/apache/roboto/Roboto-Regular.ttf')
        control_points_l = [
            [
                [0.1, 0.1],
                [0.9, 0.9],
                [0.5, 0.9]
            ],
            [
                [0.5, 0.9],
                [0.1, 0.9],
                [0.3, 0.3]
            ],
            [
                [0.3, 0.3],
                [0.9, 0.9],
                [0.9, 0.1]
            ],
        ]
    if args.batch:
        control_points_l = control_points_l * args.batch

    #  control_points_batch = control_points_l * batch_size
    # print(torch.Tensor(np.array(control_points_batch)).size())

    control_points_t = Variable(torch.Tensor(np.array(control_points_l)), requires_grad=True)

    elapsed_fw, elapsed_bw = 0, 0

    passes = args.passes
    crit = torch.nn.L1Loss().cuda()
    tic_total = time()
    memory_cached, memory_allocated = 0, 0
    for i in range(passes):
        tic = time()
        curve = net.forward(control_points_t)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed_fw += time() - tic
        #  print('{}: Total.'.format(time() - tic))
        loss = crit(curve, curve.clone().detach())
        tic = time()
        #  print('{}: Loss.'.format(time() - tic))
        loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        elapsed_bw += time() - tic
        if use_cuda:
            memory_allocated = torch.cuda.max_memory_allocated()
            memory_cached = torch.cuda.max_memory_cached()

        #  print('{}: Backwards.'.format(time() - tic))

    elapsed = time() - tic_total
    print('forwards:  {:4d} passes in {:7.3f} seconds [{:8.3f} iter/s {:>5.1f} ms/iter].'
            .format(passes, elapsed_fw, passes/elapsed_fw, elapsed_fw/passes * 1e3))
    print('backwards: {:4d} passes in {:7.3f} seconds [{:8.3f} iter/s {:>5.1f} ms/iter].'
            .format(passes, elapsed_bw, passes/elapsed_bw, elapsed_bw/passes * 1e3))
    print('total:     {:4d} passes in {:7.3f} seconds [{:8.3f} iter/s {:>5.1f} ms/iter].'
            .format(passes, elapsed, passes/elapsed, elapsed/passes * 1e3))
    print('memusage:  {:5d} MB allocated, {:5d} MB cached.'
            .format(int(memory_allocated // 1e6), int(memory_cached // 1e6)))

    curve_ = curve.data.cpu().numpy()

    if args.display:
        import matplotlib
        matplotlib.use('tkagg')
        import matplotlib.pyplot as plt
        plt.matshow(curve_)
        plt.show()
