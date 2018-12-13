#!/bin/python3

import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from torch.autograd import Variable
from torch.multiprocessing import Pool
from time import time

from bezier import Bezier

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='')
parser.add_argument('--display', action='store_true', help='')
parser.add_argument('--save', action='store_true', help='')
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
# Make all tensors cuda tensors
#  torch.set_default_tensor_type(torch.cuda.FloatTensor if use_cuda else torch.FloatTensor)
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device "{}"'.format(device))

net = Bezier(res=args.res, steps=args.steps, method=args.method, device=device, debug=args.debug)

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

elif args.draw == 'composite':
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
elif args.draw == 'char':
    A = pickle.load(open("./fonts/arial.quadratic","rb"))
    B = 0 #letter chooser
    C = 1.5 #scaling
    D = np.max(A[B])
    control_points_l = np.zeros((len(A[B][0]),len(A[B]),2))
    k = -1
    for i in A[B]:
        k += 1
        for j in range(len(i)):
            control_points_l[j][k][0] = A[B][k][j][0]/C/D+(C-1)/C/2
            control_points_l[j][k][1] = 1-(A[B][k][j][1]/C/D+(C-1)/C/2)
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
    sns.set()
    sns.set_style('white')
    #  sns.set_palette('Reds')
    sns.heatmap(curve_, cmap='Greys')
    #  plt.matshow(curve_)
    plt.show()
