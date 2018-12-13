#!/bin/python3

import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Variable
from torch.multiprocessing import Pool
from tabulate import tabulate
from time import time

from bezier import Bezier
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--save-plot', action='store_true', help='save plot of benchmark results')
parser.add_argument('--display', action='store_true', help='show plot of benchmark results')
parser.add_argument('--debug', action='store_true', help='')
parser.add_argument('--steps', default=128, type=int, help='')
parser.add_argument('--res', default=512, type=int, help='')
parser.add_argument('--passes', default=100, type=int, help='')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

def benchmark(net, input, passes):
        crit = torch.nn.L1Loss().to(net.device)
        elapsed_fw, elapsed_bw = 0, 0
        memory_cached, memory_allocated = 0, 0
        
        tic_total = time()
        for i in range(passes):
            tic = time()
            curve = net.forward(control_points_t)
            if use_cuda:
                torch.cuda.synchronize()
            elapsed_fw += time() - tic
       
            loss = crit(curve, curve.clone().detach())
            
            tic = time()
            loss.backward()
            if use_cuda:
                torch.cuda.synchronize()
            elapsed_bw += time() - tic
           
            if str(net.device) == 'cuda':
                memory_allocated = max(torch.cuda.memory_allocated(), memory_allocated)
                memory_cached = max(torch.cuda.memory_cached(), memory_allocated)

        elapsed = time() - tic_total
        return [net.method + ' + cuda' if str(net.device) == 'cuda' else net.method, (elapsed_fw + elapsed_bw) / passes * 1e3, elapsed_fw / passes * 1e3, elapsed_bw / passes * 1e3, memory_cached / 1e9 if memory_cached > 0 else 0]
        curve_ = curve.data.cpu().numpy()


cpu_methods = ['base', 'shrunk']
cuda_methods = ['shrunk', 'base', 'half', 'bounded', 'tiled']
test_curves = {
    #  'quadratic':  [
        #  [
            #  [0.1, 0.1],
            #  [0.9, 0.9],
            #  [0.5, 0.9]
        #  ]
    #  ],
    #  'cubic': [
        #  [
            #  [1.0, 0.0],
            #  [0.21, 0.12],
            #  [0.72, 0.83],
            #  [0.0, 1.0]
        #  ]
    #  ],
    #  'glyph-A': utils.load_glyph(26, scale=1.5),
    'glyph-a': utils.load_glyph(0, scale=1.5)
}

save_dir = './benchmarks'

headers = ['Method', 'Total (ms)', 'Forward (ms)', 'Backward (ms)', 'GPU mem (GB)'] 
for curve, control_points in test_curves.items():
    print('-' * 80)
    print('Benchmarking rasterization of {} curve:'.format(curve))
    print('') 
    control_points_t = Variable(torch.Tensor(np.array(control_points)), requires_grad=True)
    results = []
    for method in cpu_methods:
        device = torch.device('cpu')
        print('Measuring algorithm {} ({})...'.format(method, device))
        net = Bezier(res=args.res, steps=args.steps, method=method, device=device, debug=args.debug)
        if use_cuda:
            torch.cuda.empty_cache()
        results.append(benchmark(net, control_points_t, 10))
    for method in cuda_methods:
        device = torch.device('cuda' if use_cuda else 'cpu')
        print('Measuring algorithm {} ({})...'.format(method, device))
        net = Bezier(res=args.res, steps=args.steps, method=method, device=device, debug=args.debug)
        if use_cuda:
            torch.cuda.empty_cache()
        results.append(benchmark(net, control_points_t, args.passes))
   
    table = tabulate(results, headers, tablefmt='rst', floatfmt=('', '.1f', '.1f', '.1f', '.3f'))
    print('')
    print(table)
    print('')
    
    pathname = '{}/{}.md'.format(save_dir, curve)
    print('Saving results to "{}"...'.format(pathname))
    with open(pathname, 'w+') as f:
        table = tabulate(results, headers, tablefmt='pipe', floatfmt=('', '.1f', '.1f', '.1f', '.3f'))
        f.write(table + '\n')
    
    pathname = '{}/{}.tex'.format(save_dir, curve)
    print('Saving results to "{}"...'.format(pathname))
    with open(pathname, 'w+') as f:
        table = tabulate(results, headers, tablefmt='latex', floatfmt=('', '.1f', '.1f', '.1f', '.3f'))
        f.write(table + '\n')
    print('')

headers = ['Method', 'N=2 (ms)', 'N=4 (ms)', 'N=8 (ms)', 'N=16 (ms)', 'N=32 (ms)'] 
print('-' * 80)
print('Benchmarking rasterization of N quadratic curves:')
print('') 

results_time = []
results_mem = []
num_curves = [2, 4, 8, 16]
for method in cpu_methods:
    device = torch.device('cpu')
    print('Measuring algorithm {} ({})...'.format(method, device))
    result_time = []
    result_mem = []
    for N in num_curves:
        control_points = test_curves['quadratic'] * N 
        control_points_t = Variable(torch.Tensor(np.array(control_points)), requires_grad=True)
        print('  N={}...'.format(N))
        net = Bezier(res=args.res, steps=args.steps, method=method, device=device, debug=args.debug)
        if use_cuda:
            torch.cuda.empty_cache()
        result = benchmark(net, control_points_t, 3)
        meth = result[0]
        result_time.append(result[1])
        result_mem.append(result[4])
    results_time.append([meth] + result_time)
    results_mem.append([meth] + result_mem)
for method in cuda_methods:
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Measuring algorithm {} ({})...'.format(method, device))
    result_time = []
    result_mem = []
    for N in num_curves:
        control_points = test_curves['quadratic'] * N 
        control_points_t = Variable(torch.Tensor(np.array(control_points)), requires_grad=True)
        print('  N={}...'.format(N))
        net = Bezier(res=args.res, steps=args.steps, method=method, device=device, debug=args.debug)
        if use_cuda:
            torch.cuda.empty_cache()
        result = benchmark(net, control_points_t, args.passes)
        meth = result[0]
        result_time.append(result[1])
        result_mem.append(result[4])
    results_time.append([meth] + result_time)
    results_mem.append([meth] + result_mem)
print('')

table = tabulate(results_time, headers, tablefmt='rst', floatfmt='.1f')
print(table)
print('')

pathname = '{}/multicurve_time.md'.format(save_dir, curve)
print('Saving results to "{}"...'.format(pathname))
with open(pathname, 'w+') as f:
    table = tabulate(results_time, headers, tablefmt='pipe', floatfmt='.1f')
    f.write(table + '\n')

pathname = '{}/multicurve_time.tex'.format(save_dir, curve)
print('Saving results to "{}"...'.format(pathname))
with open(pathname, 'w+') as f:
    table = tabulate(results_time, headers, tablefmt='latex', floatfmt='.1f')
    f.write(table + '\n')
print('')

table = tabulate(results_mem, headers, tablefmt='rst', floatfmt='.3f')
print(table)
print('')

pathname = '{}/multicurve_mem.md'.format(save_dir, curve)
print('Saving results to "{}"...'.format(pathname))
with open(pathname, 'w+') as f:
    table = tabulate(results_mem, headers, tablefmt='pipe', floatfmt='.3f')
    f.write(table + '\n')

pathname = '{}/multicurve_mem.tex'.format(save_dir, curve)
print('saving results to "{}"...'.format(pathname))
with open(pathname, 'w+') as f:
    table = tabulate(results_mem, headers, tablefmt='latex', floatfmt='.3f')
    f.write(table + '\n')
print('')

def get_data(results):
    return np.array([result[0] for result in results]), np.array([result[1:] for result in results]).T

if args.save_plot:
    sns.set()
    sns.set_style('white')
    legend, data = get_data(results_mem)
    plt.plot(num_curves, data)
    pathname = '{}/multicurve_mem.svg'.format(save_dir)
    result = benchmark(net, control_points_t, 1)
    plt.savefig(pathname)
    if args.display:
        plt.show()
