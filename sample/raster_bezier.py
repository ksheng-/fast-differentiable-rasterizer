#!/bin/python3

import numpy as np

control_points_1 = [
    [0.1, 0.1],
    [0.9, 0.9],
    [0.5, 0.9]
    ]

control_points_2 = [
    [0.5, 0.9],
    [0.1, 0.9],
    [0.3, 0.3]
    ]

control_points_3 = [
    [0.3, 0.3],
    [0.9, 0.9],
    [0.9, 0.1]
    ]

def bezier_curve(control_points_l):
    control_points = np.array(control_points_l)

    def lin_interp(point1, point2, num_steps):
        a = point1[0]
        b = point1[1]
        a_= point2[0]
        b_ = point2[1]

        t = np.linspace(0, 1, num_steps)
        
        interp1 = a + t * (a_ - a)
        interp2 = b + t * (b_ - b)
        
        return np.stack([interp1, interp2])

    steps = 100
    set1 = lin_interp(control_points[0], control_points[1], steps)
    set2 = lin_interp(control_points[1], control_points[2], steps)

    curve_l = []
    for i in range(steps):
        a = set1[:,i]
        b = set2[:,i]
        
        point = [
            a[0] + (i / steps) * (b[0] - a[0]),
            a[1] + (i / steps) * (b[1] - a[1])
        ]
        curve_l.append(point)

    curve = np.array(curve_l)

    return curve

curve1 = bezier_curve(control_points_1)
curve2 = bezier_curve(control_points_2)
curve3 = bezier_curve(control_points_3)

curve = np.concatenate([curve1, curve2, curve3], axis=0)

x = curve[:, 0]
y = curve[:, 1]

import matplotlib.pyplot as plt
from itertools import product
import numpy as np
C, D = np.meshgrid(range(-10, 10), range(-10, 10))

# def sample():
#     res = 512
#     raster = np.zeros((res, res))
#     for a, b in zip(x, y):    
#         c_ = a + C / res
#         d_ = b + D / res
#         raster[np.round(d_*res).astype(np.int32), np.round(c_*res).astype(np.int32)]  += 1 # np.exp(-(a - c_)**2 / 1e-4 - (b - d_) ** 2 / 1e-4)        
#         print(np.amax(raster))
#         yield raster

def sample():
    res = 128
    raster = np.zeros((res, res))
    #for a, b in zip(x, y):
    for c, d in product(range(res), range(res)):
        c_ = c / res
        d_ = d / res
        raster[d, c] = np.sum(np.exp(-(x - c_)**2 / 1e-4 - (y - d_) ** 2 / 1e-4))
        if raster[d, c] < 1e-3:
            continue 
        print(np.amax(raster))
        yield raster
        
def sample():
    res = 256
    raster = np.zeros((res, res))
    C, D = np.meshgrid(range(res), range(res))    
    C_e = C[np.newaxis, :, :]
    D_e = D[np.newaxis, :, :]
    x_ = x[:, np.newaxis, np.newaxis]
    y_ = y[:, np.newaxis, np.newaxis]    
    c_ = C_e / res
    d_ = D_e / res
    #raster[np.round(d_*res).astype(np.int32), np.round(c_*res).astype(np.int32)]  = np.exp(-(x - c_)**2 / 1e-4 - (y - d_) ** 2 / 1e-4)
    from time import time
    tic = time()
    raster = np.exp(-(x_ - c_)**2 / 2e-4 - (y_ - d_) ** 2 / 2e-4)

    return raster

    # #raster = (x_ - c_)**2 + (y_ - d_) ** 2 < 1e-3
    # print(np.amax(raster))
    # toc = time() - tic
    # print(toc)
    # print(raster.shape)
    # plt.matshow(raster)
    # plt.show()
    # exit()
    # yield raster

raster = sample()
print(raster.shape)

def sample2():
    rastered = np.zeros_like(np.squeeze(raster[0]))
    for i in range(300):
        rastered += np.squeeze(raster[i])        
        yield rastered

raster_gen = sample2()
rastered = next(raster_gen)
fig, ax = plt.subplots()
image = ax.matshow(rastered, vmax=5)

def animate(i):
    rastered = next(raster_gen)
    image.set_array(rastered)

    return image

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()    

# print(raster)

# plt.show()
