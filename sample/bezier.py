#!/bin/python3

import numpy as np

control_points_l = [
    [0, 0],
    [0.5, 0.5],
    [0, 1]
    ]

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

x = curve[:, 0]
y = curve[:, 1]

import matplotlib.pyplot as plt

plt.plot(x, y)
plt.show()