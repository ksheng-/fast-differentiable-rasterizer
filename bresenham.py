from skimage.draw import bezier_curve
import time

tic = time.time()
for i in range(1000):
    bezier_curve(51, 51, 491, 491, 51, 491, 128)
print(time.time() - tic)
