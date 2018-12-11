# fast-differentiable-rasterizer

### Resources
http://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/opengl/gpupathrender.pdf

### Preliminary performance testing

##### Machine specs
MacBook: 2.2 GHz 2 core Broadwell i7, 8GB RAM, no GPU\
GCP n1-standard-8 VM : 2.2 GHz 8 vCPU Broadwell E5, 30GB RAM, Tesla V100

|Algorithm |MacBook  |GCP VM    |
|----------|--------:|---------:|
|naive     |0.965s   |0.405s    |

## Experiments
GCP n1-standard-8 (2.2 GHz / 8 core Broadwell, 30GB RAM) instance with Tesla V100
PyTorch 1.0.0 running with Python 3.7, CUDA 9.0, cuDNN 7.1
res=512, steps=128


### Test 1: ms/iter, quadratic curve (100 passes)
python bezier.py --passes 100 --method <method> [--disable-cuda]

|method         |forward  |backward |total    |speedup |peak_mem |
|---------------|--------:|--------:|--------:|-------:|--------:|
|base           |414.9 ms |448.4 ms |863.3 ms |1.00x   |N/A      |
|cuda           |3.1 ms   |5.7 ms   |8.8 ms   |98.10x  |942 MB   |
|half           |2.7 ms   |3.6 ms   |6.3 ms   |137.03x |472 MB   |
|bounded        |1.9 ms   |2.5 ms   |4.3 ms   |196.20x |245 MB   |
|tiled          |9.4 ms   |2.0 ms   |11.4 ms  |75.73x  |153 MB   |
|shrunk_cpu     |16.2 ms  |142.5 ms |158.7 ms |5.44x   |N/A      |
|shrunk_cuda    |18.3 ms  |22.9 ms  |41.2 ms  |20.95x  |8 MB     |

### Test 2: ms/iter cubic curve (100 passes)
python bezier.py --passes 100 --draw cubic --method <method> [--disable-cuda]

|method         |forward  |backward |total    |speedup |peak_mem |
|---------------|--------:|--------:|--------:|-------:|--------:|
|base           |418.4 ms |449.0 ms |867.4 ms |1.00x   |N/A      |
|cuda           |3.3 ms   |6.1 ms   |9.4 ms   |92.28x  |942 MB   |
|half           |2.9 ms   |4.0 ms   |6.9 ms   |125.71x |472 MB   |
|bounded        |2.9 ms   |4.2 ms   |7.1 ms   |122.17x |472 MB   |
|tiled          |10.4 ms  |2.3 ms   |12.7 ms  |68.30x  |148 MB   |
|shrunk_cuda    |18.2 ms  |22.9 ms  |41.1 ms  |21.10x  |8 MB     |

comments: interpolation is cheap (number of points does not increase, but spacial coverage does)

### Test 3: ms/iter 3 section composite quadratic curve (100 passes)
python bezier.py --passes 100 --draw char --method <method> [--disable-cuda]

|method         |forward  |backward |total    |speedup |peak_mem |
|---------------|--------:|--------:|--------:|-------:|--------:|
|base           |1131.9 ms|1217.5 ms|2349.4 ms|1.00x   |N/A      |
|cuda           |8.9 ms   |15.8 ms  |24.7 ms  |95.12x  |2821 MB  |
|half           |7.4 ms   |9.4 ms   |16.8 ms  |139.85x |1412 MB  |
|bounded        |5.3 ms   |7.9 ms   |13.2 ms  |177.98x |1053 MB  |
|tiled          |20.1 ms  |2.7 ms   |22.8 ms  |130.04x |434 MB   |
|shrunk_cpu     |48.8 ms  |383.8 ms |432.6 ms |5.43x   |N/A      |
|shrunk_cuda    |55.9 ms  |69.7 ms  |125.6 ms |18.71x  |15 MB    |

### Test 4: ms/iter 16 section composite quadratic curve (100 passes)
NOT UP TO DATE
python bezier.py --passes 100 --batch 16 --method <method> [--disable-cuda]

|method         |forward  |backward |total    |speedup |peak_mem |
|---------------|--------:|--------:|--------:|-------:|--------:|
|base           |414.9 ms |448.4 ms |863.3 ms |1.00x   |N/A      |
|cuda           |0.6 ms   |5.7 ms   |6.3 ms   |137.03x |942 MB   |
|half           |0.8 ms   |4.1 ms   |4.9 ms   |176.18x |607 MB   |
|bounded        |2.5 ms   |3.3 ms   |5.8 ms   |148.84x |486 MB   |
|tiled          |15.2 ms  |13.1 ms  |28.3 ms  |30.51x  |347 MB   |
|shrunk_cpu     |16.2 ms  |142.5 ms |158.7 ms |5.44x   |N/A      |
|shrunk_cuda    |18.3 ms  |22.9 ms  |41.2 ms  |20.95x  |8 MB     |

### Test 5: ms/iter vs num_curves
python bezier.py --passes 100 --batch <num_curves> --method <method> [--disable-cuda]

|num_curves     |base     |half     |tiled    |
|---------------|--------:|--------:|--------:|
|2              |1.6 s    |12.7 ms  |17.7 ms  |
|4              |3.3 s    |22.8 ms  |27.0 ms  |
|8              |6.3 s    |43.4 ms  |47.1 ms  |
|16             |12.9 s   |86.4 ms  |80.4 ms  |
|32             |26.1 s   |168.9 ms |163.8 ms |

### Test 6: max_memory_allocated vs num_curves
python bezier.py --passes 100 --batch <num_curves> --method <method>
  
|num_curves     |half     |tiled    |
|---------------|--------:|--------:|
|2              |0.9 GB   |0.3 GB   |
|4              |1.9 GB   |0.6 GB   |
|8              |3.8 GB   |1.2 GB   |
|16             |7.5 GB   |2.4 GB   |
|32             |15.0 GB  |4.8 GB   |


Test 7: Snakeviz / profiler output
