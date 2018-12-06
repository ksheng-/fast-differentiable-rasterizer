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
Averaged across 100 passes with params res=512, steps=128 on a GCP n1-standard-8 VM (2.2 GHz 8 core Broadwell E5, 30GB RAM, Tesla V100) running pytorch 0.4.1 with python 3.7, cuda 9.2, and cudnn 7.1.

GCP n1-standard-8 instance with Tesla V100
PyTorch 0.4.1 running with Python 3.7, CUDA 9.2.148, cuDNN 7.1
res=512, steps=128

### Test 1: ms/iter, quadratic curve (100 passes)
python bezier.py --passes 100 --method <method> [--disable-cuda]

|method         |forward  |backward |total    |speedup |peak_mem |
|---------------|--------:|--------:|--------:|-------:|--------:|
|base           |414.9 ms |448.4 ms |    ms   |1.00x   |N/A      |
|cuda           |0.6 ms   |5.7 ms   |    ms   |        |942 MB   |
|half           |0.8 ms   |4.1 ms   |    ms   |        |607 MB   |
|bounded        |2.5 ms   |3.3 ms   |    ms   |        |486 MB   |
|tiled          |15.2 ms  |13.1 ms  |    ms   |        |347 MB   |
|shrunk_cpu     |16.2 ms  |142.5 ms |    ms   |        |N/A      |
|shrunk_cuda    |18.3 ms  |22.9 ms  |    ms   |        |8 MB     |

### Test 2: ms/iter cubic curve (100 passes)
python bezier.py --passes 100 --draw cubic --method <method> [--disable-cuda]

|method         |forward  |backward |total    |speedup |peak_mem |
|---------------|--------:|--------:|--------:|-------:|--------:|
|base           |418.4 ms |449.0 ms |    ms   |1.00x   |N/A      |
|cuda           |0.8 ms   |5.9 ms   |    ms   |        |942 MB   |
|half           |0.9 ms   |4.4 ms   |    ms   |        |607 MB   |
|bounded        |4.2 ms   |5.9 ms   |    ms   |        |486 MB   |
|tiled          |13.6 ms  |12.2 ms  |    ms   |        |340 MB   |
|shrunk_cpu     |15.9 ms  |136.4 ms |    ms   |        |N/A      |
|shrunk_cuda    |18.2 ms  |22.9 ms  |    ms   |        |8 MB     |

comments: interpolation is cheap (number of points does not increase, but spacial coverage does)

### Test 3: ms/iter 3 section composite quadratic curve (100 passes)
python bezier.py --passes 100 --draw char --method <method> [--disable-cuda]

|method         |forward  |backward |total    |speedup |peak_mem |
|---------------|--------:|--------:|--------:|-------:|--------:|
|base           |1131.9 ms|1217.5 ms|    ms   |1.00x   |N/A      |
|cuda           |0.9 ms   |15.7 ms  |    ms   |        |2821 MB  |
|half           |1.4 ms   |11.2 ms  |    ms   |        |1815 MB  |
|bounded        |7.6 ms   |11.7 ms  |    ms   |        |2104 MB  |
|tiled          |28.8 ms  |51.0 ms  |    ms   |        |1031 MB  |
|shrunk_cpu     |48.8 ms  |383.8 ms |    ms   |        |N/A      |
|shrunk_cuda    |55.9 ms  |69.7 ms  |    ms   |        |15 MB    |

### Test 4: ms/iter 16 section composite quadratic curve (100 passes)
python bezier.py --passes 100 --batch 16 --method <method> [--disable-cuda]

|method         |forward  |backward |total    |speedup |peak_mem |
|---------------|--------:|--------:|--------:|-------:|--------:|
|base           |414.9 ms |448.4 ms |    ms   |1.00x   |N/A      |
|cuda           |0.6 ms   |5.7 ms   |    ms   |        |942 MB   |
|half           |0.8 ms   |4.1 ms   |    ms   |        |607 MB   |
|tiled          |15.2 ms  |13.1 ms  |    ms   |        |347 MB   |
|bounded        |2.5 ms   |3.3 ms   |    ms   |        |486 MB   |
|shrunk_cpu     |16.2 ms  |142.5 ms |    ms   |        |N/A      |
|shrunk_cuda    |18.3 ms  |22.9 ms  |    ms   |        |8 MB     |

### Test 5: ms/iter vs num_curves
python bezier.py --passes 100 --batch <num_curves> --method <method> [--disable-cuda]

|num_curves     |base                 |half                 |tiled                  |
|---------------|--------------------:|--------------------:|----------------------:|
|2              |1.6 (0.8F+0.8B) s    |11.5 (0.8F+10.7B) ms |41.6 (18.5F+23.1B) ms  |
|4              |3.3 (1.6F+1.7B) s    |21.9 (1.2F+20.7B) ms |67.1 (23.7F+43.4B) ms  |
|8              |6.3 (2.9F+3.4B) s    |42.3 (1.9F+40.4B) ms |118.5 (33.9F+84.6B) ms |
|16             |12.9  (6.0F+6.9B) s  |84.1 (3.6F+80.5B) ms |DNC                    |

### Test 6: max_memory_allocated vs num_curves
python bezier.py --passes 100 --batch <num_curves> --method <method>
  
|num_curves     |half     |cuda     |
|---------------|--------:|--------:|
|2              |1.9 GB   |0.7 GB   |
|4              |3.8 GB   |1.4 GB   |
|8              |7.5 GB   |2.7 GB   |
|16             |15.0 GB  |DNC      |


Test 7: Snakeviz / profiler output
