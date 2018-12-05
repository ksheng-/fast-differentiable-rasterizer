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

<<<<<<< HEAD
##### Machine specs 2
Desktop: 3.5 GHz AMD FX(tm)-6300 Six-Core Processor, 10GB RAM, Geforce GTX 1070

=======
## Benchmarks
Averaged across 100 passes with params res=512, steps=128 on a GCP n1-standard-8 VM (2.2 GHz 8 core Broadwell E5, 30GB RAM, Tesla V100) running pytorch 0.4.1 with python 3.7, cuda 9.2, and cudnn 7.1.

|algorithm      |forward  |backward |total    |speedup |
|---------------|--------:|--------:|--------:|-------:|
|baseline       |354.4 ms |391.2 ms |745.7 ms |1.00x   |
|shrunk_cpu     |12.7 ms  |113.4 ms |126.1 ms |5.91x   |
|shrunk_cuda    |49.4 ms  |20.0 ms  |69.4 ms  |10.74x  |
|baseline_cuda  |0.9 ms   |8.3 ms   |9.2 ms   |81.05x  |
>>>>>>> upstream/master

