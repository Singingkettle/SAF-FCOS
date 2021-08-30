# SAF-FCOS: Spatial Attention Fusion for Obstacle Detection using MmWave Radar and Vision Sensor

This project hosts the code for implementing the SAF-FCOS algorithm for object detection, as presented in our paper:

    SAF-FCOS: Spatial Attention Fusion for Obstacle Detection using MmWave Radar and Vision Sensor;
    Shuo Chang, YiFan Zhang, Fan Zhang, Xiaotong Zhao, Sai Huang, ZhiYong Feng and Zhiqing Wei;
    In: Sensors, 2019.
And the whole project is built upon FCOS, Below is FCOS license.

```
FCOS for non-commercial purposes

Copyright (c) 2019 the authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```

The full paper is available at: [https://www.mdpi.com/1424-8220/20/4/956](https://www.mdpi.com/1424-8220/20/4/956). 

## You should known
Please read the FCOS project first [FCOS-README.md](FCOS-README.md) 

## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions.

## Generate Data

1. Please download Full dataset (v1.0) of nuScenes dataset from the [link](https://www.nuscenes.org/download).
![download](./image/download.png)
2. Then, upload all download tar files to an ubuntu server, and uncompress all *.tar files in a specific folder.
```shell
mkdir ~/Data/nuScenes
mv AllDownloadTarFiles ~/Data/nuScenes
cd ~/Data/nuScenes
for f in *.tar; do tar -xvf "$f"; done
```
3. 