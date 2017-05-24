# [SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd): Changing it to use [Residual Nets](http://arxiv.org/abs/1512.03385).

The basic algorithm is [SSD (*Single Shot Detection*)](http://arxiv.org/abs/1512.02325).

## Software Installation
* In order to take advantage of GPU enabled TensorFlow, install using `pip` and the appropriate binary download (make sure the URL contains GPU, not CPU) (https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-1.1.0-py3-none-any.whl). Or just `pip install tensorflow-gpu` if that works.
* Download CUDA manually (here version 8.0) and install as directed (files go in `/Developer/NVIDIA/`)
* Install cuDNN (it requires signing into NVIDIA Developer), copy the libs to `/Developer/NVIDIA/CUDA-8.0/lib`. Currently TensorFlow (or just Keras) requires cuDNN version 5, not the latest, 6.
* `pip install numpy` (DON’T allow Anaconda to install other versions; it will mess up a lot of things)  — should be installed automatically by tensorflow
* `pip install matplotlib`
* `pip install keras==1.2.2` the version that this SSD was written to handle
* `pip install h5py` for the initialization/pretrained weights 
* Speaking of weights, download pretrained VGG-16 weights at [mega.nz](https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA) and ResNet weights at [fchollet's github](https://github.com/fchollet/deep-learning-models/releases/tag/v0.1). It is recommended to train the new ResNet version of SSD with those weights, as it will help transfer learning (the wave of the future, according to Andrew Ng)
* `pip install pillow` scipy will not function without it
* `pip install opencv-python` (don’t forget the -python)

To test the training, you will need to download the [PASCAL VOC 2007](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) data; untar it into the `PASCAL_VOC` directory. You will also need to initialize the weights with the [ImageNet data](https://github.com/fchollet/deep-learning-models/releases/tag/v0.1); expand that into the root level.

## Other Requirements
 A machine with a GPU with at least 2 GB of memory (on my 1GB GeForce 650GT, training results in out-of-memory errors). I thus spun up a GPU instance (`g2.2xlarge`) on AWS. I probably could do it on Google Cloud also (because I still have credits!), but I'm less familiar with that. This sufficed for training under the VGG-16 architecture (823 seconds per epoch), but not for ResNets, for which I upgraded to `p2.xlarge` (with 12GB GPUs); the VGG-16 architecture here now takes "only" 384 seconds per epoch. The total training time for `p2.xlarge` was 3 hours and 13 minutes.

I tested this with `Keras` v1.2.2, `Tensorflow` v1.1.0, `OpenCV` v3.2.0.7, in Python 3.6.
