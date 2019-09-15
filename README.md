DeepHunter: A Coverage-Guided Fuzzer for DeepNeural Networks
======

This repository contains a general coverage-guided fuzz testing framework, `DeepHunter`, for testing deep neural networks.


## Installation

We have tested DeepHunter based on Python 2.7 on Ubuntu 16.04 and Mac OS, theoretically it should also work on other operating systems. To get all the dependencies, it is sufficient to run the following command.

```
pip install -r requirements.txt
```

## The structure of the repository

Our original DeepHunter (version 1.0) is implemented on top of [AFL](http://lcamtuf.coredump.cx/afl/).
We publicize DeepHunter 2.0 which is a reimplementation on top of the [TensorFuzz](https://github.com/brain-research/tensorfuzz) framework. It is structured as follows:

```
├── README.md
├── deephunter/
├── lib/
├── test_seeds/
└── requirements.txt
```

### deephunter/

This directory contains the core implementation of DeepHunter, including the metamorphic mutation, coverage analysis (i.e., Neuron Coverage and the metrics in [DeepGuage](https://dl.acm.org/citation.cfm?id=3238202)), and test selection strategies.

The subdirectory *profile* provides 7 profiling files (for 5 trained models and 2 ImageNet models in Keras) required for coverage analysis in DeepGuage. 4 trained models including LeNet-1 LeNet-4 LeNet-5 and ResNet-20 can be found. 
For other 3 models, VGG16 are available [here](https://drive.google.com/drive/folders/1OSv-IXDxnIclnBVoHiUGlm_UqVX2YJTy?usp=sharing), and MobileNet and RestNet-50 are the pre-trained weights from Keras2.1.3.

### lib/

Similar to TensorFuzz, this directory contains the general implementation of fuzz loop and the seed queue.


### test_seeds/
We provide some initial test seeds for MNIST, CIFAR-10 and ImageNet-10.
The user can generate more initial seeds with `utils/ConstructInitialSeeds.py`.



## Usage

### An example script:
We provide a script to automatically run DeepHunter fuzzer as well as results analysis.

```
cd deephunter_source/deephunter/
sh test_example.sh
```

The scrpit tests LeNet-5 with different seed selection strategies and two testing criteria (i.e., KMNC, NBC).
The user can follow the commands to add configurations with other testing criteria.



### Example of Runing DeepHunter:

```
cd deephunter_source/deephunter/
python image_fuzzer.py
       -i ../test_seeds/mnist_seeds
       -o mnist_output
       -model lenet5
       -criteria kmnc
       -random 0 -select prob
```

The meanings of the options are:

1. `-model` determines which model to be tested, where the 7 pre-profiled files are: `vgg16`, `resnet20`, `mobilenet`, `resnet50`, `lenet1`, `lenet4`, `lenet5`. To be started easily, we integrated 7 models, i.e., their profiling files can be found by DeepHunter. It is fairly possible to test other models; in such scenarios, please use `utils/Profile.py` to profile the new model firstly. Then, it is also easy to integrate the new model and the profiling file in DeepHunter.  
2. `-criteria` selects different testing criteria; possible values are: `kmnc`, `nbc`, `snac`, `bknc`, `tknc`, `nc`.  
3. `-random` determines whether applying random testing (1) or coverage-guided testing (0); defaults to 0.
4. `-select` chooses the selection strategies (when `-random` is 0) from `uniform`, `tensorfuzz`, `deeptest`, `prob`.  

Note that LeNet models are used for MNIST; vgg16 and resnet20 are for CIFAR-10; mobilenet and resnet50 are for ImageNet. 
So if you choose one specific model, please make sure that the initial seeds (`-i`) are consistent with the model.

### Profiling for new models

```
cd deephunter_source/deephunter/
python utils/Profile.py 
       -model model_path
       -train  mnist
       -o example.pickle
```

The meanings of the options are:  
1. `-model` specifies the path of the model which can some `h5` file such as lenet1.h5.  
2. `-train` chooses the datasets to be profiled which can be either `mnist` (MNIST) and `cifar` (CIFAR-10). These are widely used datasets that can be obtained from Keras. It is absolutely easy to extend with other datasets.  



## Citation
Please cite the following paper if DeepHunter helps you on the research:

```
@inproceedings{deephunter,  
	Author = {Xiaofei Xie and Lei Ma and Felix Juefei-Xu and Minhui Xue and Hongxu Chen
	          and Yang Liu and Jianjun Zhao and Bo Li and Jianxiong Yin and Simon See},
	Booktitle = {28th ACM SIGSOFT International Symposium on Software Testing and Analysis},
	Title = {DeepHunter: A Coverage-Guided Fuzz Testing Framework for Deep Neural Networks},
	Year = {2019}}
```
