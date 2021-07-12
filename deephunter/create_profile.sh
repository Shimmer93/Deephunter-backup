: '
To use the scripts, the output (i.e., -o) of DeepHunter should follow the structure: root_dir/strategy/metrics/id
The strategy and metrics must be the same name with the option, i.e., strategy must be one of [random,uniform,tensorfuzz,deeptest,prob]
metrics must be one of [nbc,snac,tknc,kmnc,nc]. To get the coverage of random strategy in terms of a specific metric, we also need to select the specific metric.
id can be any number.

Before using the new scripts, please install the xxhash by "pip install xxhash"
'
#python utils/Profile.py -model /data/dnntest/zpengac/models/resnet/cifar10_resnet20v1_keras_deephunter_prob_kmnc2.h5 -train cifar -o /data/dnntest/zpengac/deephunter/deephunter/profile/cifar10_resnet20v1_keras_deephunter_prob_kmnc2
#python utils/Profile.py -model /data/dnntest/zpengac/models/resnet/cifar10_resnet20v1_keras_deephunter_prob_nbc2.h5 -train cifar -o /data/dnntest/zpengac/deephunter/deephunter/profile/cifar10_resnet20v1_keras_deephunter_prob_nbc2
#python utils/Profile.py -model /data/dnntest/zpengac/models/resnet/cifar10_resnet20v1_keras_deephunter_uniform_kmnc2.h5 -train cifar -o /data/dnntest/zpengac/deephunter/deephunter/profile/cifar10_resnet20v1_keras_deephunter_uniform_kmnc2
python utils/Profile.py -model /data/dnntest/zpengac/models/resnet/cifar10_resnet20v1_keras_deephunter_uniform_nbc2.h5 -train cifar -o /data/dnntest/zpengac/deephunter/deephunter/profile/cifar10_resnet20v1_keras_deephunter_uniform_nbc2