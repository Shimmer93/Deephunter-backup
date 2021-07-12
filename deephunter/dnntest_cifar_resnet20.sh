: '
To use the scripts, the output (i.e., -o) of DeepHunter should follow the structure: root_dir/strategy/metrics/id
The strategy and metrics must be the same name with the option, i.e., strategy must be one of [random,uniform,tensorfuzz,deeptest,prob]
metrics must be one of [nbc,snac,tknc,kmnc,nc]. To get the coverage of random strategy in terms of a specific metric, we also need to select the specific metric.
id can be any number.

Before using the new scripts, please install the xxhash by "pip install xxhash"
'
#python image_fuzzer.py  -i ../test_seeds/dnntest_cifar  -o dnntest_resnet20_retrain3/uniform/nbc/0 -model resnet20 -criteria nbc -random 0 -select uniform -max_iteration 1000
#python image_fuzzer.py  -i ../test_seeds/dnntest_cifar  -o dnntest_resnet20_retrain3/uniform/kmnc/0 -model resnet20 -criteria kmnc -random 0 -select uniform -max_iteration 1000
#python image_fuzzer.py  -i ../test_seeds/dnntest_cifar  -o dnntest_resnet20_retrain3/prob/nbc/0 -model resnet20 -criteria nbc -random 0 -select prob -max_iteration 1000
python image_fuzzer.py  -i ../test_seeds/dnntest_cifar  -o dnntest_resnet20_retrain3/prob/kmnc/0 -model resnet20 -criteria kmnc -random 0 -select prob -max_iteration 1000
#python utils/CoveragePlot.py -i dnntest_resnet20_retrain2 -type coverage -iterations 1000 -o  results_cifar/dnntest_coverage_plot.pdf
#python utils/CoveragePlot.py -i dnntest_resnet20_retrain2 -type seedattack -iterations 1000 -o  results_cifar/dnntest_diverse_plot.pdf
#python utils/UniqCrashBar.py -i dnntest_resnet20_retrain2 -iterations 1000 -o  results_cifar/dnntest_uniq_crash.pdf
echo 'Finish! Please find the results in the results directory.'


