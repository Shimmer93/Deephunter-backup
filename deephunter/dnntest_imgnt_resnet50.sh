  GNU nano 2.3.1                                           File: test_example.sh

: '
To use the scripts, the output (i.e., -o) of DeepHunter should follow the structure: root_dir/strategy/metrics/id
The strategy and metrics must be the same name with the option, i.e., strategy must be one of [random,uniform,tensorfuzz,deeptest,prob]
metrics must be one of [nbc,snac,tknc,kmnc,nc]. To get the coverage of random strategy in terms of a specific metric, we also need to select the specific metric.
id can be any number.

Before using the new scripts, please install the xxhash by "pip install xxhash"
'
python image_fuzzer.py  -i ../test_seeds/dnntest_imgnt  -o dnntest_resnet50_out/random/kmnc/0 -model resnet50 -random 0 -max_iteration 1000 -criteria kmnc
python image_fuzzer.py  -i ../test_seeds/dnntest_imgnt  -o dnntest_resnet50_out/random/nbc/0 -model resnet50 -random 0 -max_iteration 1000 -criteria nbc
python image_fuzzer.py  -i ../test_seeds/dnntest_imgnt  -o dnntest_resnet50_out/uniform/nbc/0 -model resnet50 -criteria nbc -random 0 -select uniform -max_iteration 1000
python image_fuzzer.py  -i ../test_seeds/dnntest_imgnt  -o dnntest_resnet50_out/uniform/kmnc/0 -model resnet50 -criteria kmnc -random 0 -select uniform -max_iteration 1000
python image_fuzzer.py  -i ../test_seeds/dnntest_imgnt  -o dnntest_resnet50_out/prob/nbc/0 -model resnet50 -criteria nbc -random 0 -select prob -max_iteration 1000
python image_fuzzer.py  -i ../test_seeds/dnntest_imgnt  -o dnntest_resnet50_out/prob/kmnc/0 -model resnet50 -criteria kmnc -random 0 -select prob -max_iteration 1000
python utils/CoveragePlot.py -i dnntest_resnet50_out -type coverage -iterations 1000 -o  results_imgnt/dnntest_imgnt_coverage_plot.pdf
python utils/CoveragePlot.py -i dnntest_resnet50_out -type seedattack -iterations 1000 -o  results_imgnt/dnntest_imgnt_diverse_plot.pdf
python utils/UniqCrashBar.py -i dnntest_resnet50_out -iterations 1000 -o  results_imgnt/dnntest_imgnt_uniq_crash.pdf
echo 'Finish! Please find the results in the results directory.'


