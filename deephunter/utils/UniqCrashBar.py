import argparse, pickle
from subprocess import Popen, PIPE
import time
import os
import csv
import numpy as np
import matplotlib
from collections import defaultdict
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import xxhash
def read_from_file(crash_dir, bugs):
    s = set()

    cp = sorted(os.listdir(crash_dir))
    if bugs is not None:
        cp = cp[:bugs]
    for file in cp:
        if not file.startswith('id'):
            continue

        file = os.path.join(crash_dir, file)
        new_img = np.load(file)

        h = xxhash.xxh64()
        h.update(new_img)
        q = h.intdigest()
        s.add(q)
        h.reset()
    return len(s)


# strategy = ['deeptest','prob', 'queue_random', 'tensorfuzz','random']
# # strategy = ['deeptest','random', 'prob', 'queue_random', 'tensorfuzz']
# criteria = ['nbc', 'kmnc', 'snac', 'tknc', 'bknc', 'nc']
# #'kmnc', 'nbc', 'snac', 'tknc', 'bknc', 'nc'   , 'random', 'prob', 'queue_random', 'tensorfuzz'
# # strategy = ['prob', 'queue_random']

data_info = {}
def get_data(base_path, strategy_for_criteria, prefix, axs, iterations):
    ax_num = 0
   
    for cri, strategy in strategy_for_criteria.items():
        ax = axs[ax_num]
        ax.set_title(prefix + cri.upper())
        
        ax.set_ylabel('#unique crashes')
        x_labels = []
        cur_results = []

        for prio in strategy:
            if not os.path.exists(os.path.join(base_path, prio)):
                continue
            if prio == 'uniform': # queue_random
                x_labels.append('DH+UF')
            elif prio == 'prob':
                x_labels.append('DH+Prob')
            elif prio == 'random':
                x_labels.append('Random')
            elif prio == 'tensorfuzz':
                x_labels.append('TensorFuzz')
            elif prio == 'deeptest':
                x_labels.append('DeepTest')

            strategy_path = os.path.join(base_path,prio)
            cri_path = os.path.join(strategy_path,cri)
            # repeat experiments folder:0,1,2,3...
            fuzz_dirs = os.listdir(cri_path)
            count = 0
            total_bugs = 0
            for dir in fuzz_dirs:
                crashes = os.path.join(cri_path, dir, 'crashes')
                bugs = None
                plot_file = os.path.join(cri_path, dir, 'plot.log')
                if iterations is not None:
                    file = open(plot_file, 'r')
                    contents = file.readlines()
                    if len(contents) > iterations:
                        line = contents[iterations-1]
                        bugs = int(line.split(',')[5])
                    if os.path.isdir(crashes):
                        total_bugs += read_from_file(crashes, bugs)
                        count += 1
            data_info[(cri,prio)] = float(total_bugs)/count
            cur_results.append(float(total_bugs)/count)
            print("finish ", cri, prio, float(total_bugs)/count)
        
        ax.bar(x_labels, cur_results,color=['b', 'orange', 'g','r','m'])
        ax_num += 1
      


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='coverage guided fuzzing')
    parser.add_argument('-i',help='root output dir for results used as input dir here')
    parser.add_argument('-m',help='the chosen model')
    # parser.add_argument('-i', help='input seed dir')
    parser.add_argument('-iterations', help='seed output', type=int)
    parser.add_argument('-o', help='figure name to output(no postfix)')

    args = parser.parse_args()

    parent = os.path.dirname(args.o)
    if not os.path.exists(parent):
        try:
            os.makedirs(parent)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    input = args.i
    # model = args.m
    base_path = input #os.path.join(input,model)


    strategies = os.listdir(base_path)
    criteria_for_strategy = defaultdict() # { prob:[cri1,cri2,...], }
    strategy_for_criteria = defaultdict(list) # { kmnc:[stratgey1,strategy2,...] }
    for strategy in strategies:
        criteria_for_strategy[strategy] = os.listdir("{0}/{1}".format(base_path,strategy))
        for criteria in criteria_for_strategy[strategy]:
            strategy_for_criteria[criteria].append(strategy)

    effective_criteria = []
    effective_criteria_dict={}
    for cri,strs in strategy_for_criteria.items():
        if len(strs)>0:
            effective_criteria.append(cri)
            effective_criteria_dict[cri] = strs

    if len(effective_criteria) <= 3:
        row = 1
        col = len(effective_criteria)
    elif len(effective_criteria) == 4:
        row,col = 2,2
    else:
        row,col = 2,3


    fig, axs = plt.subplots(nrows=row, ncols=col, constrained_layout=True)

    get_data(base_path, effective_criteria_dict, "", axs.flatten(), args.iterations)

    plt.legend()
    fig.set_size_inches(18, 10)
    plt.savefig("{0}.pdf".format(args.o),  format='pdf')
    # plt.show()

    import csv
    full_criteria = ['nc','kmnc', 'nbc', 'snac', 'tknc']
    head_row = ['strategy\\criteria'] + full_criteria
    csv_rows = [head_row]
    for i in strategies:
        line = [i]
        for j in full_criteria:
            if (j,i) not in data_info:
                line.append("N/A")
                continue
            line.append(data_info[(j,i)])
        csv_rows.append(line)

    with open('{0}.csv'.format(args.o), 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_rows)

