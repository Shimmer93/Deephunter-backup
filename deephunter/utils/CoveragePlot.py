import argparse, pickle
from subprocess import Popen, PIPE
import time
import os
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read_from_file(log_file, results, nums, type):
    with open(log_file) as f:
        if type == 'seedattack':
            c = 6
        if type == 'coverage':
            c = 4
        lines = f.readlines()
        for content in lines:
            data = content.split(',')
            #time, id, queue_len, init_cov, cur_cov, bugs
            if int(data[1]) in results:
                results[int(data[1])] +=  float(data[c])
                nums[int(data[1])] += 1
                # results[int(data[1])][1] +=  int(data[5])
            else:
                results[int(data[1])] = float(data[c])
                nums[int(data[1])] = 1


last_coverage = {}
def get_data(base_path, strategy_for_criteria, prefix, axs, iter, type):
    ax_num = 0
        #{ cri1:[strategy1,strategy2,...] }
    for cri, strategy in strategy_for_criteria.items():
        ax = axs[ax_num]
        ax.set_title(prefix + cri.upper())
        ax.set_xlabel('Iterations')
        if type == 'seedattack':
            ax.set_ylabel('Diversity of unique errors')
        if type == 'coverage':
            ax.set_ylabel('Coverage(%)')

        for prio in strategy:
            nums = {}
            cur_results = {}
            strategy_path = os.path.join(base_path,prio)
            cri_path = os.path.join(strategy_path,cri)
            # repeat experiments folder:0,1,2,3...
            fuzz_dirs = os.listdir(cri_path)
            for dir in fuzz_dirs:
                log_file = os.path.join(cri_path, dir, 'plot.log')
                if os.path.isfile(log_file):
                    read_from_file(log_file, cur_results, nums,type)

            count = 0
            for k in cur_results:
                if nums[k] != nums[0]:
                    break
                count += 1
                cur_results[k] = float(cur_results[k])/nums[k]
                # print(cur_results[k], nums[k])
            # count = 3000
            # count = 5000 if count > 5000 else count
            if not (iter is None):
                count = iter if count > iter else count
            x_values = cur_results.keys()[:count]
            y_values = cur_results.values()[:count]
            pp = prio
            if prio == 'uniform':
                prio = 'DH+UF'
            elif prio == 'prob':
                prio = 'DH+Prob'
            elif prio == 'random':
                prio = 'Random'
            elif prio == 'tensorfuzz':
                prio = 'TensorFuzz'
            elif prio == 'deeptest':
                prio = 'DeepTest'

            print(cri, prio, x_values[-1],y_values[-1])

            last_coverage[(cri,pp)] = y_values[-1]

            if type == 'seedattack':
                y_values =  [x / 1000 for x in y_values]
            ax.plot(x_values, y_values, label=prio)
            if type == 'seedattack':
                vals = ax.get_yticks()
                ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
            ax.legend(loc="upper left")
        ax_num += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='coverage guided fuzzing')
    parser.add_argument('-i',help='root output dir for results used as input dir here')
    # parser.add_argument('-m',help='the chosen model')
    # parser.add_argument('-i', help='input seed dir')
    parser.add_argument('-iterations', help='seed output', type=int)
    parser.add_argument('-o', help='figure name to output(no postfix)')
    parser.add_argument('-type',choices=['coverage', 'seedattack'], default='coverage',help='seed output')
    args = parser.parse_args()

    input = args.i
    # model = args.m
    base_path = input #os.path.join(input,model)

    parent = os.path.dirname(args.o)
    if not os.path.exists(parent):
        try:
            os.makedirs(parent)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


    strategies = os.listdir(base_path)
    criteria_for_strategy = defaultdict()
    strategy_for_criteria = defaultdict(list)

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

    get_data(base_path,effective_criteria_dict,'', axs.flatten(), args.iterations,args.type)

    fig.set_size_inches(18,5)
    plt.savefig("{0}.pdf".format(args.o), format='pdf')


    import csv
    full_criteria = ['nc','kmnc', 'nbc', 'snac', 'tknc', 'random']
    head_row = ['strategy\\criteria'] + full_criteria
    csv_rows = [head_row]
    for i in strategies:
        line = [i]
        for j in full_criteria:
            if (j,i) not in last_coverage:
                line.append("N/A")
                continue
            line.append(last_coverage[(j,i)])
        csv_rows.append(line)

    with open('{0}.csv'.format(args.o), 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_rows)
