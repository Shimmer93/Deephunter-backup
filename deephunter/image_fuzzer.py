from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import argparse, pickle
import shutil

from keras.models import load_model
import tensorflow as tf
import os

sys.path.append('../')

import keras
from keras import Input
from deephunter.coverage import Coverage

from keras.applications import MobileNet, VGG19, ResNet50
from keras.applications.vgg16 import preprocess_input

import random
import time
import numpy as np
from PIL import Image
from deephunter.image_queue import ImageInputCorpus, TensorInputCorpus
from deephunter.fuzzone import build_fetch_function

from lib.queue import Seed
from lib.fuzzer import Fuzzer

from deephunter.mutators import Mutators
from keras.utils.generic_utils import CustomObjectScope

#_, (x_test, y_test) = keras.datasets.cifar10.load_data()
#x_test=x_test/255.0
#x_test=x_test.reshape(10000,32,32,3)
#y_test=y_test.reshape(-1)
#yy = np.zeros((10000, 10))
#for i in range(10000):
#    yy[i][y_test[i]] = 1

def imagenet_preprocessing(input_img_data):
    temp = np.copy(input_img_data)
    temp = np.float32(temp)
    qq = preprocess_input(temp)
    return qq

def imgnt_preprocessing(x_test):
    return x_test

def mnist_preprocessing(x):
    x = x.reshape(x.shape[0], 28, 28)
    new_x = []
    for img in x:
        img = Image.fromarray(img.astype('uint8'), 'L')
        img = img.resize(size=(32, 32))
        img = np.asarray(img).astype(np.float32) / 255.0 - 0.1306604762738431
        new_x.append(img)
    new_x = np.stack(new_x)
    new_x = np.expand_dims(new_x, axis=-1)
    return new_x


def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp


model_weight_path = {
    'vgg16': "./profile/cifar10/models/vgg16.h5",
    'resnet20': "/data/dnntest/zpengac/models/resnet/cifar10_resnet20v1_keras_deephunter_prob_kmnc2.h5",
    'lenet1': "./profile/mnist/models/lenet1.h5",
    'lenet4': "./profile/mnist/models/lenet4.h5",
    'lenet5': "/data/dnntest/zpengac/models/lenet/mnist_lenet5_keras_32_py2.h5"
}

model_profile_path = {
    'vgg16': "./profile/cifar10/profiling/vgg16/0_50000.pickle",
    'resnet20': "/data/dnntest/zpengac/deephunter/deephunter/profile/cifar10_resnet20v1_keras_deephunter_prob_kmnc2.pickle",
    'lenet1': "./profile/mnist/profiling/lenet1/0_60000.pickle",
    'lenet4': "./profile/mnist/profiling/lenet4/0_60000.pickle",
    'lenet5': "/data/dnntest/zpengac/deephunter/deephunter/profile/mnist_lenet5_32_py2.pickle",
    'mobilenet': "./profile/imagenet/profiling/mobilenet_merged.pickle",
    'vgg19': "./profile/imagenet/profiling/vgg19_merged.pickle",
    'resnet50': "./profile/imagenet/profiling/resnet50_merged.pickle"
}

preprocess_dic = {
    'vgg16': cifar_preprocessing,
    'resnet20': cifar_preprocessing,
    'lenet1': mnist_preprocessing,
    'lenet4': mnist_preprocessing,
    'lenet5': mnist_preprocessing,
    'mobilenet': imagenet_preprocessing,
    'vgg19': imagenet_preprocessing,
    'resnet50': imgnt_preprocessing
}

shape_dic = {
    'vgg16': (32, 32, 3),
    'resnet20': (32, 32, 3),
    'lenet1': (28, 28, 1),
    'lenet4': (28, 28, 1),
    'lenet5': (32, 32, 1),
    'mobilenet': (224, 224, 3),
    'vgg19': (224, 224, 3),
    'resnet50': (256, 256, 3)
}
metrics_para = {
    'kmnc': 1000,
    'bknc': 10,
    'tknc': 10,
    'nbc': 10,
    'newnc': 10,
    'nc': 0.75,
    'fann': 1.0,
    'snac': 10
}
execlude_layer_dic = {
    'vgg16': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'resnet20': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet1': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet4': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'mobilenet': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout',
                  'bn', 'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'vgg19': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
              'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'resnet50': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
                 'reshape', 'relu', 'pool', 'concat', 'add', 'res4', 'res5']
}


def metadata_function(meta_batches):
    return meta_batches


def image_mutation_function(batch_num):
    # Given a seed, randomly generate a batch of mutants
    def func(seed):
        return Mutators.image_random_mutate(seed, batch_num)

    return func


def objective_function(seed, names):
    metadata = seed.metadata
    ground_truth = seed.ground_truth
    assert (names is not None)
    results = []
    if len(metadata) == 1:
        # To check whether it is an adversarial sample
        if metadata[0] != ground_truth:
            results.append('')
    else:
        # To check whether it has different results between original model and quantized model
        # metadata[0] is the result of original model while metadata[1:] is the results of other models.
        # We use the string adv to represent different types;
        # adv = '' means the seed is not an adversarial sample in original model but has a different result in the
        # quantized version.  adv = 'a' means the seed is adversarial sample and has a different results in quan models.
        if metadata[0] == ground_truth:
            adv = ''
        else:
            adv = 'a'
        count = 1
        while count < len(metadata):
            if metadata[count] != metadata[0]:
                results.append(names[count] + adv)
            count += 1

    # results records the suffix for the name of the failed tests
    return results


def iterate_function(names):
    def func(queue, root_seed, parent, mutated_coverage_list, mutated_data_batches, mutated_metadata_list,
             objective_function):

        ref_batches, batches, cl_batches, l0_batches, linf_batches = mutated_data_batches

        successed = False
        bug_found = False
        # For each mutant in the batch, we will check the coverage and whether it is a failed test
        for idx in range(len(mutated_coverage_list)):

            input = Seed(cl_batches[idx], mutated_coverage_list[idx], root_seed, parent, mutated_metadata_list[:, idx],
                         parent.ground_truth, l0_batches[idx], linf_batches[idx])

            # The implementation for the isFailedTest() in Algorithm 1 of the paper
            results = objective_function(input, names)

            if len(results) > 0:
                # We have find the failed test and save it in the crash dir.
                for i in results:
                    queue.save_if_interesting(input, batches[idx], True, suffix=i)
                bug_found = True
            else:

                new_img = np.append(ref_batches[idx:idx + 1], batches[idx:idx + 1], axis=0)
                # If it is not a failed test, we will check whether it has a coverage gain
                result = queue.save_if_interesting(input, new_img, False)
                successed = successed or result
        return bug_found, successed

    return func


def dry_run(indir, fetch_function, coverage_function, queue):
    seed_lis = os.listdir(indir)
    # Read each initial seed and analyze the coverage
    for seed_name in seed_lis:
        tf.logging.info("Attempting dry run with '%s'...", seed_name)
        path = os.path.join(indir, seed_name)
        img = np.load(path)
        # Each seed will contain two images, i.e., the reference image and mutant (see the paper)
        input_batches = img[1:2]
        # Predict the mutant and obtain the outputs
        # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
        coverage_batches, metadata_batches = fetch_function((0, input_batches, 0, 0, 0))
        # Based on the output, compute the coverage information
        coverage_list = coverage_function(coverage_batches)
        metadata_list = metadata_function(metadata_batches)
        # Create a new seed
        input = Seed(0, coverage_list[0], seed_name, None, metadata_list[0][0], metadata_list[0][0])
        new_img = np.append(input_batches, input_batches, axis=0)
        # Put the seed in the queue and save the npy file in the queue dir
        queue.save_if_interesting(input, new_img, False, True, seed_name)


if __name__ == '__main__':

    start_time = time.time()

    tf.logging.set_verbosity(tf.logging.INFO)
    random.seed(time.time())

    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')

    parser.add_argument('-i', help='input seed directory')
    parser.add_argument('-o', help='output directory')

    parser.add_argument('-model', help="target model to fuzz", choices=['vgg16', 'resnet20', 'mobilenet', 'vgg19',
                                                                        'resnet50', 'lenet1', 'lenet4', 'lenet5'], default='lenet5')
    parser.add_argument('-criteria', help="set the criteria to guide the fuzzing",
                        choices=['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc', 'fann'], default='kmnc')
    parser.add_argument('-batch_num', help="the number of mutants generated for each seed", type=int, default=20)
    parser.add_argument('-max_iteration', help="maximum number of fuzz iterations", type=int, default=10000000)
    parser.add_argument('-metric_para', help="set the parameter for different metrics", type=float)
    parser.add_argument('-quantize_test', help="fuzzer for quantization", default=0, type=int)
    # parser.add_argument('-ann_threshold', help="Distance below which we consider something new coverage.", type=float,
    #                     default=1.0)
    parser.add_argument('-quan_model_dir', help="directory including the quantized models for testing")
    parser.add_argument('-random', help="whether to adopt random testing strategy", type=int, default=0)
    parser.add_argument('-select', help="test selection strategy",
                        choices=['uniform', 'tensorfuzz', 'deeptest', 'prob'], default='prob')

    args = parser.parse_args()

    img_rows, img_cols = 256, 256
    input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)

    # Get the layers which will be excluded during the coverage computation
    exclude_layer_list = execlude_layer_dic[args.model]

    # Create the output directory including seed queue and crash dir, it is like AFL
    if os.path.exists(args.o):
        shutil.rmtree(args.o)
    os.makedirs(os.path.join(args.o, 'queue'))
    os.makedirs(os.path.join(args.o, 'crashes'))

    # Load model. For ImageNet, we use the default models from Keras framework.
    # For other models, we load the model from the h5 file.
    model = None
    if args.model == 'mobilenet':
        model = MobileNet(input_tensor=input_tensor)
    elif args.model == 'vgg19':
        model = VGG19(input_tensor=input_tensor, input_shape=input_shape)
    elif args.model == 'resnet50':
        model = ResNet50(input_tensor=input_tensor)
    else:
        model = load_model(model_weight_path[args.model])

    # Get the preprocess function based on different dataset
    preprocess = preprocess_dic[args.model]

    # Load the profiling information which is needed by the metrics in DeepGauge
    profile_dict = pickle.load(open(model_profile_path[args.model], 'rb'))

    # Load the configuration for the selected metrics.
    if args.metric_para is None:
        cri = metrics_para[args.criteria]
    elif args.criteria == 'nc':
        cri = args.metric_para
    else:
        cri = int(args.metric_para)

    # The coverage computer
    coverage_handler = Coverage(model=model, criteria=args.criteria, k=cri,
                                profiling_dict=profile_dict, exclude_layer=exclude_layer_list)

    # The log file which records the plot data after each iteration of the fuzzing
    plot_file = open(os.path.join(args.o, 'plot.log'), 'a+')

    # If testing for quantization, we will load the quantized versions
    # fetch_function is to perform the prediction and obtain the outputs of each layers
    if args.quantize_test == 1:
        model_names = os.listdir(args.quan_model_dir)
        model_paths = [os.path.join(args.quan_model_dir, name) for name in model_names]
        if args.model == 'mobilenet':
            import keras

            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                    'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                models = [load_model(m) for m in model_paths]
        else:
            models = [load_model(m) for m in model_paths]
        fetch_function = build_fetch_function(coverage_handler, preprocess, models)
        model_names.insert(0, args.model)
    else:
        fetch_function = build_fetch_function(coverage_handler, preprocess)
        model_names = [args.model]

    # Like AFL, dry_run will run all initial seeds and keep all initial seeds in the seed queue
    dry_run_fetch = build_fetch_function(coverage_handler, preprocess)

    # The function to update coverage
    coverage_function = coverage_handler.update_coverage
    # The function to perform the mutation from one seed
    mutation_function = image_mutation_function(args.batch_num)

    # The seed queue
    if args.criteria == 'fann':
        queue = TensorInputCorpus(args.o, args.random, args.select, cri, "kdtree")
    else:
        queue = ImageInputCorpus(args.o, args.random, args.select, coverage_handler.total_size, args.criteria)

    # Perform the dry_run process from the initial seeds
    dry_run(args.i, dry_run_fetch, coverage_function, queue)

    # For each seed, compute the coverage and check whether it is a "bug", i.e., adversarial example
    image_iterate_function = iterate_function(model_names)

    # The main fuzzer class
    fuzzer = Fuzzer(queue, coverage_function, metadata_function, objective_function, mutation_function, fetch_function,
                    image_iterate_function, args.select)

    # The fuzzing process
    fuzzer.loop(args.max_iteration)
    
    #x_test = cifar_preprocessing(x_test)
    #model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    #print(model.metrics_names)
    #print(model.evaluate(x_test, yy, verbose=1))

    spent_time = time.time() - start_time
    print('finish',  spent_time)
    f = open('time.txt', 'a+')
    f.write(args.model + '\t' + args.criteria + '\t' + args.select + '\t' + str(spent_time) + '\n')
    f.close()


