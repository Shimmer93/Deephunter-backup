'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

import pickle
import pprint
from PIL import Image

from keras import Model
from keras.datasets import mnist,cifar10
from keras.models import load_model
import numpy as np
import collections

import os, sys, errno
from keras import backend as K

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DNNProfile():
    def __init__(self, model, exclude_layer=['input', 'flatten'],
                 only_layer=""):
        '''
        Initialize the model to be tested
        :param threshold: threshold to determine if the neuron is activated
        :param model_name: ImageNet Model name, can have ('vgg16','vgg19','resnet50')
        :param neuron_layer: Only these layers are considered for neuron coverage
        '''
        self.model = model
        self.outputs = []

        print('models loaded')

        # the layers that are considered in neuron coverage computation
        self.layer_to_compute = []
        for layer in self.model.layers:
            if all(ex not in layer.name for ex in exclude_layer):
                self.outputs.append(layer.output)
                self.layer_to_compute.append(layer.name)

        if only_layer != "":
            self.layer_to_compute = [only_layer]

        self.cov_dict = collections.OrderedDict()

        print("* target layer list:", self.layer_to_compute)


        for layer_name in self.layer_to_compute:
            for index in xrange(self.model.get_layer(layer_name).output_shape[-1]):
                # [mean_value_new, squared_mean_value, standard_deviation, lower_bound, upper_bound]
                self.cov_dict[(layer_name, index)] = [0.0, 0.0, 0.0, None, None]



    def count_layers(self):
        return len(self.layer_to_compute)

    def count_neurons(self):
        return len(self.cov_dict.items())

    def count_paras(self):
        return self.model.count_params()

    def update_coverage(self, input_data):

        inp = self.model.input
        functor = K.function([inp] + [K.learning_phase()], self.outputs)
        outputs = functor([input_data, 0])


        for layer_idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[layer_idx]

            # handle the layer output by each data
            # iter is the number of data
            for iter, layer_output in enumerate(layer_outputs):
                if iter % 1000 == 0:
                    print("*layer {0}, current/total iteration: {1}/{2}".format(layer_idx, iter + 1, len(layer_outputs)))

                for neuron_idx in xrange(layer_output.shape[-1]):
                    neuron_output = np.mean(layer_output[..., neuron_idx])



                    profile_data_list = self.cov_dict[(layer_name, neuron_idx)]

                    mean_value = profile_data_list[0]
                    squared_mean_value = profile_data_list[1]

                    lower_bound = profile_data_list[3]
                    upper_bound = profile_data_list[4]

                    total_mean_value = mean_value * iter
                    total_squared_mean_value = squared_mean_value * iter

                    mean_value_new = (neuron_output + total_mean_value) / (iter + 1)
                    squared_mean_value = (neuron_output * neuron_output + total_squared_mean_value) / (iter + 1)


                    standard_deviation = np.math.sqrt(abs(squared_mean_value - mean_value_new * mean_value_new))

                    if (lower_bound is None) and (upper_bound is None):
                        lower_bound = neuron_output
                        upper_bound = neuron_output
                    else:
                        if neuron_output < lower_bound:
                            lower_bound = neuron_output

                        if neuron_output > upper_bound:
                            upper_bound = neuron_output

                    profile_data_list[0] = mean_value_new
                    profile_data_list[1] = squared_mean_value
                    profile_data_list[2] = standard_deviation
                    profile_data_list[3] = lower_bound
                    profile_data_list[4] = upper_bound

                    self.cov_dict[(layer_name, neuron_idx)] = profile_data_list



    def dump(self, output_file):

        print("*profiling neuron size:", len(self.cov_dict.items()))
        for item in self.cov_dict.items():
            print(item)
        pickle_out = open(output_file, "wb")
        pickle.dump(self.cov_dict, pickle_out)
        pickle_out.close()

        print("write out profiling coverage results to ", output_file)
        print("done.")


def preprocessing_test_batch(x_test):

    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
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

cifar_mean = np.array([125.307, 122.95, 113.865])
cifar_std = np.array([62.9932, 62.0887, 66.7048])
def cifar_preprocessing(x):
    new_x = []
    for img in x:
        img = (img.astype('float32') - cifar_mean) / cifar_std
        new_x.append(img)
    new_x = np.stack(new_x)
    return new_x


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise






if __name__ == '__main__':

    # argument parsing
    parser = argparse.ArgumentParser(description='neuron output profiling')
    parser.add_argument('-model', help="target model to profile")
    parser.add_argument('-train', help="training data", choices=['mnist', 'cifar'])
    parser.add_argument('-o', help="output path")
    args = parser.parse_args()

    model = load_model(args.model)
    print('Successfully loaded', model.name)
    model.summary()


    make_sure_path_exists(args.o)

    profiling_dict_result ="{0}.pickle".format(args.o)
    print("profiling output file name {0}".format(profiling_dict_result))


    # get the training data for profiling

    if args.train == 'mnist':
        (x_train, train_label), (x_test, test_label) = mnist.load_data()
        x_train = mnist_preprocessing(x_train)
        profiler = DNNProfile(model)
        print(np.shape(x_train))
        profiler.update_coverage(x_train)
        profiler.dump(profiling_dict_result)
    elif args.train == 'cifar':
        (x_train, train_label), (x_test, test_label) = cifar10.load_data()
        profiler = DNNProfile(model)
        for i in range(50):
            x_sub = cifar_preprocessing(x_train[i*1000:(i+1)*1000])
            print(x_sub.shape)
            profiler.update_coverage(x_sub)
        profiler.dump(profiling_dict_result)
    else:
        print('Please extend the new train data here!')
