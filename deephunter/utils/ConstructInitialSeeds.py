#!/usr/bin/env python2.7
import argparse
import os
import sys
import random
from keras.datasets import mnist

from keras.models import load_model
import numpy as np
from keras.datasets import cifar10
sys.path.append('../')
model_weight_path = {
    'vgg16': "./profile/cifar10/models/vgg16.h5",
    'resnet20': "./profile/cifar10/models/resnet.h5",
    'lenet1': "./profile/mnist/models/lenet1.h5",
    'lenet4': "./profile/mnist/models/lenet4.h5",
    'lenet5': "./profile/mnist/models/lenet5.h5"
}


def color_preprocessing(x_test):
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_test
def preprocessing_test_batch(x_test):
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test

def createBatch(x_batch, batch_size, output_path, prefix):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    batch_num = len(x_batch) / batch_size
    batches = np.split(x_batch, batch_num, axis=0)
    for i, batch in enumerate(batches):
        test = np.append(batch, batch, axis=0)
        saved_name = prefix + str(i) + '.npy'
        np.save(os.path.join(output_path, saved_name), test)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='control experiment')


    parser.add_argument('-model_type', help='Model type', choices=['lenet1','lenet4','lenet5','resnet20', 'vgg16'], default='lenet5')
    parser.add_argument('-output_path', help='Out path')
    parser.add_argument('-batch_size', type=int, help='Number of images in one batch', default=1)
    parser.add_argument('-batch_num', type=int, help='Number of batches', default=30)
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.model_type in ['lenet1','lenet4','lenet5']:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        batch = preprocessing_test_batch(x_test)
        model = load_model(model_weight_path[args.model_type])
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        batch = color_preprocessing(x_test)
        model = load_model(model_weight_path[args.model_type])


    num_in_each_class = args.batch_num / 10
    result = np.argmax(model.predict(batch),axis=1)
    new_label = np.reshape(y_test, result.shape)
    idx_good = np.where(new_label == result)[0]


    for cl in range(10):
        cl_indexes  = [i for i in idx_good if new_label[i] == cl]
        selected = random.sample(cl_indexes, num_in_each_class)
        createBatch(x_test[selected], args.batch_size, args.output_path, str(cl)+'_')
    print('finish')