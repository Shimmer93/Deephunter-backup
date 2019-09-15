
from os.path import isfile, join

import os

from PIL import Image

import numpy as np

import argparse



def check(queue, imag_path):
    if not os.path.exists(imag_path):
        os.makedirs(imag_path)

    total = 0
    lst = os.listdir(queue)

    for f in lst:
        file = os.path.abspath(join(queue, f))
        if not isfile(file):
            continue


        ori_batch = np.load(file)
        sp = ori_batch.shape
        if len(sp) == 4 and sp[-4] > 1:
            ori_batch = ori_batch[1]
        m = sp[-3]
        n = sp[-2]
        q = sp[-1]



        if q == 1:
            x = np.reshape( ori_batch, (m,n))
        else:
            x = np.reshape(ori_batch, (m, n, q))

        x = np.uint8(x)

        img0 = Image.fromarray(x)
        img0.save(imag_path + '/'+f+ ".png")
        total += 1






import shutil
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-i', help='Npy file path')
    parser.add_argument('-o', help='Png outputs')

    args = parser.parse_args()
    if os.path.exists(args.o):
        shutil.rmtree(args.o)
    os.makedirs(args.o)

    check(args.i, args.o)
    print('finish')


