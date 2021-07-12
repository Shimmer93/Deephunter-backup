import os
from tensorflow import keras
import numpy as np
from PIL import Image
from tqdm import tqdm
import re

def mnist_preprocessing(x):
    if x.shape == (28, 28, 1) or x.shape == (28, 28):
        x = x.reshape(1, 28, 28)
    else:
        x = x.reshape(x.shape[0], 28, 28)
    new_x = []
    for img in x:
        img = Image.fromarray(img.astype('uint8'), 'L')
        img = img.resize(size=(32, 32))
        new_x.append(img)
    new_x = np.stack(new_x)
    new_x = np.expand_dims(new_x, axis=-1)
    return new_x

s1s = ['prob', 'uniform']
s2s = ['kmnc', 'nbc']
for s1 in s1s:
    for s2 in s2s:
        input_path = '/data/dnntest/zpengac/deephunter/test_seeds/dnntest_cifar'
        output_path = '/data/dnntest/zpengac/deephunter/deephunter/dnntest_resnet20_retrain2/{s1}/{s2}/0/queue'.format(s1=s1, s2=s2)
        crash_path = '/data/dnntest/zpengac/deephunter/deephunter/dnntest_resnet20_retrain2/{s1}/{s2}/0/crashes'.format(s1=s1, s2=s2)

        result_src_l2 = 0
        result_src_linf = 0
        srcs = []
        l2_dict = {}
        linf_dict = {}

        files = os.listdir(output_path)
        files.sort(key=lambda f:int(f[3:9]))
        for f in tqdm(files):
            src = re.search(r'(?<=src:)\S+?(?=(:|\.))', f).group(0)
            if '_' in src:
                srcs.append(src)
                #src_dict[src] = i
            else:
                src = srcs[int(src)]
                srcs.append(src)
                #src_dict[src] = i
            imgs = np.load(os.path.join(output_path, f))
            src_imgs = np.load(os.path.join(input_path, src+'.npy'))
            #imgs = mnist_preprocessing(imgs)
            #src_imgs = mnist_preprocessing(src_imgs)
            src_l2 = np.sqrt(np.sum(np.square(src_imgs[0].astype('float32') - imgs[0]).astype('float32')))
            src_linf = np.max(np.abs(src_imgs[0].astype('float32') - imgs[0].astype('float32')))
            if src not in l2_dict or l2_dict[src] == 0 or src_l2 < l2_dict[src]:
                l2_dict[src] = src_l2
            if src not in linf_dict or linf_dict[src] == 0 or src_linf < linf_dict[src]:
                linf_dict[src] = src_linf

        crashes = os.listdir(crash_path)
        crashes.sort(key=lambda f:int(f[3:9]))
        first = True
        for f in tqdm(crashes):
            src = re.search(r'(?<=src:)\S+?(?=(:|\.))', f).group(0)
            src = srcs[int(src)]
            img = np.load(os.path.join(crash_path, f))
            src_img = np.load(os.path.join(input_path, src+'.npy'))
            #img = mnist_preprocessing(img)
            #src_img = mnist_preprocessing(src_img)
            #if first:
            #    first = False
            #    img_file = Image.fromarray(imgs[0].reshape(32,32).astype('uint8'), 'L')
            #    img_file.save('example_{s1}_{s2}.bmp'.format(s1=s1,s2=s2))
            #    orig_file = Image.fromarray(src_imgs[0].reshape(32,32).astype('uint8'), 'L')
            #    orig_file.save('example_orig_{s1}_{s2}.bmp'.format(s1=s1,s2=s2))
            src_l2 = np.sqrt(np.sum(np.square(src_img.astype('float32') - img.astype('float32'))))
            src_linf = np.max(np.abs(src_img.astype('float32') - img.astype('float32')))
            if src not in l2_dict or l2_dict[src] == 0 or src_l2 < l2_dict[src]:
                l2_dict[src] = src_l2
            if src not in linf_dict or linf_dict[src] == 0 or src_linf < linf_dict[src]:
                linf_dict[src] = src_linf

        for _, v in l2_dict.items():
            result_src_l2 += v
        for _, v in linf_dict.items():
            result_src_linf += v
        result_src_l2 /= 1000.0
        result_src_linf /= 1000.0
        '''
        for k, v in src_dict.items():
            imgs = np.load(os.path.join(output_path, files[v]))
            src_imgs = np.load(os.path.join(input_path, k+'.npy'))
            src_l2 = np.sqrt(np.sum(np.square(src_imgs[0] - imgs[1])))
            src_linf = np.max(np.abs(src_imgs[0] - imgs[1]))
            result_src_l2 += src_l2
            result_src_linf += src_linf
        '''
        '''
        for f in tqdm(files):
            src = re.search(r'(?<=src:)\S+?(?=(:|\.))', f).group(0)
            if '_' in src:
                srcs.append(src)
            else:
                src = srcs[int(src)]
                srcs.append(src)
            imgs = np.load(os.path.join(output_path, f))
            l2 = np.sqrt(np.sum(np.square(imgs[0] - imgs[1])))
            linf = np.max(np.abs(imgs[0] - imgs[1]))
            result_l2 += l2
            result_linf += linf

            src_imgs = np.load(os.path.join(input_path, src+'.npy'))
            src_l2 = np.sqrt(np.sum(np.square(src_imgs[0] - imgs[1])))
            src_linf = np.max(np.abs(src_imgs[0] - imgs[1]))
            emm_l2 = np.sqrt(np.sum(np.square(src_imgs[0] - src_imgs[1])))
            result_src_l2 += src_l2
            result_src_linf += src_linf
            result_emm_l2 += emm_l2

            i += 1
        '''
        print(s1, s2)
        print('l2:', result_src_l2)
        print('linf:', result_src_linf)