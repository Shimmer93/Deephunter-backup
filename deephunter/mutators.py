from __future__ import print_function
import sys
import cv2
import numpy as np
import random
import time
import copy
reload(sys)
sys.setdefaultencoding('utf8')


# keras 1.2.2 tf:1.2.0
class Mutators():
    def image_translation(img, params):

        rows, cols, ch = img.shape
        # rows, cols = img.shape

        # M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
        M = np.float32([[1, 0, params], [0, 1, params]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def image_scale(img, params):

        # res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
        rows, cols, ch = img.shape
        res = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
        res = res.reshape((res.shape[0],res.shape[1],ch))
        y, x, z = res.shape
        if params > 1:  # need to crop
            startx = x // 2 - cols // 2
            starty = y // 2 - rows // 2
            return res[starty:starty + rows, startx:startx + cols]
        elif params < 1:  # need to pad
            sty = (rows - y) / 2
            stx = (cols - x) / 2
            return np.pad(res, [(sty, rows - y - sty), (stx, cols - x - stx), (0, 0)], mode='constant',
                          constant_values=0)
        return res

    def image_shear(img, params):
        rows, cols, ch = img.shape
        # rows, cols = img.shape
        factor = params * (-1.0)
        M = np.float32([[1, factor, 0], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def image_rotation(img, params):
        rows, cols, ch = img.shape
        # rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
        dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_AREA)
        return dst

    def image_contrast(img, params):
        alpha = params
        new_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha
        # new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

        return new_img

    def image_brightness(img, params):
        beta = params
        new_img = cv2.add(img, beta)  # new_img = img*alpha + beta
        return new_img

    def image_blur(img, params):

        # print("blur")
        blur = []
        if params == 1:
            blur = cv2.blur(img, (3, 3))
        if params == 2:
            blur = cv2.blur(img, (4, 4))
        if params == 3:
            blur = cv2.blur(img, (5, 5))
        if params == 4:
            blur = cv2.GaussianBlur(img, (3, 3), 0)
        if params == 5:
            blur = cv2.GaussianBlur(img, (5, 5), 0)
        if params == 6:
            blur = cv2.GaussianBlur(img, (7, 7), 0)
        if params == 7:
            blur = cv2.medianBlur(img, 3)
        if params == 8:
            blur = cv2.medianBlur(img, 5)
        # if params == 9:
        #     blur = cv2.blur(img, (6, 6))
        if params == 9:
            blur = cv2.bilateralFilter(img, 6, 50, 50)
            # blur = cv2.bilateralFilter(img, 9, 75, 75)
        return blur

    def image_pixel_change(img, params):
        # random change 1 - 5 pixels from 0 -255
        img_shape = img.shape
        img1d = np.ravel(img)
        arr = np.random.randint(0, len(img1d), params)
        for i in arr:
            img1d[i] = np.random.randint(0, 256)
        new_img = img1d.reshape(img_shape)
        return new_img

    def image_noise(img, params):
        if params == 1:  # Gaussian-distributed additive noise.
            row, col, ch = img.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img + gauss
            return noisy.astype(np.uint8)
        elif params == 2:  # Replaces random pixels with 0 or 1.
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(img)
            # Salt mode
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt))
                      for i in img.shape]
            out[tuple(coords)] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper))
                      for i in img.shape]
            out[tuple(coords)] = 0
            return out
        elif params == 3:  # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
            row, col, ch = img.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = img + img * gauss
            return noisy.astype(np.uint8)















    transformations = [image_translation, image_scale, image_shear, image_rotation,
                       image_contrast, image_brightness, image_blur, image_pixel_change, image_noise]

    # these parameters need to be carefullly considered in the experiment
    # to consider the feedbacks
    params = []
    params.append(list(xrange(-3, 3)))  # image_translation
    params.append(list(map(lambda x: x * 0.1, list(xrange(7, 12)))))  # image_scale
    params.append(list(map(lambda x: x * 0.1, list(xrange(-6, 6)))))  # image_shear
    params.append(list(xrange(-50, 50)))  # image_rotation
    params.append(list(map(lambda x: x * 0.1, list(xrange(5, 13)))))  # image_contrast
    params.append(list(xrange(-20, 20)))  # image_brightness
    params.append(list(xrange(1, 10)))  # image_blur
    params.append(list(xrange(1, 10)))  # image_pixel_change
    params.append(list(xrange(1, 4)))  # image_noise

    classA = [7, 8]  # pixel value transformation
    classB = [0, 1, 2, 3, 4, 5, 6] # Affine transformation
    @staticmethod
    def mutate_one(ref_img, img, cl, l0_ref, linf_ref, try_num=50):

        # ref_img is the reference image, img is the seed

        # cl means the current state of transformation
        # 0 means it can select both of Affine and Pixel transformations
        # 1 means it only select pixel transformation because an Affine transformation has been used before

        # l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
        # between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1}) in Equation 2 of the paper

        # tyr_num is the maximum number of trials in Algorithm 2


        x, y, z = img.shape

        # a, b is the alpha and beta in Equation 1 in the paper
        a = 0.02
        b = 0.20

        # l0: alpha * size(s), l_infinity: beta * 255 in Equation 1
        l0 = int(a * x * y * z)
        l_infinity = int(b * 255)

        ori_shape = ref_img.shape
        for ii in range(try_num):
            random.seed(time.time())
            if cl == 0:  # 0: can choose class A and B
                tid = random.sample(Mutators.classA + Mutators.classB, 1)[0]
                # Randomly select one transformation   Line-7 in Algorithm2
                transformation = Mutators.transformations[tid]
                params = Mutators.params[tid]
                # Randomly select one parameter Line 10 in Algo2
                param = random.sample(params, 1)[0]

                # Perform the transformation  Line 11 in Algo2
                img_new = transformation(copy.deepcopy(img), param)
                img_new = img_new.reshape(ori_shape)

                if tid in Mutators.classA:
                    sub = ref_img - img_new
                    # check whether it is a valid mutation. i.e., Equation 1 and Line 12 in Algo2
                    l0_ref = np.sum(sub != 0)
                    linf_ref = np.max(abs(sub))
                    if l0_ref < l0 or linf_ref < l_infinity:
                        return ref_img, img_new, 0, 1, l0_ref, linf_ref
                else:  # B, C
                    # If the current transformation is an Affine trans, we will update the reference image and
                    # the transformation state of the seed.
                    ref_img = transformation(copy.deepcopy(ref_img), param)
                    ref_img = ref_img.reshape(ori_shape)
                    return ref_img, img_new, 1, 1, l0_ref, linf_ref
            if cl == 1: # 0: can choose class A
                tid = random.sample(Mutators.classA, 1)[0]
                transformation = Mutators.transformations[tid]
                params = Mutators.params[tid]
                param = random.sample(params, 1)[0]
                img_new = transformation(copy.deepcopy(img), param)
                sub = ref_img - img_new

                # To compute the value in Equation 2 in the paper.
                l0_new = l0_ref +  np.sum(sub != 0)
                linf_new = max(linf_ref , np.max(abs(sub)))

                if  l0_new < l0 or linf_new < l_infinity:
                    return ref_img, img_new, 1, 1, l0_ref, linf_ref
        # Otherwise the mutation is failed. Line 20 in Algo 2
        return ref_img, img, cl, 0, l0_ref, linf_ref

    @staticmethod
    def mutate_without_limitation(ref_img):

        tid = random.sample(Mutators.classA + Mutators.classB, 1)[0]
        transformation = Mutators.transformations[tid]
        ori_shape = ref_img.shape
        params = Mutators.params[tid]
        param = random.sample(params, 1)[0]
        img_new = transformation(ref_img, param)
        img_new = img_new.reshape(ori_shape)
        return img_new
    @staticmethod
    #Algorithm 2
    def image_random_mutate(seed, batch_num):

        test = np.load(seed.fname)
        ref_img = test[0]
        img = test[1]
        cl = seed.clss
        ref_batches = []
        batches = []
        cl_batches = []
        l0_ref_batches = []
        linf_ref_batches = []
        for i in range(batch_num):
            ref_out, img_out, cl_out, changed, l0_ref, linf_ref = Mutators.mutate_one(ref_img, img, cl, seed.l0_ref, seed.linf_ref)
            if changed:
                ref_batches.append(ref_out)
                batches.append(img_out)
                cl_batches.append(cl_out)
                l0_ref_batches.append(l0_ref)
                linf_ref_batches.append(linf_ref)

        return np.asarray(ref_batches), np.asarray(batches), cl_batches, l0_ref_batches, linf_ref_batches

    # TODO: Adapt this function for text mutations
    @staticmethod
    def text_mutate_one(ref_img, img, cl, l0_ref, linf_ref, try_num=50):

        # ref_img is the reference text, text is the seed

        # Note - This variable is from image mutations. Might use it for some kind of text transformation or might ignore it.
        # cl means the current state of transformation
        # 0 means it can select both of Affine and Pixel transformations
        # 1 means it only select pixel transformation because an Affine transformation has been used before

        # l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
        # between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1}) in Equation 2 of the paper

        # tyr_num is the maximum number of trials in Algorithm 2


        x, y, z = img.shape

        # a, b is the alpha and beta in Equation 1 in the paper
        a = 0.02
        b = 0.20

        # l0: alpha * size(s), l_infinity: beta * 255 in Equation 1
        l0 = int(a * x * y * z)
        l_infinity = int(b * 255)

        ori_shape = ref_img.shape
        for ii in range(try_num):
            random.seed(time.time())
            if cl == 0:  # 0: can choose class A and B
                tid = random.sample(Mutators.classA + Mutators.classB, 1)[0]
                # Randomly select one transformation   Line-7 in Algorithm2
                transformation = Mutators.transformations[tid]
                params = Mutators.params[tid]
                # Randomly select one parameter Line 10 in Algo2
                param = random.sample(params, 1)[0]

                # Perform the transformation  Line 11 in Algo2
                img_new = transformation(copy.deepcopy(img), param)
                img_new = img_new.reshape(ori_shape)

                if tid in Mutators.classA:
                    sub = ref_img - img_new
                    # check whether it is a valid mutation. i.e., Equation 1 and Line 12 in Algo2
                    l0_ref = np.sum(sub != 0)
                    linf_ref = np.max(abs(sub))
                    if l0_ref < l0 or linf_ref < l_infinity:
                        return ref_img, img_new, 0, 1, l0_ref, linf_ref
                else:  # B, C
                    # If the current transformation is an Affine trans, we will update the reference image and
                    # the transformation state of the seed.
                    ref_img = transformation(copy.deepcopy(ref_img), param)
                    ref_img = ref_img.reshape(ori_shape)
                    return ref_img, img_new, 1, 1, l0_ref, linf_ref
            if cl == 1: # 0: can choose class A
                tid = random.sample(Mutators.classA, 1)[0]
                transformation = Mutators.transformations[tid]
                params = Mutators.params[tid]
                param = random.sample(params, 1)[0]
                img_new = transformation(copy.deepcopy(img), param)
                sub = ref_img - img_new

                # To compute the value in Equation 2 in the paper.
                l0_new = l0_ref +  np.sum(sub != 0)
                linf_new = max(linf_ref , np.max(abs(sub)))

                if  l0_new < l0 or linf_new < l_infinity:
                    return ref_img, img_new, 1, 1, l0_ref, linf_ref
        # Otherwise the mutation is failed. Line 20 in Algo 2
        return ref_img, img, cl, 0, l0_ref, linf_ref

    def text_random_mutate(seed, batch_num):

        test = np.load(seed.fname)
        ref_text = test[0]
        text = test[1]
        cl = seed.clss
        ref_batches = []
        batches = []
        cl_batches = []
        l0_ref_batches = []
        linf_ref_batches = []
        for i in range(batch_num):
            ref_out, text_out, cl_out, changed, l0_ref, linf_ref = Mutators.text_mutate_one(ref_text, text, cl, seed.l0_ref, seed.linf_ref)
            if changed:
                ref_batches.append(ref_out)
                batches.append(text_out)
                cl_batches.append(cl_out)
                l0_ref_batches.append(l0_ref)
                linf_ref_batches.append(linf_ref)

        return np.asarray(ref_batches), np.asarray(batches), cl_batches, l0_ref_batches, linf_ref_batches


if __name__ == '__main__':
    print("main Test.")
