import math

import matplotlib.pyplot as plt
import numpy as np

from cv2 import Canny
from PIL import Image
from numba import jit, prange


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def nearest_neighbour(array_in, array_out, width_out, height_out, _scale):
    for W in prange(height_out):
        for K in prange(width_out):
            array_out[W, K] = array_in[math.floor(W / _scale), math.floor(K / _scale)]
    return array_out


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def interpolate_bilinear(array_in, width_in, height_in, array_out, width_out, height_out):
    w_step = height_in / height_out
    k_step = width_in / width_out
    for W in prange(height_out):
        w_value = w_step * W - 0.5 + w_step / 2
        for K in prange(width_out):
            k_value = k_step * K - 0.5 + k_step / 2
            w_prev = math.floor(w_value)
            w_next = math.ceil(w_value)
            k_prev = math.floor(k_value)
            k_next = math.ceil(k_value)

            w_next = min(w_next, height_in - 1)
            k_next = min(k_next, width_in - 1)

            w_prev = max(w_prev, 0)
            k_prev = max(k_prev, 0)

            wV = max(w_value, 0) % 1
            kV = max(k_value, 0) % 1

            array_out[W, K] = (array_in[w_prev, k_prev] * (1 - kV) * (1 - wV)
                               + array_in[w_next, k_prev] * (1 - kV) * wV
                               + array_in[w_prev, k_next] * kV * (1 - wV)
                               + array_in[w_next, k_next] * kV * wV)

    return array_out


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def mean_decrease(array_in, array_out, width_out, height_out, _scale):
    s = 1 / _scale
    for W in prange(height_out):
        for K in prange(width_out):
            w = list(range(int(W * s - s), int(W * s + s)))
            k = list(range(int(K * s - s), int(K * s + s)))
            w = [x for x in w if x >= 0]
            k = [x for x in k if x >= 0]

            array = []
            for i in w:
                for j in k:
                    array.append(array_in[i, j])

            res = []
            for i in range(3):
                suma = 0
                for j in range(len(array)):
                    suma += array[j][i]
                res.append(suma / len(array))

            array_out[W, K] = res
    return array_out


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def mean_weighted_decrease(array_in, array_out, width_out, height_out, _scale):
    s = 1 / _scale
    for W in prange(height_out):
        for K in prange(width_out):
            w = list(range(int(W * s - s), int(W * s + s)))
            k = list(range(int(K * s - s), int(K * s + s)))
            w = [x for x in w if x >= 0]
            k = [x for x in k if x >= 0]

            array = []
            for i in w:
                for j in k:
                    array.append(array_in[i, j])

            weights = []
            sumOFWeights = 0
            X = len(w)
            Y = len(k)
            x_sr = (X - 1) / 2
            y_sr = (Y - 1) / 2
            base_value = x_sr ** 2 + y_sr ** 2 + 1
            for x in range(X):
                for y in range(Y):
                    weights.append(base_value - ((x - x_sr) ** 2 + (y - y_sr) ** 2))
                    sumOFWeights += base_value - ((x - x_sr) ** 2 + (y - y_sr) ** 2)

            res = []
            for i in range(3):
                suma = 0
                for j in range(len(array)):
                    suma += array[j][i] * weights[j]
                res.append(suma / sumOFWeights)

            array_out[W, K] = res
    return array_out


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def median_decrease(array_in, array_out, width_out, height_out, _scale):
    s = 1 / _scale
    for W in prange(height_out):
        for K in prange(width_out):
            w = list(range(int(W * s - s), int(W * s + s)))
            k = list(range(int(K * s - s), int(K * s + s)))
            w = [x for x in w if x >= 0]
            k = [x for x in k if x >= 0]

            array = []
            for i in w:
                for j in k:
                    array.append(array_in[i, j])

            killmeplease = []
            res = np.zeros((len(array)))
            for i in range(3):
                for j in range(len(array)):
                    res[j] = array[j][i]
                killmeplease.append(np.median(res))
            array_out[W, K] = killmeplease
    return array_out


def runIncrease():
    images_to_increase = [
        "data/0001.jpg",
        # "data/0002.jpg",
        "data/0003.jpg",
        # "data/0004.jpg",
        # "data/0005.jpg",
        # "data/0006.jpg",
        # "data/0007.jpg",
        # "data/0008.tif",
        # "data/0009.jpg",
    ]
    scale_increase = 2

    for image in images_to_increase:
        OG = plt.imread(image)

        plt.subplot(2, 3, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(OG)

        plt.subplot(2, 3, 4)
        plt.xticks([])
        plt.yticks([])
        edges = Canny(OG, 100, 200)
        plt.imshow(edges, cmap='gray')

        og_height = OG.shape[0]
        og_width = OG.shape[1]

        new_height_increase = math.floor(OG.shape[0] * scale_increase)
        new_width_increase = math.floor(OG.shape[1] * scale_increase)

        if (len(OG.shape) < 3):
            nn_resize = np.zeros((new_height_increase, new_width_increase))
            id_resize = np.zeros((new_height_increase, new_width_increase))
        else:
            nn_resize = np.zeros((new_height_increase, new_width_increase, OG.shape[2]), dtype=np.uint8)
            id_resize = np.zeros((new_height_increase, new_width_increase, OG.shape[2]), dtype=np.uint8)

        nn_resize = nearest_neighbour(OG, nn_resize, new_width_increase, new_height_increase, scale_increase)

        plt.subplot(2, 3, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(nn_resize)

        plt.subplot(2, 3, 5)
        plt.xticks([])
        plt.yticks([])
        edges = Canny(nn_resize, 100, 200)
        plt.imshow(edges, cmap='gray')

        id_resize = interpolate_bilinear(OG, og_width, og_height, id_resize, new_width_increase, new_height_increase)

        plt.subplot(2, 3, 3)
        plt.imshow(id_resize)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 6)
        plt.xticks([])
        plt.yticks([])
        edges = Canny(id_resize, 100, 200)
        plt.imshow(edges, cmap='gray')

        plt.show()

        im = Image.fromarray(nn_resize)
        im.save("nn_" + image)

        im = Image.fromarray(id_resize)
        im.save("id_" + image)


def runDecrease():

    images_to_decrease = [
        # "data/0001.jpg",
        # "data/0002.jpg",
        # "data/0003.jpg",
        # "data/0004.jpg",
        # "data/0005.jpg",
        # "data/0006.jpg",
        # "data/0007.jpg",
        # "data/0008.tif",
        # "data/0009.jpg",
        "data/0010.jpg",
    ]

    scale_decrease = 0.05
    for image in images_to_decrease:

        OG = plt.imread(image)

        plt.figure(dpi=700)
        plt.subplot(2, 4, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(OG)

        plt.subplot(2, 4, 5)
        plt.xticks([])
        plt.yticks([])
        edges = Canny(OG, 100, 200)
        plt.imshow(edges, cmap='gray')

        og_height = OG.shape[0]
        og_width = OG.shape[1]

        new_height_decrease = int(og_height * scale_decrease)
        new_width_decrease = int(og_width * scale_decrease)

        if len(OG.shape) < 3:
            mean_resize = np.zeros((new_height_decrease, new_width_decrease))
            mean_weighted_resize = np.zeros((new_height_decrease, new_width_decrease))
            median_resize = np.zeros((new_height_decrease, new_width_decrease))
        else:
            mean_resize = np.zeros((new_height_decrease, new_width_decrease, OG.shape[2]), dtype=np.uint8)
            mean_weighted_resize = np.zeros((new_height_decrease, new_width_decrease, OG.shape[2]), dtype=np.uint8)
            median_resize = np.zeros((new_height_decrease, new_width_decrease, OG.shape[2]), dtype=np.uint8)

        mean_resize = mean_decrease(OG, mean_resize, new_width_decrease, new_height_decrease, scale_decrease)

        plt.subplot(2, 4, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(mean_resize)

        plt.subplot(2, 4, 6)
        plt.xticks([])
        plt.yticks([])
        edges = Canny(mean_resize, 100, 200)
        plt.imshow(edges, cmap='gray')

        mean_weighted_resize = mean_weighted_decrease(OG, mean_weighted_resize, new_width_decrease, new_height_decrease,
                                                      scale_decrease)

        plt.subplot(2, 4, 3)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(mean_weighted_resize)

        plt.subplot(2, 4, 7)
        plt.xticks([])
        plt.yticks([])
        edges = Canny(mean_weighted_resize, 100, 200)
        plt.imshow(edges, cmap='gray')

        median_resize = median_decrease(OG, median_resize, new_width_decrease, new_height_decrease, scale_decrease)

        plt.subplot(2, 4, 4)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(median_resize)

        plt.subplot(2, 4, 8)
        plt.xticks([])
        plt.yticks([])
        edges = Canny(median_resize, 100, 200)
        plt.imshow(edges, cmap='gray')

        plt.show()

        im = Image.fromarray(mean_resize)
        im.save("mean_" + image)

        im = Image.fromarray(mean_weighted_resize)
        im.save("mean_weighted_" + image)

        im = Image.fromarray(median_resize)
        im.save("median_" + image)


if __name__ == '__main__':
    # runIncrease()
    runDecrease()
