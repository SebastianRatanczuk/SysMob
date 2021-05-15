import os
from functools import reduce
from operator import add

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pympler import asizeof
from tqdm import tqdm


def read_image(path):
    image = Image.open(path)
    return np.asarray(image, dtype=int)


def rle_encode(data):
    encoding = []
    prev_char = data[0]
    count = 0

    for i in tqdm(range(data.shape[0])):
        if data[i] != prev_char:
            encoding.append(count)
            encoding.append(prev_char)
            count = 1
            prev_char = data[i]
        else:
            count += 1
    else:
        encoding.append(count)
        encoding.append(prev_char)
        return encoding


def rle_decode(data):
    decode = []
    for i in tqdm(range(0, len(data), 2)):
        count = data[i]
        for _ in range(count):
            decode.append(data[i + 1])
    return np.array(decode).astype(np.int64)


def rle(folder, image):
    im = read_image(folder + image).astype(int)
    Zc = np.array(im)

    Zc = Zc.astype(np.int64)
    ogShape = Zc.shape

    rozwiniete = Zc.flatten()

    encoded = rle_encode(rozwiniete)

    decoded = rle_decode(encoded)

    tmp = np.array(encoded)

    recrated_image = decoded.reshape(ogShape)

    encoded_size = asizeof.asizeof(tmp)
    decoded_size = asizeof.asizeof(decoded)

    reduce = encoded_size / decoded_size * 100

    plt.figure(dpi=600)
    plt.subplot(1, 2, 1)
    plt.title('Obrazek oryginalny rozmiar\n: ' + str(decoded_size))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(Zc, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(
        'Obrazek zdekodowany rle \nrozmiar: ' + str(encoded_size) + '\nProcent obrazu oryginalnego:\n ' + str(
            reduce))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(recrated_image, cmap='gray')

    plt.savefig('plots/' + image + '.png')
    plt.close()


# https://medium.com/analytics-vidhya/transform-an-image-into-a-quadtree-39b3aa6e019a
def split(image):
    half_split = np.array_split(image, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    return reduce(add, res)


def concatenate(n_w, n_e, s_w, s_e):
    top = np.concatenate((n_w, n_e), axis=1)
    bottom = np.concatenate((s_w, s_e), axis=1)
    return np.concatenate((top, bottom), axis=0)


class QuadTree:

    def __init__(self, image):
        self.end = True
        self.quad_size = (image.shape[0], image.shape[1])

        if image.size == 0:
            self.tile_mean = np.zeros((3,)).astype(np.uint8) if len(image.shape) == 3 else 0
            return

        self.tile_mean = np.mean(image, axis=(0, 1)).astype(np.uint8)

        if not (image == self.tile_mean).all():
            split_image = split(image)

            self.end = False
            self.d = (
                QuadTree(split_image[0]),
                QuadTree(split_image[1]),
                QuadTree(split_image[2]),
                QuadTree(split_image[3])
            )

    def get_image(self):
        if self.end:
            return np.tile(self.tile_mean, (*self.quad_size, 1))

        return concatenate(self.d[0].get_image(), self.d[1].get_image(), self.d[2].get_image(),
                           self.d[3].get_image()).astype(int)


def Quad(folder, image):
    im = read_image(folder + image).astype(int)

    encoded_quad = QuadTree(im)
    decoded_quad = encoded_quad.get_image()

    encoded_size = asizeof.asizeof(encoded_quad)
    decoded_size = asizeof.asizeof(im)
    reduce = encoded_size / decoded_size * 100

    plt.figure(dpi=600)
    plt.subplot(1, 2, 1)
    plt.title('Obrazek oryginalny rozmiar\n: ' + str(decoded_size))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(
        'Obrazek zdekodowany QUAD \nrozmiar: ' + str(encoded_size) + '\nProcent obrazu oryginalnego:\n ' + str(
            reduce))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(decoded_quad, cmap='gray')

    plt.savefig('plots/' + image + 'QUAD.png')
    plt.close()


def main():
    directory = os.fsencode('data/')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        rle('data/', filename)
        Quad('data/', filename)


if __name__ == '__main__':
    main()
