import cv2
import numpy as np
import scipy.fftpack
from PIL import Image


class JpgConverter:

    def __init__(self):
        self.shape = None

    def readImage(self, image):
        self.rgbImage = image

    def convertToYCRCB(self):
        self.YCRCBImage = cv2.cvtColor(self.rgbImage, cv2.COLOR_RGB2YCrCb).astype(int)

    def convertToRGB(self):
        self.rgbImage = cv2.cvtColor(self.YCRCBImage.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    def dct2(self, data):
        return scipy.fftpack.dct(scipy.fftpack.dct(data.astype(float) - 128, axis=0, norm='ortho'), axis=1,
                                 norm='ortho')

    def idct2(self, data):
        return scipy.fftpack.idct(scipy.fftpack.idct(data.astype(float), axis=0, norm='ortho'), axis=1,
                                  norm='ortho') + 128

    def splitIntoBlocks(self, data):
        blocks = []
        horizontal = np.hsplit(data, data.shape[1] // 8)
        for h in horizontal:
            vertical = np.vsplit(h, h.shape[0] // 8)
            for v in vertical:
                blocks.append(v)

        return blocks

    def zigzag(self, A):
        template = n = np.array([
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
        ])
        if len(A.shape) == 1:
            B = np.zeros((8, 8))
            for r in range(0, 8):
                for c in range(0, 8):
                    B[r, c] = A[template[r, c]]
        else:
            B = np.zeros((64,))
            for r in range(0, 8):
                for c in range(0, 8):
                    B[template[r, c]] = A[r, c]
        return B


def main():
    im = Image.open("data/data-1.jpg")
    data = np.array(im)

    converter = JpgConverter()
    converter.shape = (1, 1)
    converter.readImage(data)

    print(converter.splitIntoBlocks(data[:, :, 0]))


if __name__ == '__main__':
    main()
