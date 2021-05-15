from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

pap = np.array(
    [
        [0.22352941, 0., 0.],
        [0.56470588, 0.04705882, 0.05882353],
        [0.56470588, 0.04705882, 0.05882353],
        [0.42352941, 0.21568627, 0.15294118],
        [0.76862745, 0.19607843, 0.1372549],
        [0.74509804, 0.40392157, 0.32941176],
        [0.43529412, 0.47843137, 0.24313725],
        [0.71372549, 0.8, 0.31372549],
        [0.4745098, 0.64705882, 0.30588235],
        [0.73333333, 0.15686275, 0.1254902],
        [0.23529412, 0, 0.00392157],
        [0.74901961, 0.8627451, 0.74509804],
        [0.64313725, 0.7372549, 0.29803922],
        [0.78431373, 0.87058824, 0.81568627],
        [0.40392157, 0.21960784, 0.10196078],
        [0.5254902, 0.43921569, 0.28627451],
        [0.34117647, 0.50196078, 0.21960784],
        [0.5254902, 0.04705882, 0.05098039],
        [0.7372549, 0.77254902, 0.3254902],
        [0.69411765, 0.81960784, 0.61960784],
        [0.83921569, 0.29019608, 0.22352941],
        [0.32156863, 0.4, 0.14509804],
    ]
)

black_whiteArray1 = np.array([0, 1])
black_whiteArray2 = np.array([0, 0.25, 0.5, 0.75, 1])
black_whiteArray3 = np.array(
    [0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375])

malekolorkiArray = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

kolorkiArray = np.array(
    [
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 1],
        [0, 0.5, 0],
        [0.5, 0.5, 0.5],
        [0, 1, 0],
        [0.5, 0, 0],
        [0, 0, 0.5],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [1, 0, 0],
        [0.75, 0.75, 0.75],
        [0, 0.5, 0.5],
        [1, 1, 1],
        [1, 1, 0]
    ])


def getBayer(bits):
    m = np.array([[0, 2], [3, 1]])
    i = 2
    while i != bits:
        m = np.vstack((np.hstack((m * 4, m * 4 + 2)),
                       np.hstack((m * 4 + 3, m * 4 + 1))))
        i <<= 1
    m = m / bits ** 2 - 0.5
    return m


def colorFitGray(reference, color_array):
    return color_array[np.argmin(np.abs(color_array - reference))]


def colorFit(reference, color_array):
    return color_array[np.argmin(np.linalg.norm(color_array - reference, axis=1))]


def randomDithering(input_image):
    output_image = np.random.uniform(0, 1, input_image.shape)

    for K in range(input_image.shape[0]):
        for W in range(input_image.shape[1]):
            output_image[K, W] = 1 if output_image[K, W] < input_image[K, W] else 0
    return output_image


def fitToPallet(input_image, pal):
    output_image = np.zeros(input_image.shape)

    for K in range(input_image.shape[0]):
        for W in range(input_image.shape[1]):
            output_image[K, W] = colorFit(input_image[K, W], pal)

    return output_image


def fitToPalletGray(input_image, pal):
    output_image = np.ones(input_image.shape)

    for K in range(input_image.shape[0]):
        for W in range(input_image.shape[1]):
            output_image[K, W] = colorFitGray(input_image[K, W], pal)

    return output_image


def orderedDithering(input_image, pal):
    output_image = np.zeros(input_image.shape)
    M = getBayer(4)
    n = M.shape[0]

    for K in range(input_image.shape[0]):
        for W in range(input_image.shape[1]):
            output_image[K, W] = colorFit(input_image[K, W] + M[K % n, W % n], pal)

    return output_image


def orderedDitheringGray(input_image, pal):
    output_image = np.zeros(input_image.shape)
    M = getBayer(4)
    n = M.shape[0]

    for K in range(input_image.shape[0]):
        for W in range(input_image.shape[1]):
            output_image[K, W] = colorFitGray(input_image[K, W] + M[K % n, W % n], pal)

    return output_image


def FordWarszawaDithering(input_image, pal):
    output_image = np.copy(input_image)

    for K in range(input_image.shape[0]):
        for W in range(input_image.shape[1]):
            oldpixel = np.copy(output_image[K, W])
            newpixel = colorFit(oldpixel, pal)
            output_image[K, W] = newpixel
            quant_error = oldpixel - newpixel

            if W + 1 < input_image.shape[1]:
                output_image[K][W + 1] = output_image[K][W + 1] + quant_error * 5 / 16

            if K + 1 < input_image.shape[0]:
                output_image[K + 1][W] = output_image[K + 1][W] + quant_error * 7 / 16
                output_image[K + 1][W - 1] = output_image[K + 1][W - 1] + quant_error * 3 / 16

                if W + 1 < input_image.shape[1]:
                    output_image[K + 1][W + 1] = output_image[K + 1][W + 1] + quant_error * 1 / 16

    return np.clip(output_image, 0, 1)


def FordWarszawaDitheringGray(input_image, pal):
    output_image = np.copy(input_image)

    for K in range(input_image.shape[0]):
        for W in range(input_image.shape[1]):
            oldpixel = np.copy(output_image[K, W])
            newpixel = colorFitGray(oldpixel, pal)

            output_image[K, W] = newpixel

            quant_error = oldpixel - newpixel

            if W + 1 < input_image.shape[1]:
                output_image[K][W + 1] = output_image[K][W + 1] + quant_error * 5 / 16

            if K + 1 < input_image.shape[0]:
                output_image[K + 1][W] = output_image[K + 1][W] + quant_error * 7 / 16
                output_image[K + 1][W - 1] = output_image[K + 1][W - 1] + quant_error * 3 / 16
                if W + 1 < input_image.shape[1]:
                    output_image[K + 1][W + 1] = output_image[K + 1][W + 1] + quant_error * 1 / 16

    return np.clip(output_image, 0, 1)


def black_white_dithering1(image):
    folder = 'data/'
    im = Image.open(folder + image)
    Zc = np.array(im)

    m = np.min(Zc)
    n = np.max(Zc)

    Za = (Zc - m) / (n - m)
    plt.figure(figsize=(18, 10), dpi=150)
    plt.subplot(1, 5, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(Za, cmap='gray')
    plt.title("Oryginalny obrazek")

    random_Matrix = randomDithering(Za)

    plt.subplot(1, 5, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(random_Matrix, cmap='gray')
    plt.title("Random dithering")

    ra = random_Matrix * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("random_" + folder + image)

    pallet_fit = fitToPalletGray(Za, black_whiteArray1)
    plt.subplot(1, 5, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pallet_fit, cmap='gray')
    plt.title("Color Fit ")

    ra = pallet_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("pallet_" + folder + image)

    ordered_fit = orderedDitheringGray(Za, black_whiteArray1)
    plt.subplot(1, 5, 4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(ordered_fit, cmap='gray')
    plt.title("Ordered Dithering")

    ra = ordered_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ordered_" + folder + image)

    floyd_fit = FordWarszawaDitheringGray(Za, black_whiteArray1)
    plt.subplot(1, 5, 5)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(floyd_fit, cmap='gray')
    plt.title("Floyd Dithering")
    ra = floyd_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ford_" + folder + image)

    plt.savefig("1bit" + image + ".png")


def black_white_dithering2(image):
    folder = 'data/'
    im = Image.open(folder + image)
    Zc = np.array(im)

    m = np.min(Zc)
    n = np.max(Zc)

    Za = (Zc - m) / (n - m)
    plt.figure(figsize=(18, 10), dpi=150)
    plt.subplot(1, 5, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(Za, cmap='gray')
    plt.title("Oryginalny obrazek")

    random_Matrix = randomDithering(Za)

    plt.subplot(1, 5, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(random_Matrix, cmap='gray')
    plt.title("Random dithering")

    ra = random_Matrix * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("random_" + folder + image)

    pallet_fit = fitToPalletGray(Za, black_whiteArray2)
    plt.subplot(1, 5, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pallet_fit, cmap='gray')
    plt.title("Color Fit ")

    ra = pallet_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("pallet_" + folder + image)

    ordered_fit = orderedDitheringGray(Za, black_whiteArray2)
    plt.subplot(1, 5, 4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(ordered_fit, cmap='gray')
    plt.title("Ordered Dithering")

    ra = ordered_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ordered_" + folder + image)

    floyd_fit = FordWarszawaDitheringGray(Za, black_whiteArray2)
    plt.subplot(1, 5, 5)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(floyd_fit, cmap='gray')
    plt.title("Floyd Dithering")
    ra = floyd_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ford_" + folder + image)

    plt.savefig("2bit" + image + ".png")


def black_white_dithering3(image):
    folder = 'data/'
    im = Image.open(folder + image)
    Zc = np.array(im)

    m = np.min(Zc)
    n = np.max(Zc)

    Za = (Zc - m) / (n - m)
    plt.figure(figsize=(18, 10), dpi=150)
    plt.subplot(1, 5, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(Za, cmap='gray')
    plt.title("Oryginalny obrazek")

    random_Matrix = randomDithering(Za)

    plt.subplot(1, 5, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(random_Matrix, cmap='gray')
    plt.title("Random dithering")

    ra = random_Matrix * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("random_" + folder + image)

    pallet_fit = fitToPalletGray(Za, black_whiteArray3)
    plt.subplot(1, 5, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pallet_fit, cmap='gray')
    plt.title("Color Fit ")

    ra = pallet_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("pallet_" + folder + image)

    ordered_fit = orderedDitheringGray(Za, black_whiteArray3)
    plt.subplot(1, 5, 4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(ordered_fit, cmap='gray')
    plt.title("Ordered Dithering")

    ra = ordered_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ordered_" + folder + image)

    floyd_fit = FordWarszawaDitheringGray(Za, black_whiteArray3)
    plt.subplot(1, 5, 5)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(floyd_fit, cmap='gray')
    plt.title("Floyd Dithering")
    ra = floyd_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ford_" + folder + image)

    plt.savefig("4bit" + image + ".png")


def colored_dithering(image):
    folder = 'data/'
    im = Image.open(folder + image)
    Zc = np.array(im)

    m = np.min(Zc)
    n = np.max(Zc)

    Za = (Zc - m) / (n - m)

    plt.figure(figsize=(18, 10), dpi=150)
    plt.subplot(2, 4, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(Za, cmap='gray')
    plt.title("Oryginalny obrazek")

    plt.subplot(2, 4, 5)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(Za, cmap='gray')
    plt.title("Oryginalny obrazek")

    pallet_fit = fitToPallet(Za, malekolorkiArray)
    plt.subplot(2, 4, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pallet_fit)
    plt.title("Fit 8 bit")

    ra = pallet_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("pallet_" + folder + image)

    ordered_fit = orderedDithering(Za, malekolorkiArray)
    plt.subplot(2, 4, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(ordered_fit)
    plt.title("Ordered 8bit")

    ra = ordered_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ordered_" + folder + image)

    floyd_fit = FordWarszawaDithering(Za, malekolorkiArray)
    plt.subplot(2, 4, 4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(floyd_fit)
    plt.title("Floyd 8bit")

    ra = floyd_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ford_" + folder + image)

    pallet_fit = fitToPallet(Za, kolorkiArray)
    plt.subplot(2, 4, 6)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pallet_fit)
    plt.title("Fit 16bit")

    ra = pallet_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("pallet_" + folder + "kolorki" + image)

    ordered_fit = orderedDithering(Za, kolorkiArray)
    plt.subplot(2, 4, 7)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(ordered_fit)
    plt.title("Ordered 16bit")

    ra = ordered_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ordered_" + folder + "kolorki" + image)

    floyd_fit = FordWarszawaDithering(Za, kolorkiArray)
    plt.subplot(2, 4, 8)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(floyd_fit)
    plt.title("Floyd 16bit")

    ra = floyd_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ford_" + folder + "kolorki" + image)

    plt.savefig(image + ".png")


def papryka(image, pallete):
    folder = 'data/'
    im = Image.open(folder + image)
    Zc = np.array(im)

    m = np.min(Zc)
    n = np.max(Zc)

    Za = (Zc - m) / (n - m)

    plt.figure(figsize=(18, 10), dpi=150)
    plt.subplot(1, 4, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(Za, cmap='gray')

    pallet_fit = fitToPallet(Za, pallete)
    plt.subplot(1, 4, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pallet_fit)

    ra = pallet_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("pallet_" + folder + image)

    ordered_fit = orderedDithering(Za, pallete)
    plt.subplot(1, 4, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(ordered_fit)

    ra = ordered_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ordered_" + folder + image)

    floyd_fit = FordWarszawaDithering(Za, pallete)
    plt.subplot(1, 4, 4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(floyd_fit)

    ra = floyd_fit * 255
    ra = ra.astype(np.uint8)
    im = Image.fromarray(ra)
    im.save("ford_" + folder + image)
    plt.savefig("paprytga.png")


def main():
    rgbImages = ["0005.tif", "0006.tif", "0010.jpg", "0011.jpg", "0013.jpg", "0014.jpg",
                 "0015.jpg", "0016.jpg"]

    grayImages = ["0007.tif", "0008.png", "0009.png"]

    with Pool(3) as p:
        p.map(black_white_dithering1, grayImages)

    with Pool(3) as p:
        p.map(black_white_dithering2, grayImages)

    with Pool(3) as p:
        p.map(black_white_dithering3, grayImages)

    with Pool(8) as p:
        p.map(colored_dithering, rgbImages)


if __name__ == '__main__':
    papryka("0006.tif", pap)
    # main()

    print("end")
