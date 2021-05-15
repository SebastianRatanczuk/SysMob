import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import soundfile as sf
from scipy.interpolate import interp1d

filesAll = ['sing_high1.wav', 'sing_high2.wav', 'sing_low1.wav', 'sing_low2.wav', 'sing_medium1.wav',
            'sing_medium2.wav', 'sin_440Hz.wav', 'sin_60Hz.wav', 'sin_8000Hz.wav', 'sin_combined.wav']

filesSong = ['sing_high1.wav', 'sing_high2.wav', 'sing_low1.wav', 'sing_low2.wav', 'sing_medium1.wav',
             'sing_medium2.wav']

filesSin = ['sin_440Hz.wav', 'sin_60Hz.wav', 'sin_8000Hz.wav', 'sin_combined.wav']

bits = [4, 8, 16, 24]
freqs = [1, 2, 3, 6, 12, 24]

dataFolder = "data/"
outputFolderReduce = "outputReduce/"
outputFolderResample = "outputRes/"
imagesFolder = "plots/"


def plotFourierFromFile(folder, file):
    data, fs = sf.read(folder + file, dtype=np.int32)
    fsize = 2 ** 14

    plt.figure(dpi=300)

    plt.subplot(2, 1, 1)
    plt.xlim([0, 0.2])
    plt.title(file)
    plt.plot(np.arange(0, data.shape[0]) / fs, data)

    plt.subplot(2, 1, 2)

    yf = scipy.fftpack.fft(data, fsize)
    plt.plot(np.arange(0, fs / 2, fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))

    plt.savefig(imagesFolder + file + "_widmno.png")
    plt.close()


def reduceBites(file, bitsToReduce):
    data, fs = sf.read(dataFolder + file, dtype=np.int32)

    Min = np.iinfo(np.int32).min
    Max = np.iinfo(np.int32).max

    newMin = 2 ** bitsToReduce * -1 / 2
    newMax = 2 ** bitsToReduce / 2 - 1

    # vector = np.arange(np.iinfo(np.int32).min, np.iinfo(np.int32).max, 1000, dtype=np.int32)
    vector = data
    plt.figure(dpi=300)

    plt.subplot(3, 1, 1)
    plt.xlim([0, 10000])
    plt.title(file + "_skalowanie_" + str(bitsToReduce))
    plt.plot(vector)

    Za = (np.int64(vector) - Min) / (Max - Min)
    Zc = np.round(Za * (newMax - newMin)) + newMin

    plt.subplot(3, 1, 2)
    plt.xlim([0, 10000])
    plt.plot(Zc)

    Za = (np.int64(Zc) - newMin) / (newMax - newMin)
    Zc = np.round(Za * (Max - Min)) + Min

    plt.subplot(3, 1, 3)
    plt.xlim([0, 10000])
    plt.plot(Zc)

    plt.savefig(imagesFolder + file + "_skalowanie_" + str(bitsToReduce) + "_bit.png")
    plt.close()

    sf.write(outputFolderReduce + "skalowanie_" + str(bitsToReduce) + "_bit_" + file, Zc.astype(np.int32), fs)


def decymacja(folder, file, skip):
    data, fs = sf.read(folder + file, dtype=np.int32)

    s = data.shape[0]
    newData = data[0:s:skip]
    newFs = int(fs / skip)

    sf.write(outputFolderResample + "probkowanie_" + str(newFs) + "_" + file, newData.astype(np.int32), newFs)


def interpolacja(folder, file, newFs):
    data, fs = sf.read(folder + file, dtype=np.int32)

    x = np.arange(0, data.shape[0])

    if (len(data.shape) == 2):
        y = data[:, 0]
    else:
        y = data

    f = interp1d(x, y)

    xnew = np.arange(0, data.shape[0] - 1, fs / newFs)
    newData = f(xnew)

    sf.write(outputFolderResample + "probkowanie_" + str(newFs) + "_" + file, newData.astype(np.int32), newFs)


def main():
    for bit in bits:
        for filename in filesSong:
            reduceBites(filename, bit)

    directory = os.fsencode(outputFolderReduce)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        interpolacja(outputFolderReduce, filename, 16950)

        for freq in freqs:
            decymacja(outputFolderReduce, filename, freq)

    directory = os.fsencode(outputFolderResample)

    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     plotFourierFromFile(outputFolderResample, filename)


if __name__ == '__main__':
    main()
