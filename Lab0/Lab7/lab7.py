import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import soundfile as sf
from tqdm import tqdm

filesAll = ['sing_high1.wav', 'sing_high2.wav', 'sing_low1.wav', 'sing_low2.wav', 'sing_medium1.wav',
            'sing_medium2.wav', 'sin_440Hz.wav', 'sin_60Hz.wav', 'sin_8000Hz.wav', 'sin_combined.wav']

filesSong = ['sing_high1.wav', 'sing_high2.wav', 'sing_low1.wav', 'sing_low2.wav', 'sing_medium1.wav',
             'sing_medium2.wav']

filesSin = ['sin_440Hz.wav', 'sin_60Hz.wav', 'sin_8000Hz.wav', 'sin_combined.wav']

dataFolder = "data/"
outputFolderAlaw = "alaw/"
outputFolderDPCM = "dpcm/"


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

    plt.savefig('plots/' + file + "_widmno.png")
    plt.close()


def compressALaw(x, A=87.6, bit=8):
    x = np.copy(x)
    indexes = np.abs(x) < 1 / A
    s = np.sign(x)
    x[indexes] = (A * np.abs(x[indexes])) / (1 + np.log(A))
    x[~indexes] = (1 + np.log(A * np.abs(x[~indexes]))) / (1 + np.log(A))

    x = x * s
    gz = x > 0
    x[gz] = np.round(x[gz] * 2 ** (bit - 1)) / 2 ** (bit - 1)
    x[~gz] = np.round(x[~gz] * (2 ** (bit - 1) - 1)) / (2 ** (bit - 1) - 1)
    return x


def decompressALaw(x, A=87.6):
    indexes = np.abs(x) < 1 / (1 + np.log(A))
    s = np.sign(x)
    x[indexes] = (np.abs(x[indexes]) * (1 + np.log(A))) / A
    x[~indexes] = np.exp(np.abs(x[~indexes]) * (1 + np.log(A)) - 1) / A
    return x * s


def ALawManager(file, bit=8):
    data, fs = sf.read(dataFolder + file, dtype=np.int32)

    Max = np.iinfo(np.int32).max
    data = data / Max

    y = compressALaw(data, bit=bit)
    z = decompressALaw(y)

    plt.figure(figsize=(12, 6), dpi=600)
    plt.subplot(1, 2, 1)
    plt.title("Dźwiek oryginalny")
    plt.xlim([0, 5000])
    plt.plot(data)

    plt.subplot(1, 2, 2)
    plt.title("Dźwiek skompresowany ALaw " + str(bit) + "bit")

    plt.xlim([0, 5000])
    plt.plot(z)

    plt.savefig('plots/' + file + '_ALaw_' + str(bit) + '_bit' + '.png')
    plt.close()
    z = z * Max
    sf.write(outputFolderAlaw + "kompresjaAlaw_" + str(bit) + "_bit_" + file, z.astype(np.int32), fs)


def compressDPCM(x, bit=8):
    E = x[0]
    x_out = np.zeros(x.shape[0])
    x_out[0] = x[0] / (np.iinfo(np.int32).max - 1) * (2 ** (bit - 1) - 1)
    for i in tqdm(range(1, x.shape[0])):
        yPrim = round((x[i] - E) / (np.iinfo(np.int32).max - 1) * (2 ** (bit - 1) - 1))
        x_out[i] = yPrim
        Y = yPrim / (2 ** (bit - 1) - 1) * (np.iinfo(np.int32).max - 1)
        E += Y
    return x_out


def decompressDPCM(x, bit=8):
    return (np.round(np.cumsum(x)) / (2 ** (bit - 1) - 1) * (np.iinfo(np.int32).max - 1)).astype(np.int32)


def DPCMManager(file, bit=8):
    data, fs = sf.read(dataFolder + file, dtype=np.int32)

    y = compressDPCM(data, bit=bit)
    z = decompressDPCM(y, bit=bit)

    plt.figure(figsize=(12, 6), dpi=600)
    plt.subplot(1, 2, 1)
    plt.title("Dźwiek oryginalny")
    plt.xlim([0, 5000])
    plt.plot(data)

    plt.subplot(1, 2, 2)
    plt.title("Dźwiek skompresowany DPCM " + str(bit) + "bit")

    plt.xlim([0, 5000])
    plt.plot(z)

    plt.savefig('plots/' + file + '_DPCM_' + str(bit) + '_bit' + '.png')
    plt.close()

    sf.write(outputFolderDPCM + "kompresjaDPCM_" + str(bit) + "_bit_" + file, z.astype(np.int32), fs)


def main():
    for f in filesAll:
        for b in range(2, 9):
            ALawManager(f, b)
            DPCMManager(f, b)

    # directory = os.fsencode(outputFolderAlaw)
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     plotFourierFromFile(outputFolderAlaw, filename)
    #
    # directory = os.fsencode(outputFolderDPCM)
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     plotFourierFromFile(outputFolderDPCM, filename)


if __name__ == '__main__':
    main()
    print("END")
