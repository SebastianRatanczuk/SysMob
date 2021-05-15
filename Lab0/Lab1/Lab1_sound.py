import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

data, fs = sf.read('../data/sound1.wav', dtype='float32')
print(data.dtype)
print(data.shape)

# sd.play(data, fs)
# status = sd.wait()

plt.subplot(2, 1, 1)
plt.plot(data[:, 0])

plt.subplot(2, 1, 2)
plt.plot(data[:, 1])
plt.show()

dataL = data.copy()
dataL[:, 1] = dataL[:, 0]

plt.subplot(2, 1, 1)
plt.plot(dataL[:, 0])

plt.subplot(2, 1, 2)
plt.plot(dataL[:, 1])
plt.show()

sf.write('../data/sound_L.wav', dataL, fs)

dataR = data.copy()
dataR[:, 0] = dataR[:, 1]

plt.subplot(2, 1, 1)
plt.plot(dataR[:, 0])

plt.subplot(2, 1, 2)
plt.plot(dataR[:, 1])
plt.show()

sf.write('../data/sound_R.wav', dataR, fs)

dataM = (data[:, 0] + data[:, 1]) / 2

sound_mix = np.array([dataM, dataM]).T

plt.subplot(2, 1, 1)
plt.plot(sound_mix[:, 0])

plt.subplot(2, 1, 2)
plt.plot(sound_mix[:, 1])
plt.show()

sf.write('../data/sound_mix.wav', dataR, fs)
