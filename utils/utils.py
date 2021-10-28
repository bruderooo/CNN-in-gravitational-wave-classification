import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from tqdm import tqdm


def file_structure_generator(path_from, path_to):
    for dirpath, dirnames, filenames in tqdm(os.walk(path_from)):
        structure = os.path.normpath(path_to + dirpath[len(str(path_from)):])
        if not os.path.isdir(structure):
            os.mkdir(structure)


def sampling(three_signals, no_samples):
    new_signals = []
    t_old = np.linspace(0, 2, 4096)
    t_new = np.linspace(0, 2, no_samples)

    for signal in three_signals:
        new_signals.append(np.interp(t_new, t_old, signal))

    return np.array(new_signals)


def dft(three_signals):
    new_signals = []

    for signal in three_signals:
        new_signals.append(np.fft.fft(signal))

    return np.array(new_signals)


def load(ID):
    return np.load(f"..\\data\\train\\{ID[0]}\\{ID[1]}\\{ID[2]}\\{ID}.npy")


if __name__ == '__main__':
    from scipy.signal import butter, lfilter

    x = np.linspace(0, 2, 4096)

    id_ = "00014b7a9d"
    wave = load(id_)[0]

    lowcut = 35
    highcut = 350
    fs = 2048

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # order = buttord(wp)
    b, a = butter(10, (low, high), btype='bandpass', output='ba', analog=False)

    # wave_filtered = fft.dct(lfilter(b, a, fft.dct(wave)))
    wave_filtered = lfilter(b, a, wave)

    _, axs = plt.subplots(2, sharex=True)
    axs[0].plot(x, wave)
    axs[1].plot(x, wave_filtered)
    plt.savefig('comparision.png', facecolor='white', dpi=256)

    _, axs = plt.subplots(2, sharex=True)
    axs[0].plot(fft.dct(wave))
    axs[1].plot(fft.dct(wave_filtered))
    plt.savefig('comparision_fft.png', facecolor='white', dpi=256)
