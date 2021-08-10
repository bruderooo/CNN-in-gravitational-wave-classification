import numpy as np


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
