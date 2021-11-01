import numpy as np
import pandas as pd
import torch
from nnAudio.Spectrogram import CQT1992v2
from tqdm import tqdm


def apply_qtransform_stacking(waves, transform):
    waves = np.hstack(waves)
    waves = torch.from_numpy(waves).float()
    waves = transform(waves)
    channel, x, y = waves.shape
    return np.resize(waves, (x, y, channel))


def apply_qtransform_single(wave, transform):
    # (64, 64, 1)
    signal = torch.from_numpy(wave).float()
    signal = transform(signal)
    channel, x, y = signal.shape
    return np.resize(signal, (x, y, channel))[:64, :64, :]


def spectrogram_casting(transform, record_id: str, part='train'):
    signals = np.load(f"..\\data\\{part}\\{record_id[0]}\\{record_id[1]}\\{record_id[2]}\\{record_id}.npy")

    return apply_qtransform_single(signals[0], transform=transform)


def change_all_files(labels_csv, part='train'):
    df = pd.read_csv(labels_csv)

    # transform = CQT1992v2(sr=4096, fmin=20, fmax=2048, hop_length=64)
    transform = CQT1992v2(
        sr=4096, fmin=20, fmax=2048,
        hop_length=64, bins_per_octave=14, pad_mode='constant'
    )

    for record_id in tqdm(df.id):
        tmp = spectrogram_casting(
            record_id=record_id,
            part=part,
            transform=transform
        )

        np.save(
            f"..\\data_spectogram_one_signal\\{part}\\{record_id[0]}\\{record_id[1]}\\{record_id[2]}\\{record_id}.npy",
            tmp
        )


if __name__ == '__main__':
    change_all_files('..\\data\\training_labels.csv')
    change_all_files('..\\data\\sample_submission.csv', part='test')
    # os.system("shutdown /s /t 1")
