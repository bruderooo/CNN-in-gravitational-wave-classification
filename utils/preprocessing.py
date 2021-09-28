import os

import numpy as np
import pandas as pd
import torch
from nnAudio.Spectrogram import CQT1992v2
from tqdm import tqdm


def file_structure_generator(path_from, path_to):
    for dirpath, dirnames, filenames in tqdm(os.walk(path_from)):
        structure = os.path.join(path_to, dirpath[len(path_from):])
        if not os.path.isdir(structure):
            os.mkdir(structure)


def apply_qtransform(waves, transform=CQT1992v2(sr=4096, fmin=20, fmax=2048, hop_length=64)):
    waves = np.hstack(waves)
    waves = waves / np.max(waves)
    waves = torch.from_numpy(waves).float()
    return transform(waves)


def spectrogram_casting(record_id: str):
    signals = np.load(f"..\\data\\train\\{record_id[0]}\\{record_id[1]}\\{record_id[2]}\\{record_id}.npy")
    return apply_qtransform(signals)


def change_all_files(labels_csv, part='train'):
    df = pd.read_csv(labels_csv)

    for record_id in tqdm(df.id):
        tmp = spectrogram_casting(record_id)
        np.save(f"..\\data_spectrogram\\{part}\\{record_id[0]}\\{record_id[1]}\\{record_id[2]}\\{record_id}.npy",
                tmp)


if __name__ == '__main__':
    # change_all_files('..\\data\\training_labels.csv')
    change_all_files('..\\data\\sample_submission.csv', part='test')
