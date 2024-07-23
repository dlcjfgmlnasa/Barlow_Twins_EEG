# -*- coding:utf-8 -*-
import os
import abc
import mne
import pickle
import numpy as np
from scipy import io
from typing import List, Dict
from mspca import mspca


class Base(object):
    def __init__(self, labels: Dict, sfreq: float, second: float, ch_names: List[str]):
        self.labels = labels
        self.sfreq = sfreq
        self.second = second
        self.ch_names = ch_names
        self.l_freq, self.h_freq = 1, 50
        self.mspca_threshold = 0.2

    @abc.abstractmethod
    def parser(self, path) -> ((np.array, np.array), mne.Info):
        pass

    def preprocessing(self, raw: mne.EpochsArray):
        channel_names = ['CP6', 'F3', 'P4', 'C3', 'Pz', 'Fp1', 'FC1', 'O2', 'F7', 'O1', 'FC5', 'TP10', 'TP9',
                         'FT9', 'Fp2', 'CP5', 'F8', 'F4', 'CP2', 'FC6', 'P3', 'CP1', 'FC2', 'FT10', 'P8', 'C4',
                         'P7', 'Cz', 'T8', 'Oz', 'T7']
        second = 3
        info = raw.info

        # 1. slicing signal
        x = raw.get_data()[..., :second*int(info['sfreq'])]
        raw = mne.EpochsArray(x, info=info)

        # 2. resampling
        raw = raw.resample(125)

        # 3. Band Pass Filter   ex) 1 Hz ~ 50 Hz
        raw = raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)

        # 4. Channel Selection
        drop_ch_name = [ch_name for ch_name in raw.info.ch_names if ch_name not in channel_names]
        raw = raw.drop_channels(drop_ch_name)

        # 5. Multiscale PCA (MSPCA)
        new_x = []
        x = raw.get_data(copy=True)
        for i in range(x.shape[1]):
            sample = x[:, i, :].squeeze()
            pca = mspca.MultiscalePCA()
            sample = pca.fit_transform(sample, wavelet_func='db4', threshold=self.mspca_threshold)
            new_x.append(sample)
        new_x = np.stack(new_x, axis=1)
        return new_x, raw.info

    def save(self, scr_path: str, trg_path: str):
        (x, y), info = self.parser(scr_path)
        with open(trg_path, 'wb') as fp:
            pickle.dump({'x': x, 'y': y, 'info': info}, fp)


class OpenBMI_MI(Base):
    # EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms: An Investigation into BCI Illiteracy
    # https://academic.oup.com/gigascience/article/8/5/giz002/5304369
    def __init__(self):
        super(OpenBMI_MI, self).__init__(**{
            'ch_names': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                         'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
                         'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4',
                         'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h',
                         'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h',
                         'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'],
            'sfreq': 1000,
            'labels': {'left': 'Left-Hand', 'right': 'Right-Hand'},
            'second': 4
        })

    def parser(self, path) -> ((np.array, np.array), mne.Info):
        mat = io.loadmat(path)['EEG_MI_train'][0][0]
        y = np.array([self.labels[str(ch)[2:-2]] for ch in mat[6][0]])
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg')
        x = np.swapaxes(np.swapaxes(mat[0], 0, 2), 0, 1)
        raw = mne.EpochsArray(x, info=info)
        x, info = self.preprocessing(raw)
        return (x, y), info


class BrainLab_MI(Base):
    def __init__(self):
        super().__init__(**{
            'ch_names': ['C3', 'C4', 'Cz', 'CP1', 'CP2', 'CP5', 'CP6', 'F3', 'F4', 'F7', 'F8',
                         'FC1', 'FC2', 'FC5', 'FC6', 'Fp1', 'Fp2', 'FT9', 'FT10',
                         'O1', 'O2', 'Oz', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8', 'TP10', 'TP9'],
            'sfreq': 500,
            'labels': {4: 'Reach', 5: 'Grasp', 6: 'Lift', 7: 'Twist'},
            'second': 3
        })

    def parser(self, path) -> ((np.array, np.array), mne.Info):
        total_x, total_y = [], []
        for i in range(3):
            data_path, label_path = os.path.join(path, 'sess_{}_data.mat'.format(i+1)), \
                                    os.path.join(path, 'sess_{}_label.mat'.format(i+1))
            x = io.loadmat(data_path)['Data'].squeeze()
            x = np.swapaxes(x, 2, 0).swapaxes(2, 1)
            y = io.loadmat(label_path)['Label'].squeeze()
            total_x.append(x)
            total_y.append(y)

        total_x, total_y = np.concatenate(total_x, axis=0), np.concatenate(total_y, axis=0)
        y = [self.labels[y_] for y_ in total_y]
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg')
        raw = mne.EpochsArray(total_x, info=info)
        raw = self.preprocessing(raw)
        return (raw.get_data(), y), raw.info


if __name__ == '__main__':
    # 1. OpenBCI [MI]
    # src_base_path_ = os.path.join('..', '..', '..', '..', 'Dataset', 'DeepBCI', 'MI')
    src_base_path_ = r'C:\Users\chlee\Desktop\DeepBCI\MI'
    trg_base_path_ = os.path.join('..', 'data', 'openbmi')
    obj = OpenBMI_MI()
    for path_ in os.listdir(src_base_path_):
        name = path_.split('.')[0] + '.pkl'
        src_path_ = os.path.join(src_base_path_, path_)
        trg_path_ = os.path.join(trg_base_path_, name)
        obj.save(src_path_, trg_path_)

    # # 2. BrainLab [MI]
    # src_base_path_ = os.path.join('..', '..', '..', '..', 'Dataset', 'Disabled', 'EEG')
    # trg_base_path_ = os.path.join('..', 'data', 'multimodal')
    # obj = BrainLab_MI()
    # for path_ in os.listdir(src_base_path_):
    #     name = path_.split('.')[0] + '.pkl'
    #     src_path_ = os.path.join(src_base_path_, path_)
    #     trg_path_ = os.path.join(trg_base_path_, name)
    #     obj.save(src_path_, trg_path_)
    #
