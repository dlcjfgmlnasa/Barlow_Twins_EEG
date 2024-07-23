# -*- coding:utf-8 -*-
import os
import copy
import numpy as np
from sklearn.model_selection import KFold
from scipy.signal import butter, lfilter


def butter_bandpass_filter(signal, low_cut, high_cut, fs, order=5):
    if low_cut == 0:
        low_cut = 0.5
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    return y


def get_train_ft_eval_split(base_path, n_splits: int, data_type: str):
    # Subject Variability
    if data_type == 'openbmi':
        train_path = os.path.join(base_path, 'openbmi_sess01')
        train_paths = [os.path.join(train_path, name) for name in os.listdir(train_path)]
        train_paths.sort()
        train_paths = np.array(train_paths)
        eval_path = os.path.join(base_path, 'openbmi_sess02')
        eval_paths = [os.path.join(eval_path, name) for name in os.listdir(eval_path)]
        eval_paths.sort()
        eval_paths = np.array(eval_paths)

        kf = KFold(n_splits=n_splits)

        temp = {}
        for fold, (ft_idx, eval_idx) in enumerate(kf.split(eval_paths)):
            temp[fold] = {'train_paths': train_paths,
                          'ft_paths': eval_paths[ft_idx],
                          'eval_paths': eval_paths[eval_idx]}
        return temp

    if data_type == 'multimodal':
        train_path = os.path.join(base_path, 'openbmi_sess01')
        train_paths = [os.path.join(train_path, name) for name in os.listdir(train_path)]
        train_paths.sort()
        train_paths = np.array(train_paths)
        eval_path = os.path.join(base_path, 'multimodal')
        eval_paths = [os.path.join(eval_path, name) for name in os.listdir(eval_path)]
        eval_paths.sort()
        eval_paths = np.array(eval_paths)

        temp = {}
        for fold, eval_path in enumerate(eval_paths):
            ft_paths = copy.deepcopy(eval_paths)
            ft_paths = list(ft_paths)
            ft_paths.remove(eval_path)

            temp[fold] = {'train_paths': list(train_paths),
                          'ft_paths': list(ft_paths),
                          'eval_paths': [eval_path]}
        return temp



