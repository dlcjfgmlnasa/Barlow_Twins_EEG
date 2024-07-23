# -*- coding:utf-8 -*-
import copy
import random
import numpy as np
from tslearn.preprocessing import TimeSeriesResampler
from dataset.utils import butter_bandpass_filter
from mspca import mspca


class SignalAugmentation(object):
    # https://arxiv.org/pdf/2109.07839.pdf
    # https://arxiv.org/pdf/1706.00527.pdf

    def __init__(self):
        self.sampling_rate = 125
        self.second = 3
        self.input_length = self.second * self.sampling_rate
        self.l_freq, self.h_freq = 1, 50

        self.band_size = 10                                     # for random_bandpass_filter
        self.gn_scaling = list(np.arange(0.05, 0.15, 0.01))     # for %5 ~ 10% gaussian noise
        self.n_permutation = 4                                  # for permutation
        self.window_width = int(self.sampling_rate / 2)         # window width

    def process(self, x, aug_name, p=0.5):
        x = copy.deepcopy(x)
        if aug_name == 'none':
            return x
        if aug_name == 'random_crop':
            x = self.random_crop(x, p)
            return x
        elif aug_name == 'random_bandpass_filter':
            x = self.random_bandpass_filter(x, p)
            return x
        elif aug_name == 'random_gaussian_noise':
            x = self.random_gaussian_noise(x, p)
            return x
        elif aug_name == 'random_horizontal_flip':
            x = self.random_horizontal_flip(x, p)
            return x
        elif aug_name == 'random_permutation':
            x = self.random_permutation(x, p)
            return x
        elif aug_name == 'random_temporal_delay':
            x = self.random_temporal_delay(x, p)
            return x
        elif aug_name == 'random_temporal_cutout':
            x = self.random_temporal_cutout(x, p)
            return x
        else:
            raise NotImplementedError()

    def random_crop(self, x, p=0.5):
        half_sr = int(self.sampling_rate / 3)

        new_x = []
        for x_split in x:
            if random.random() < p:
                index_1 = np.random.randint(low=0, high=half_sr - 1, size=1)[0]
                index_2 = np.random.randint(low=self.input_length - half_sr, high=self.input_length - 1, size=1)[0]
                x_split = x_split[:, index_1:index_2]                                       # 1. Crop
                x_split = TimeSeriesResampler(sz=self.input_length).fit_transform(x_split)  # 2. Resample
                x_split = np.squeeze(x_split, axis=-1)
                new_x.append(x_split)
            else:
                new_x.append(x_split)
        new_x = np.array(new_x)
        return new_x

    def random_bandpass_filter(self, x, p=0.5):
        low_cut_range = list(range(self.l_freq, self.l_freq + self.band_size))
        high_cut_range = list(range(self.h_freq - self.band_size, self.h_freq))

        new_x = []
        for x_split in x:
            if random.random() < p:
                low_cut, high_cut = np.random.choice(low_cut_range, 1)[0], np.random.choice(high_cut_range, 1)[0]
                x_split = butter_bandpass_filter(x_split, fs=self.sampling_rate, low_cut=low_cut, high_cut=high_cut)
                new_x.append(x_split)
            else:
                new_x.append(x_split)
        new_x = np.array(new_x)
        return new_x

    def random_gaussian_noise(self, x, p=0.5):
        mu = 0.0
        new_x = []
        x = np.array(x)
        std = np.random.choice(self.gn_scaling, 1)[0] * np.std(x)

        for x_split in x:
            if random.random() < p:
                noise = np.random.normal(mu, std, x_split.shape)
                x_split = x_split + noise
                new_x.append(x_split)
            else:
                new_x.append(x_split)
        new_x = np.array(new_x)
        return new_x

    def random_temporal_delay(self, x, p=0.5):
        size = list(range(self.sampling_rate))

        new_x = []
        for x_split in x:
            if random.random() < p:
                last_idx = np.random.choice(size, 1)[0]
                f_sample = x_split[:, :last_idx]
                x_split = np.concatenate((f_sample, x_split), axis=-1)
                x_split = x_split[:, :self.sampling_rate * self.second]
                new_x.append(x_split)
            else:
                new_x.append(x_split)
        new_x = np.array(new_x)
        return new_x

    def random_temporal_cutout(self, x, p=0.5):
        new_x = []
        start_list = list(np.arange(0, self.sampling_rate * self.second))
        width_list = list(np.arange(int(self.sampling_rate / 4), int(self.sampling_rate / 2)))

        for x_split in x:
            if random.random() < p:
                start = np.random.choice(start_list, 1)[0]
                width = np.random.choice(width_list, 1)[0]
                np.put(x_split, np.arange(start, start+width), x_split.mean())
                new_x.append(x_split)
            else:
                new_x.append(x_split)
        new_x = np.array(new_x)
        return new_x

    @staticmethod
    def random_horizontal_flip(x, p=0.5):
        new_x = []
        for x_split in x:
            if random.random() < p:
                x_split = np.flip(x_split, axis=-1)
                new_x.append(x_split)
            else:
                new_x.append(x_split)
        new_x = np.array(new_x)
        return new_x

    def random_permutation(self, x, p=0.5):
        new_x = []
        for x_split in x:
            if random.random() < p:
                indexes = np.random.choice(self.input_length, self.n_permutation - 1, replace=False)
                indexes = list(np.sort(indexes))
                indexes = [0] + indexes + [self.input_length]
                samples = []
                for index_1, index_2 in zip(indexes[:-1], indexes[1:]):
                    samples.append(x_split[:, index_1:index_2])

                nx = []
                for index in np.random.permutation(np.arange(self.n_permutation)):
                    nx.append(samples[index])
                nx = np.concatenate(nx, axis=-1)
                new_x.append(nx)
            else:
                new_x.append(x_split)
        new_x = np.array(new_x)
        return new_x


