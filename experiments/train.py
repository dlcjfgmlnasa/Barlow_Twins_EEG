# -*- coding:utf-8 -*-
import os
import ray
import mne
import argparse
import torch.optim as opt
from argparse import Namespace
from dataset.utils import get_train_ft_eval_split
from experiments.evaluation import Evaluation
from experiments.data_loader import *
from models.loss import BarlowTwins
from sklearn.metrics import accuracy_score, f1_score


warnings.filterwarnings(action='ignore')


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    sampling_rate = 125
    parser = argparse.ArgumentParser()
    # [Training Hyperparameter]
    parser.add_argument('--base_path', default=os.path.join('../..', '..', '..', 'data'), type=str)
    parser.add_argument('--data_type', default='openbmi', type=str)
    parser.add_argument('--fold_idx', default=0, type=int)
    parser.add_argument('--fold_split', default=10, type=int)
    parser.add_argument('--label_percent', default=100, choices=[10, 25, 50, 75, 100])

    parser.add_argument('--train_epochs', default=3, type=int)
    parser.add_argument('--train_lr_rate', default=0.001, type=float)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--temperature', default=0.01, type=float)
    parser.add_argument('--backbone_name', default='MultiResolutionCNN', type=str, choices=['EEGNet',
                                                                                            'ShallowConvNet',
                                                                                            'DeepConvNet',
                                                                                            'MultiResolutionCNN'])
    # [Model Hyperparameter]
    parser.add_argument('--backbone_parameter', default={'f1': 4, 'f2': 8, 'd': 2,
                                                         'channel_size': 31,
                                                         'input_time_length': sampling_rate * 3,
                                                         'dropout_rate': 0.2,
                                                         'sampling_rate': sampling_rate})

    # [Data Augmentation Hyperparameter]
    parser.add_argument('--augmentations', default=[('random_crop', 0.95),
                                                    ('random_gaussian_noise', 0.95),
                                                    ('random_horizontal_flip', 0.95),
                                                    ('random_bandpass_filter', 0.95)])
    parser.add_argument('--projection_hidden', default=512, type=int)
    parser.add_argument('--projection_size', default=256, type=int)

    # [Evaluation & Fine-Tuning]
    parser.add_argument('--labels', default={'Left-Hand': 0, 'Right-Hand': 1})

    # [Setting Checkpoint Path]
    parser.add_argument('--ckpt_path', default=os.path.join('../..', '..', '..', 'ckpt', 'openbmi', 'ssl', 'eegnet'))
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = Namespace(**args)
        self.model = BarlowTwins(backbone_name=self.args.backbone_name,
                                 backbone_parameter=self.args.backbone_parameter,
                                 projection_hidden=self.args.projection_hidden,
                                 projection_size=self.args.projection_size).to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)
        self.paths = get_train_ft_eval_split(base_path=self.args.base_path,
                                             n_splits=self.args.fold_split,
                                             data_type=self.args.data_type)
        self.paths = self.paths[self.args.fold_idx]
        self.train_paths, self.ft_paths, self.eval_paths = list(self.paths['train_paths']), \
                                                           list(self.paths['ft_paths']), \
                                                           list(self.paths['eval_paths'])

    def train(self):
        ray.init(log_to_driver=False, num_cpus=4, num_gpus=2)
        train_dataloader = batch_dataloader(paths=self.train_paths, batch_size=self.args.train_batch_size,
                                            augmentations=self.args.augmentations)

        best_model, best_eval_mf1 = None, 0
        best_bf_pred, best_bf_real, best_ft_pred, best_ft_real = None, None, None, None

        for epoch in range(self.args.train_epochs):
            train_count = 0
            self.model.train()

            epoch_loss = []
            for batch in train_dataloader.gather_async(num_async=5):
                self.optimizer.zero_grad()
                x1, x2 = batch
                x1, x2 = x1.to(device), x2.to(device)
                loss = self.model(x1, x2)
                loss.backward()
                self.optimizer.step()
                train_count += 1
                epoch_loss.append(float(loss.detach().cpu().item()))

            epoch_loss = np.mean(epoch_loss)
            (bf_pred, bf_real), (ft_pred, ft_real) = self.compute_evaluation()
            bf_acc, bf_mf1 = accuracy_score(y_true=bf_real, y_pred=bf_pred), \
                             f1_score(y_true=bf_real, y_pred=bf_pred, average='macro')
            ft_acc, ft_mf1 = accuracy_score(y_true=ft_real, y_pred=ft_pred), \
                             f1_score(y_true=ft_real, y_pred=ft_pred, average='macro')

            if ft_mf1 >= best_eval_mf1:
                best_eval_mf1 = ft_mf1
                best_bf_pred, best_bf_real = bf_pred, bf_real
                best_ft_pred, best_ft_real = ft_pred, ft_real

            print('[Epoch] : {0:03d} \t '
                  '[Train Loss] => {1:.4f} \t '
                  '[Fine Tuning] => Acc: {2:.4f} MF1 {3:.4f} \t'
                  '[Frozen Backbone] => Acc: {4:.4f} MF1: {5:.4f}'.format(
                        epoch + 1, epoch_loss, ft_acc, ft_mf1, bf_acc, bf_mf1))

        self.save_ckpt_linear_prob(pred=best_bf_pred, real=best_bf_real)
        self.save_ckpt_fine_tuning(pred=best_ft_pred, real=best_ft_real)
        ray.shutdown()

    def compute_evaluation(self):
        param_names = [name for name, _ in self.model.backbone.named_modules()]

        # 1. Backbone Frozen (train only fc)
        evaluation = Evaluation(backbone=self.model.backbone, device=device, epochs=50, lr=0.005, batch_size=64,
                                label_percent=self.args.label_percent, classes=len(self.args.labels))
        bf_pred, bf_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
                                                  eval_paths=self.eval_paths,
                                                  frozen_layers=param_names,
                                                  converter=self.args.labels)
        del evaluation

        # 2. Fine Tuning (train last layer + fc)
        evaluation = Evaluation(backbone=self.model.backbone, device=device, epochs=50, lr=0.0005, batch_size=64,
                                label_percent=self.args.label_percent, classes=len(self.args.labels))
        ft_pred, ft_real = evaluation.fine_tuning(ft_paths=self.ft_paths,
                                                  eval_paths=self.eval_paths,
                                                  frozen_layers=[],
                                                  converter=self.args.labels)
        del evaluation
        return (bf_pred, bf_real), (ft_pred, ft_real)

    def save_ckpt_linear_prob(self, pred, real):
        ckpt_path = os.path.join(self.args.ckpt_path, 'linear_prob', 'fold_{0:02d}_{1:03d}.npz'.format(
                                 self.args.fold_idx + 1, self.args.label_percent))
        np.savez(ckpt_path, pred=pred, real=real)

    def save_ckpt_fine_tuning(self, pred, real):
        ckpt_path = os.path.join(self.args.ckpt_path, 'fine_tuning', 'fold_{0:02d}_{1:03d}.npz'.format(
                                 self.args.fold_idx + 1, self.args.label_percent))
        np.savez(ckpt_path, pred=pred, real=real)


if __name__ == '__main__':
    argument = vars(get_args())
    Trainer(argument).train()


