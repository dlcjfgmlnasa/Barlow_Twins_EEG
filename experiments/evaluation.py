# -*- coding:utf-8 -*-
import copy
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
from typing import List, Callable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TorchDataset(Dataset):
    def __init__(self, paths: List, label_percent=100, converter: Callable = None):
        self.paths = paths
        self.xs, self.ys = self.get_data(label_percent=label_percent, converter=converter)

    def __len__(self):
        return self.xs.shape[0]

    def get_data(self, label_percent, converter):
        xs, ys = [], []
        for path in self.paths:
            data = pickle.load(open(path, 'rb'))
            x, y = data['x'], data['y']
            y = np.array([converter[y_] for y_ in y])

            # Semi-supervised Learning with Few Labeled
            n = int(y.shape[0] * (label_percent / 100))
            n = int(n / 2)
            mask = np.hstack([np.random.choice(np.where(y == l0)[0], n, replace=False)
                              for l0 in np.unique(y)])
            xs.append(x[mask])
            ys.append(y[mask])

            xs.append(x)
            ys.append(y)
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        return xs, ys

    def __getitem__(self, idx):
        x = torch.tensor(self.xs[idx], dtype=torch.float)
        y = torch.tensor(self.ys[idx], dtype=torch.long)
        return x, y


class Model(nn.Module):
    def __init__(self, backbone: nn.Module, frozen_layers: List, classes: int):
        super().__init__()
        self.backbone = self.get_backbone(backbone=backbone, frozen_layers=frozen_layers)
        self.classes = classes
        self.fc = nn.Sequential(
            nn.Linear(backbone.final_length, backbone.final_length),
            nn.BatchNorm1d(backbone.final_length),
            nn.ELU(),
            nn.Linear(backbone.final_length, self.classes)
        )

    @staticmethod
    def get_backbone(backbone: nn.Module, frozen_layers: List):
        backbone = copy.deepcopy(backbone)
        for name, module in backbone.named_modules():
            if name in frozen_layers:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True
        return backbone

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class Evaluation(object):
    def __init__(self, backbone: nn.Module, device, epochs=200, lr=0.01, batch_size=1024,
                 classes=2, label_percent=10):
        self.backbone = backbone
        self.device = device

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.label_percent = label_percent
        self.classes = classes
        self.criterion = nn.CrossEntropyLoss()

    def fine_tuning(self, ft_paths, eval_paths, frozen_layers: List, converter: Callable = None) \
            -> (np.array, np.array):
        # 1. [frozen = False] => Fine Tuning / 2. [frozen = True] => Backbone Frozen
        backbone = copy.deepcopy(self.backbone)

        model = Model(backbone=backbone, frozen_layers=frozen_layers, classes=self.classes).to(self.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
        train_dataset = TorchDataset(paths=ft_paths, label_percent=self.label_percent, converter=converter)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        eval_dataset = TorchDataset(paths=eval_paths, converter=converter)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=500)

        best_model, best_eval_mf1 = None, 0
        best_pred, best_real = None, None
        for epoch in range(self.epochs):
            model.train()
            for data in train_dataloader:
                optimizer.zero_grad()
                x, y = data
                x, y = x.to(self.device), y.to(self.device)

                pred = model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                optimizer.step()

            model.eval()
            total_pred, total_real = [], []
            for data in eval_dataloader:
                with torch.no_grad():
                    x, y = data
                    x = torch.tensor(x, dtype=torch.float32).to(self.device)
                    pred = model(x)
                    pred = pred.argmax(dim=-1).detach().cpu().numpy()
                    real = y
                    total_pred.extend(pred)
                    total_real.extend(real)

            pred, real = np.array(total_pred), np.array(total_real)
            eval_acc, eval_mf1 = accuracy_score(y_true=real, y_pred=pred), \
                                 f1_score(y_true=real, y_pred=pred, average='macro')

            if eval_mf1 > best_eval_mf1:
                best_eval_mf1 = eval_mf1
                best_pred, best_real = pred, real

        del backbone, model, optimizer, train_dataset, train_dataloader
        return best_pred, best_real

    @staticmethod
    def train_mode(model, frozen_layers: List):
        for name, module in model.backbone.named_modules():
            if name in frozen_layers:
                module.eval()
            else:
                module.train()
        return model
