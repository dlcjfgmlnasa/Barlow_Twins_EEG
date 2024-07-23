# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn


def get_backbone_model(model_name, parameters):
    from models.backbone import EEGNet
    from models.backbone import ShallowConvNet
    from models.backbone import DeepConvNet
    from models.backbone import MultiResolutionCNN

    if model_name == 'EEGNet':
        backbone = EEGNet(**parameters)
        return backbone
    if model_name == 'ShallowConvNet':
        backbone = ShallowConvNet(**parameters)
        return backbone
    if model_name == 'DeepConvNet':
        backbone = DeepConvNet(**parameters)
        return backbone
    if model_name == 'MultiResolutionCNN':
        backbone = MultiResolutionCNN(**parameters)
        return backbone


def np_to_var(x, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
    if not hasattr(x, "__len__"):
        x = [x]
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    x_tensor = torch.tensor(x, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        x_tensor = x_tensor.pin_memory()
    return x_tensor


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return torch.log(torch.clamp(x, min=eps))


class Expression(torch.nn.Module):
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__
            + "("
            + "expression="
            + str(expression_str)
            + ")"
        )


class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias=True,
                 use_bn=False):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn

        self.linear = nn.Linear(self.in_features,
                                self.out_features,
                                bias=self.use_bias and not self.use_bn)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self, x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type='nonlinear'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features, self.out_features, False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features, self.hidden_features, True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features, self.out_features, False, True))

    def forward(self, x):
        x = self.layers(x)
        return x
