# -*- coding:utf-8 -*-
from typing import Dict
from models.utils import *


class EEGNet(nn.Module):
    # https://arxiv.org/abs/1611.08024
    def __init__(self, f1, f2, d, channel_size, input_time_length,
                 dropout_rate, sampling_rate):
        super(EEGNet, self).__init__()
        cnn_kernel_1 = int(sampling_rate // 2)

        self.cnn = nn.Sequential()
        self.cnn.add_module(
            name='conv_temporal',
            module=Conv2dWithConstraint(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, cnn_kernel_1),
                stride=1,
                bias=False,
                padding=(0, int(cnn_kernel_1 // 2))
            )
        )
        self.cnn.add_module(
            name='batch_normalization_1',
            module=nn.BatchNorm2d(f1)
        )
        self.cnn.add_module(
            name='conv_spatial',
            module=Conv2dWithConstraint(
                in_channels=f1,
                out_channels=f1 * d,
                kernel_size=(channel_size, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=f1,
                padding=(0, 0),
            )
        )
        self.cnn.add_module(
            name='batch_normalization_2',
            module=nn.BatchNorm2d(f1 * d, momentum=0.01, affine=True, eps=1e-3)
        )
        self.cnn.add_module(
            name='activation_1',
            module=nn.ELU()
        )
        self.cnn.add_module(
            name='average_pool_2d_1',
            module=nn.AvgPool2d(
                kernel_size=(1, 4)
            )
        )
        self.cnn.add_module(
            name='dropout_rate_1',
            module=nn.Dropout(dropout_rate)
        )
        self.cnn.add_module(
            name='conv_separable_depth',
            module=nn.Conv2d(
                in_channels=f1 * d,
                out_channels=f1 * d,
                kernel_size=(1, 16),
                stride=1,
                bias=False,
                groups=f1 * d,
                padding=(0, 16 // 2),
            )
        )
        self.cnn.add_module(
            name='batch_normalization_3',
            module=nn.BatchNorm2d(f2),
        )
        self.cnn.add_module(
            name='activation_2',
            module=nn.ELU()
        )
        self.cnn.add_module(
            name='average_pool_2d_2',
            module=nn.AvgPool2d(
                kernel_size=(1, 8)
            )
        )
        self.cnn.add_module(
            name='dropout_rate_2',
            module=nn.Dropout(dropout_rate)
        )
        out = self.cnn(
            np_to_var(
                np.ones(
                    (1, 1, channel_size, input_time_length),
                    dtype=np.float32,
                )
            )
        )
        self.final_length = out.reshape(-1).shape[0]

    def forward(self, x) -> Dict:
        b = x.size()[0]
        x = x.unsqueeze(dim=1)
        conv_out = self.cnn(x)
        out = torch.reshape(conv_out, [b, -1])
        return out


class ShallowConvNet(nn.Module):
    # https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730
    def __init__(self, channel_size, input_time_length, dropout_rate):
        super().__init__()
        cnn_kernel_1 = int(input_time_length // 22)
        cnn_kernel_2 = cnn_kernel_1 * 3

        self.cnn = nn.Sequential()
        self.cnn.add_module(
            name='block1_conv [temporal]',
            module=Conv2dWithConstraint(
                in_channels=1,
                out_channels=40,
                kernel_size=(1, cnn_kernel_1),
                stride=1
            )
        )
        self.cnn.add_module(
            name='block1_conv [spectral]',
            module=Conv2dWithConstraint(
                in_channels=40,
                out_channels=40,
                kernel_size=(channel_size, 1),
                bias=False,
                stride=1
            )
        )
        self.cnn.add_module(
            name='block1_bn',
            module=nn.BatchNorm2d(40, momentum=0.1, affine=True)
        )
        self.cnn.add_module(
            name='block1_square',
            module=Expression(square)
            # module=nn.ELU()
        )
        self.cnn.add_module(
            name='block1_average_pooling',
            module=nn.AvgPool2d(
                kernel_size=(1, cnn_kernel_2),
                stride=(1, int(cnn_kernel_2 // 5))
            )
        )
        self.cnn.add_module(
            name='block1_log',
            module=Expression(safe_log)
        )
        self.cnn.add_module(
            name='block1_dropout',
            module=nn.Dropout(dropout_rate)
        )
        out = self.cnn(
            np_to_var(
                np.ones(
                    (1, 1, channel_size, input_time_length),
                    dtype=np.float32,
                )
            )
        )
        self.final_length = out.reshape(-1).shape[0]

    def forward(self, x):
        b = x.size()[0]
        x = x.unsqueeze(dim=1)
        conv_out = self.cnn(x)
        out = torch.reshape(conv_out, [b, -1])
        return out


class DeepConvNet(nn.Module):
    # https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730
    def __init__(self, channel_size, input_time_length):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=False),
            nn.Conv2d(25, 25, kernel_size=(2, 1), bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.4),
            nn.Conv2d(25, 50, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.4),
            nn.Conv2d(50, 100, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.4),
            nn.Conv2d(100, 200, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.4),
        )
        self.final_length = self.get_final_length(channel_size=channel_size, input_size=input_time_length)

    def forward(self, x):
        b = x.shape[0]
        x = x.unsqueeze(dim=1)
        conv_out = self.cnn(x)
        out = torch.reshape(conv_out, [b, -1])
        return out

    def get_final_length(self, channel_size, input_size):
        x = torch.randn(1, channel_size, input_size)
        x = self.forward(x)
        return x.shape[-1]


class MultiResolutionCNN(nn.Module):
    def __init__(self, f1, f2, d, channel_size, dropout_rate, input_time_length, sampling_rate):
        super().__init__()
        cnn_kernel_size1 = sampling_rate
        cnn_kernel_size2 = sampling_rate // 2
        cnn_kernel_size3 = sampling_rate // 4
        self.stm1 = SpatialTemporalCNN(f1=f1, f2=f2, d=d, channel_size=channel_size,
                                       input_time_length=input_time_length, dropout_rate=dropout_rate,
                                       kernel_size=cnn_kernel_size1)
        self.stm2 = SpatialTemporalCNN(f1=f1, f2=f2, d=d, channel_size=channel_size,
                                       input_time_length=input_time_length, dropout_rate=dropout_rate,
                                       kernel_size=cnn_kernel_size2)
        self.stm3 = SpatialTemporalCNN(f1=f1, f2=f2, d=d, channel_size=channel_size,
                                       input_time_length=input_time_length, dropout_rate=dropout_rate,
                                       kernel_size=cnn_kernel_size3)
        self.final_length = self.stm1.final_length + self.stm2.final_length + self.stm3.final_length

    def forward(self, x):
        o1 = self.stm1(x)
        o2 = self.stm2(x)
        o3 = self.stm3(x)
        o = torch.cat([o1, o2, o3], dim=-1)
        return o


class SpatialTemporalCNN(nn.Module):
    # https://arxiv.org/abs/1611.08024
    def __init__(self, f1, f2, d, channel_size, input_time_length, dropout_rate, kernel_size):
        super().__init__()
        self.cnn = nn.Sequential()
        self.cnn.add_module(
            name='conv_temporal',
            module=Conv2dWithConstraint(in_channels=1, out_channels=f1, kernel_size=(1, kernel_size),
                                        stride=1, bias=False, padding=(0, int(kernel_size // 2)))
        )
        self.cnn.add_module(
            name='batch_normalization_1',
            module=nn.BatchNorm2d(f1)
        )
        self.cnn.add_module(
            name='conv_spatial',
            module=Conv2dWithConstraint(in_channels=f1, out_channels=f1 * d, kernel_size=(channel_size, 1),
                                        max_norm=1, stride=1, bias=False, groups=f1, padding=(0, 0))
        )
        self.cnn.add_module(
            name='batch_normalization_2',
            module=nn.BatchNorm2d(f1 * d, momentum=0.01, affine=True, eps=1e-3)
        )
        self.cnn.add_module(
            name='activation_1',
            module=nn.ELU()
        )
        self.cnn.add_module(
            name='average_pool_2d_1',
            module=nn.AvgPool2d(kernel_size=(1, 4))
        )
        self.cnn.add_module(
            name='dropout_rate_1',
            module=nn.Dropout(dropout_rate)
        )
        self.cnn.add_module(
            name='conv_separable_depth',
            module=nn.Conv2d(in_channels=f1 * d, out_channels=f1 * d, kernel_size=(1, 16), stride=(1, 1),
                             bias=False, groups=f1 * d, padding=(0, 16 // 2),
            )
        )
        self.cnn.add_module(
            name='batch_normalization_3',
            module=nn.BatchNorm2d(f2),
        )
        self.cnn.add_module(
            name='activation_2',
            module=nn.ELU()
        )
        self.cnn.add_module(
            name='average_pool_2d_2',
            module=nn.AvgPool2d(
                kernel_size=(1, 8)
            )
        )
        out = self.cnn(
            np_to_var(
                np.ones(
                    (1, 1, channel_size, input_time_length),
                    dtype=np.float32,
                )
            )
        )
        self.final_length = out.reshape(-1).shape[0]

    def forward(self, x) -> Dict:
        b = x.size()[0]
        x = x.unsqueeze(dim=1)
        conv_out = self.cnn(x)
        out = torch.reshape(conv_out, [b, -1])
        return out
