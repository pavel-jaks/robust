import torch.nn as nn
from torch import Tensor


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv_layer = nn.Conv2d(1, 10, kernel_size=3)
        self.first_activation = nn.ReLU()
        self.second_layer = nn.Linear(10 * 28 * 28, 16)
        self.second_activation = nn.ReLU()
        self.third_layer = nn.Linear(16, 16)
        self.third_activation = nn.ReLU()
        self.out_layer = nn.Linear(16, 10)
        self.out_activation = nn.Softmax(dim=1)

    def forward(self, t: Tensor):
        out = self.first_activation(self.first_conv_layer(t))
        out = out.reshape(len(t), 10 * 28 * 28)
        out = self.second_activation(self.second_layer(out))
        out = self.third_activation(self.third_layer(out))
        out = self.out_activation(self.out_layer(out))
        return out
