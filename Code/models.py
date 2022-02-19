from enum import Enum
from os import path

import torch
import torch.nn as nn


class MnistCnnAlfred(nn.Module):
    """
    Simpler Mnist CNN model
    """
    def __init__(self):
        super().__init__()
        self.first_conv_layer = nn.Conv2d(1, 16, kernel_size=5)
        self.first_activation = nn.ReLU()
        self.second_layer = nn.Linear(9216, 32)
        self.second_activation = nn.ReLU()
        self.third_layer = nn.Linear(32, 16)
        self.third_activation = nn.ReLU()
        self.out_layer = nn.Linear(16, 10)
        self.out_activation = nn.Softmax(dim=1)

    def forward(self, t: torch.Tensor):
        out = self.first_activation(self.first_conv_layer(t))
        dim = 1
        for i in range(1, len(out.shape)):
            dim *= out.shape[i]
        out = out.reshape(len(t), dim)
        out = self.second_activation(self.second_layer(out))
        out = self.third_activation(self.third_layer(out))
        out = self.out_activation(self.out_layer(out))
        return out


class MnistCnnBerta(nn.Module):
    """
    Simpler Mnist CNN model
    """
    def __init__(self):
        super().__init__()
        self.first_conv_layer = nn.Conv2d(1, 16, kernel_size=5)
        self.first_activation = nn.ReLU()
        self.second_layer = nn.Linear(9216, 32)
        self.second_activation = nn.ReLU()
        self.third_layer = nn.Linear(32, 16)
        self.third_activation = nn.ReLU()
        self.out_layer = nn.Linear(16, 10)
        self.out_activation = nn.Softmax(dim=1)

    def forward(self, t: torch.Tensor):
        out = self.first_activation(self.first_conv_layer(t))
        dim = 1
        for i in range(1, len(out.shape)):
            dim *= out.shape[i]
        out = out.reshape(len(t), dim)
        out = self.second_activation(self.second_layer(out))
        out = self.third_activation(self.third_layer(out))
        out = self.out_activation(self.out_layer(out))
        return out


class MnistCnnCyril(nn.Module):
    """
    Seven layer CNN
    """
    def __init__(self):
        super().__init__()
        self.first_conv_layer = nn.Conv2d(1, 16, kernel_size=5)
        self.first_activation = nn.ReLU()
        self.second_pooling = nn.MaxPool2d(2)
        self.second_activation = nn.ReLU()
        self.third_conv = nn.Conv2d(16, 10, kernel_size=3)
        self.third_activation = nn.ReLU()
        self.fourth_pooling = nn.MaxPool2d(2)
        self.fourth_activation = nn.ReLU()
        self.fifth_linear = nn.Linear(10 * 5 * 5, 32)
        self.fifth_activation = nn.ReLU()
        self.sixth_linear = nn.Linear(32, 16)
        self.sixth_activation = nn.ReLU()
        self.seventh_linear = nn.Linear(16, 10)
        self.seventh_activation = nn.Softmax(dim=1)
    
    def forward(self, t: torch.Tensor):
        out = self.first_activation(self.first_conv_layer(t))
        out = self.second_activation(self.second_pooling(out))
        out = self.third_activation(self.third_conv(out))
        out = self.fourth_activation(self.fourth_pooling(out))
        dim = 1
        for i in range(1, len(out.shape)):
            dim *= out.shape[i]
        out = out.reshape(len(t), dim)
        out = self.fifth_activation(self.fifth_linear(out))
        out = self.sixth_activation(self.sixth_linear(out))
        out = self.seventh_activation(self.seventh_linear(out))
        return out


class MnistCnnDouglas(nn.Module):
    """
    Seven layer CNN
    """
    def __init__(self):
        super().__init__()
        self.first_conv_layer = nn.Conv2d(1, 16, kernel_size=5)
        self.first_activation = nn.ReLU()
        self.second_pooling = nn.MaxPool2d(2)
        self.second_activation = nn.ReLU()
        self.third_conv = nn.Conv2d(16, 10, kernel_size=3)
        self.third_activation = nn.ReLU()
        self.fourth_pooling = nn.MaxPool2d(2)
        self.fourth_activation = nn.ReLU()
        self.fifth_linear = nn.Linear(10 * 5 * 5, 32)
        self.fifth_activation = nn.ReLU()
        self.sixth_linear = nn.Linear(32, 16)
        self.sixth_activation = nn.ReLU()
        self.seventh_linear = nn.Linear(16, 10)
        self.seventh_activation = nn.Softmax(dim=1)
    
    def forward(self, t: torch.Tensor):
        out = self.first_activation(self.first_conv_layer(t))
        out = self.second_activation(self.second_pooling(out))
        out = self.third_activation(self.third_conv(out))
        out = self.fourth_activation(self.fourth_pooling(out))
        dim = 1
        for i in range(1, len(out.shape)):
            dim *= out.shape[i]
        out = out.reshape(len(t), dim)
        out = self.fifth_activation(self.fifth_linear(out))
        out = self.sixth_activation(self.sixth_linear(out))
        out = self.seventh_activation(self.seventh_linear(out))
        return out


class MnistCnnEdmund(nn.Module):
    """
    Seven layer CNN
    """
    def __init__(self):
        super().__init__()
        self.first_conv_layer = nn.Conv2d(1, 16, kernel_size=5)
        self.first_activation = nn.ReLU()
        self.second_pooling = nn.MaxPool2d(2)
        self.second_activation = nn.ReLU()
        self.third_conv = nn.Conv2d(16, 10, kernel_size=3)
        self.third_activation = nn.ReLU()
        self.fourth_pooling = nn.MaxPool2d(2)
        self.fourth_activation = nn.ReLU()
        self.fifth_linear = nn.Linear(10 * 5 * 5, 32)
        self.fifth_activation = nn.ReLU()
        self.sixth_linear = nn.Linear(32, 16)
        self.sixth_activation = nn.ReLU()
        self.seventh_linear = nn.Linear(16, 10)
        self.seventh_activation = nn.Softmax(dim=1)
    
    def forward(self, t: torch.Tensor):
        out = self.first_activation(self.first_conv_layer(t))
        out = self.second_activation(self.second_pooling(out))
        out = self.third_activation(self.third_conv(out))
        out = self.fourth_activation(self.fourth_pooling(out))
        dim = 1
        for i in range(1, len(out.shape)):
            dim *= out.shape[i]
        out = out.reshape(len(t), dim)
        out = self.fifth_activation(self.fifth_linear(out))
        out = self.sixth_activation(self.sixth_linear(out))
        out = self.seventh_activation(self.seventh_linear(out))
        return out


class MnistCnnFiona(nn.Module):
    """
    Seven layer CNN
    """
    def __init__(self):
        super().__init__()
        self.first_conv_layer = nn.Conv2d(1, 16, kernel_size=5)
        self.first_activation = nn.ReLU()
        self.second_pooling = nn.MaxPool2d(2)
        self.second_activation = nn.ReLU()
        self.third_conv = nn.Conv2d(16, 10, kernel_size=3)
        self.third_activation = nn.ReLU()
        self.fourth_pooling = nn.MaxPool2d(2)
        self.fourth_activation = nn.ReLU()
        self.fifth_linear = nn.Linear(10 * 5 * 5, 32)
        self.fifth_activation = nn.ReLU()
        self.sixth_linear = nn.Linear(32, 16)
        self.sixth_activation = nn.ReLU()
        self.seventh_linear = nn.Linear(16, 10)
        self.seventh_activation = nn.Softmax(dim=1)
    
    def forward(self, t: torch.Tensor):
        out = self.first_activation(self.first_conv_layer(t))
        out = self.second_activation(self.second_pooling(out))
        out = self.third_activation(self.third_conv(out))
        out = self.fourth_activation(self.fourth_pooling(out))
        dim = 1
        for i in range(1, len(out.shape)):
            dim *= out.shape[i]
        out = out.reshape(len(t), dim)
        out = self.fifth_activation(self.fifth_linear(out))
        out = self.sixth_activation(self.sixth_linear(out))
        out = self.seventh_activation(self.seventh_linear(out))
        return out


class MnistCnnGerta(nn.Module):
    """
    Simpler Mnist CNN model
    """
    def __init__(self):
        super().__init__()
        self.first_conv_layer = nn.Conv2d(1, 16, kernel_size=5)
        self.first_activation = nn.ReLU()
        self.second_layer = nn.Linear(9216, 32)
        self.second_activation = nn.ReLU()
        self.third_layer = nn.Linear(32, 16)
        self.third_activation = nn.ReLU()
        self.out_layer = nn.Linear(16, 10)
        self.out_activation = nn.Softmax(dim=1)

    def forward(self, t: torch.Tensor):
        out = self.first_activation(self.first_conv_layer(t))
        dim = 1
        for i in range(1, len(out.shape)):
            dim *= out.shape[i]
        out = out.reshape(len(t), dim)
        out = self.second_activation(self.second_layer(out))
        out = self.third_activation(self.third_layer(out))
        out = self.out_activation(self.out_layer(out))
        return out


class MnistMLPHubert(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_linear = nn.Linear(28 * 28, 16)
        self.first_activation = nn.ReLU()
        self.second_linear = nn.Linear(16, 16)
        self.second_activation = nn.ReLU()
        self.third_linear = nn.Linear(16, 10)
        self.third_activation = nn.ReLU()
        self.fourth_linear = nn.Linear(10, 10)
        self.fourth_activation = nn.Softmax(dim=1)

    def forward(self, t: torch.Tensor):
        out = t.reshape(len(t), 28 * 28)
        out = self.first_activation(self.first_linear(out))
        out = self.second_activation(self.second_linear(out))
        out = self.third_activation(self.third_linear(out))
        return self.fourth_activation(self.fourth_linear(out))


class MnistCnnIvan(nn.Module):
    """
    Simpler Mnist CNN model
    """
    def __init__(self):
        super().__init__()
        self.first_conv_layer = nn.Conv2d(1, 16, kernel_size=5)
        self.first_activation = nn.ReLU()
        self.second_layer = nn.Linear(9216, 32)
        self.second_activation = nn.ReLU()
        self.third_layer = nn.Linear(32, 16)
        self.third_activation = nn.ReLU()
        self.out_layer = nn.Linear(16, 10)
        self.out_activation = nn.Softmax(dim=1)

    def forward(self, t: torch.Tensor):
        out = self.first_activation(self.first_conv_layer(t))
        dim = 1
        for i in range(1, len(out.shape)):
            dim *= out.shape[i]
        out = out.reshape(len(t), dim)
        out = self.second_activation(self.second_layer(out))
        out = self.third_activation(self.third_layer(out))
        out = self.out_activation(self.out_layer(out))
        return out


class ModelType(Enum):
    MnistCnnAlfred = MnistCnnAlfred
    MnistCnnBerta = MnistCnnBerta
    MnistCnnCyril = MnistCnnCyril
    MnistCnnDouglas = MnistCnnDouglas
    MnistCnnEdmund = MnistCnnEdmund
    MnistCnnFiona = MnistCnnFiona
    MnistCnnGerta = MnistCnnGerta
    MnistMLPHubert = MnistMLPHubert
    MnistCnnIvan = MnistCnnIvan

class ModelManager:
    @staticmethod
    def get_untrained(model_type: ModelType) -> nn.Module:
        return model_type.value()
    
    @staticmethod
    def get_trained(model_type: ModelType) -> nn.Module:
        try:
            return torch.load(path.join('models', f'{model_type.name}.model'))
        except:
            raise ValueError(f'Model with name {model_type.name} is not trained.')
    
    @staticmethod
    def save_model(model_type: ModelType, model: nn.Module):
        torch.save(model, path.join('models', f'{model_type.name}.model'))