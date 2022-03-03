from abc import ABC, abstractmethod
from typing import Tuple
import random
from enum import Enum

from mnist import MNIST
import torch


class DataStorageType(Enum):
    ClassificationWithIndexes = 0
    ClassificationWithDistributions = 1


class AbstractData(ABC):
    @abstractmethod
    def get_training_data(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_training_labels(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_test_data(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_test_labels(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_training_batch(self, batch_size=100) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_data_storage_type(self) -> DataStorageType:
        pass


class MnistData(AbstractData):
    def __init__(self, requires_grad: bool = False):
        mnist_data = MNIST('samples')

        training_images, training_labels = mnist_data.load_training()
        self.training_images = torch.tensor(training_images).type(torch.FloatTensor) / 255
        self.training_images = torch.reshape(self.training_images, (len(training_images), 28, 28)).unsqueeze(1)
        self.training_images.requires_grad = requires_grad
        self.training_labels = torch.tensor(training_labels)

        test_images, test_labels = mnist_data.load_testing()
        self.test_images = torch.tensor(test_images).type(torch.FloatTensor) / 255
        self.test_images = torch.reshape(self.test_images, (len(test_images), 28, 28)).unsqueeze(1)
        self.test_images.requires_grad = requires_grad
        self.test_labels = torch.tensor(test_labels)
        
        self.data_storage_type = DataStorageType.ClassificationWithIndexes

    def get_training_data(self) -> torch.Tensor:
        return self.training_images

    def get_training_labels(self) -> torch.Tensor:
        return self.training_labels

    def get_test_data(self) -> torch.Tensor:
        return self.test_images

    def get_test_labels(self) -> torch.Tensor:
        return self.test_labels

    def get_training_batch(self, batch_size=100) -> Tuple[torch.Tensor, torch.Tensor]:
        indexes = [random.randint(0, len(self.training_images) - 1) for _ in range(batch_size)]
        images = [self.training_images[index].tolist() for index in indexes]
        labels = [self.training_labels[index] for index in indexes]
        return torch.tensor(images), torch.tensor(labels)

    def get_data_storage_type(self) -> DataStorageType:
        return self.data_storage_type
