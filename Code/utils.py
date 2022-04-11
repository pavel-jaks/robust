from abc import ABC, abstractmethod
from typing import Tuple
import random
from enum import Enum

from mnist import MNIST
import torch
import numpy as np
from matplotlib import pyplot as plt


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

    @staticmethod
    def get_prediction(model, image):
        my_image = image.unsqueeze(0)
        pred = model(my_image)
        maxout = max(pred[0])
        for j in range(10):
            if pred[0, j] == maxout:
                return j, maxout

    @staticmethod
    def get_adversarials(model, benign_images, labels, altered_images):
        possible_adversarials = []
        for i in range(len(altered_images)):
            prediction, confidence = MnistData.get_prediction(model, altered_images[i])
            original_prediction, original_confidence = MnistData.get_prediction(model, benign_images[i])
            if prediction != labels[i] and original_prediction == labels[i]:
                params = {"Label": labels[i], "Prediction": prediction, "Confidence": confidence,
                        "Index": i, "OriginalPrediction": original_prediction, "OriginalConfidence": original_confidence}
                possible_adversarials.append(params)
        return possible_adversarials

    @staticmethod
    def display(image, scale=False):
        first_image = image[0].reshape((28 * 28,)).detach()
        first_image = np.array(first_image, dtype='float')
        pixels = first_image.reshape((28, 28))
        if scale:
            plt.imshow(pixels, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(pixels, cmap="gray")
        plt.show()

    @staticmethod
    def show_adversarial(adversarials, benign_images, altered_images, index):
        adversarial = adversarials[index]
        original_image = benign_images[adversarial["Index"]]
        altered_image = altered_images[adversarial["Index"]]
        mask = altered_image - original_image
        # mask = mask * 255
        # original_image = original_image * 255
        # altered_image = altered_image * 255
        MnistData.display(original_image)
        print("+++++")
        MnistData.display(mask)
        print("=====")
        MnistData.display(altered_image, scale=True)
        print(f"Label: {adversarial['Label']}, Prediction: {adversarial['Prediction']}, Confidence: {adversarial['Confidence']}")
        print(f"Original prediction: {adversarial['OriginalPrediction']}, Original confidence: {adversarial['OriginalConfidence']}")

    @staticmethod
    def clip(benign_examples: torch.Tensor, adversarial_examples: torch.Tensor, max_norm, minimum=0, maximum=1):
        difference = adversarial_examples - benign_examples
        difference = difference.detach()
        difference.apply_(lambda x: x if (abs(x) < max_norm) else (-1 if x < 0 else 1 ) * max_norm)
        clipped_stage_one = benign_examples + difference
        clipped_stage_one = clipped_stage_one.detach()
        clipped_stage_one.apply_(lambda x: maximum if x> maximum else (minimum if x < minimum else x))
        return clipped_stage_one

    @staticmethod
    def clip_for_image(examples: torch.Tensor, minimum=0, maximum=1):
        clipped_stage_one = examples
        clipped_stage_one = clipped_stage_one.detach()
        clipped_stage_one.apply_(lambda x: maximum if x> maximum else (minimum if x < minimum else x))
        return clipped_stage_one

    @staticmethod
    def clip_with_custom_norm(benign_examples: torch.Tensor, adversarial_examples: torch.Tensor, norm_function, norm_size, minimum=0, maximum=1):
        difference = adversarial_examples - benign_examples
        difference = difference.detach()
        norm = norm_function(difference)
        if norm <= norm_size:
            return adversarial_examples
        else:
            difference = (norm_size / norm) * difference
            return benign_examples + difference

    def draw_first(self, number: int, model):
        counter = 0
        indexes = []
        while len(indexes) < number:
            if MnistData.get_prediction(model, self.training_images[counter])[0] == self.training_labels[counter]:
                indexes.append(counter)
            counter += 1
        images = [self.training_images[index].tolist() for index in indexes]
        labels = [self.training_labels[index] for index in indexes]
        return torch.tensor(images), torch.tensor(labels)
