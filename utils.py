from mnist import MNIST
import torch


class Data:
    def __init__(self):
        mnist_data = MNIST('samples')
        training_images, training_labels = mnist_data.load_training()
        test_images, test_labels = mnist_data.load_testing()

        self.training_images = torch.tensor(training_images).type(torch.FloatTensor) / 255
        self.training_labels = torch.zeros([len(training_labels), 10])
        for i in range(len(training_labels)):
            self.training_labels[i, training_labels[i]] = 0.9
        self.training_labels = self.training_labels + 0.01

        self.test_images = torch.tensor(test_images).type(torch.FloatTensor) / 255
        self.test_labels = torch.zeros([len(test_labels), 10])
        for i in range(len(test_labels)):
            self.test_labels[i, test_labels[i]] = 1


class DataTwoDim:
    def __init__(self):
        mnist_data = MNIST('samples')
        training_images, training_labels = mnist_data.load_training()
        test_images, test_labels = mnist_data.load_testing()

        self.training_images = torch.tensor(training_images).type(torch.FloatTensor) / 255
        self.training_images = torch.reshape(self.training_images, (len(training_images), 28, 28)).unsqueeze(1)
        self.training_labels = torch.tensor(training_labels)
        #self.training_labels = torch.zeros([len(training_labels), 10])
        #for i in range(len(training_labels)):
        #    self.training_labels[i, training_labels[i]] = 1

        self.test_images = torch.tensor(test_images).type(torch.FloatTensor) / 255
        self.test_images = torch.reshape(self.test_images, (len(test_images), 28, 28)).unsqueeze(1)
        self.test_labels = torch.zeros([len(test_labels), 10])
        for i in range(len(test_labels)):
            self.test_labels[i, test_labels[i]] = 1
