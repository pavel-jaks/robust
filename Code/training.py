import imp
import time

import torch.nn as nn
import torch.optim as optim

import utils

class Coach:
    @staticmethod
    def train(
        model: nn.Module,
        training_data: utils.AbstractData,
        loss_function,
        optimizer,
        batch_size=100,
        epochs=1000,
        noisy=500
        ):
        t = time.time()
        print(f'Training started at {t}')
        for epoch in range(epochs):
            data, labels = training_data.get_training_batch(batch_size)
            loss = loss_function(model(data), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % noisy == 0 or epoch == 0:
                print(f'Epoch {epoch}: {loss.item()}')
        tt = time.time()
        print(f'Training finished at {tt}; lasted {tt - t} seconds.')

    @staticmethod
    def measure_performance(model: nn.Module, data: utils.AbstractData):
        preds = 0
        succeeded = 0
        if data.get_data_storage_type() == utils.DataStorageType.ClassificationWithIndexes:
            test_preds = model(data.get_test_data())    
            for i in range(len(test_preds)):
                prob_distr = test_preds[i]
                label = data.get_test_labels()[i]
                index = 0
                m = max(prob_distr)
                for j in range(len(prob_distr)):
                    if m == prob_distr[j]:
                        index = j
                        break
                if index == label:
                    succeeded += 1
                preds += 1
        print(f'{succeeded / preds * 100} % success on test data')
        return succeeded / preds

    @staticmethod
    def train_combined(model: nn.Module, training_data: utils.AbstractData, loss_function):
        print('Training Phase 1')
        Coach.train(
            model,
            training_data,
            loss_function,
            optim.SGD(model.parameters(), lr=1e-2),
            30,
            5001,
            1000
        )
        print('Training Phase 2')
        Coach.train(
            model,
            training_data,
            loss_function,
            optim.SGD(model.parameters(), lr=1e-3),
            100,
            1001,
            100
        )
        print('Training Phase 3')
        Coach.train(
            model,
            training_data,
            loss_function,
            optim.SGD(model.parameters(), lr=1e-5),
            1000,
            501,
            50
        )
        print('All training Phases finished')
