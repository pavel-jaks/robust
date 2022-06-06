import time

import torch
import torch.nn as nn

import utils
from utils import Clipper

class Coach:
    @staticmethod
    def train(
        model: nn.Module,
        training_data: utils.AbstractData,
        loss_function,
        optimizer,
        batch_size=100,
        epochs=1000,
        noisy=500,
        too_noisy=False
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
                if too_noisy:
                    perf = Coach.measure_performance(
                        model,
                        *training_data.get_training_batch(100),
                        training_data.get_data_storage_type(),
                        False
                    )
                else:
                    perf = -1
                print(f'Epoch {epoch}: {loss.item()}: {perf}')
        tt = time.time()
        print(f'Training finished at {tt}; lasted {tt - t} seconds.')

    @staticmethod
    def measure_performance(
        model: nn.Module,
        examples: torch.Tensor,
        labels: torch.Tensor,
        data_storage_type: utils.DataStorageType,
        noisy: bool = True
    ):
        preds = len(examples)
        succeeded = 0
        if data_storage_type == utils.DataStorageType.ClassificationWithIndexes:
            test_preds = model(examples)
            indeces = torch.argmax(test_preds, dim=1)
            for i in range(len(indeces)):
                if indeces[i] == labels[i]:
                    succeeded += 1
        else:
            raise NotImplementedError(f'Data storage type {data_storage_type.name} not implemented.')
        if noisy:
            print(f'{succeeded / preds * 100} % success on given data')
        return succeeded / preds

    @staticmethod
    def train_robust(
        model: nn.Module,
        training_data: utils.AbstractData,
        loss_function,
        optimizer,
        ball_radius: float,
        pgd_step_size: float,
        pgd_epochs: int,
        batch_size=100,
        epochs=1000,
        noisy=500,
        too_noisy=False,
        minimum=0, maximum=1
        ):
        t = time.time()
        print(f'Training started at {t}')
        for epoch in range(epochs):
            data, labels = training_data.get_training_batch(batch_size)
            altered_data = data.detach() + 2 * ball_radius * (torch.rand(data.shape) - 0.5)
            for i in range(pgd_epochs):
                altered_data.requires_grad = True
                if altered_data.grad is not None:
                    altered_data.grad.zero_()
                if i == pgd_epochs - 1:
                    is_retain = True
                else:
                    is_retain = False
                loss = loss_function(model(altered_data), labels).backward(retain_graph=is_retain)
                altered_data = Clipper.clip(
                    data,
                    altered_data + pgd_step_size * altered_data.grad.apply_(lambda x: 1 if x >= 0 else -1),
                    ball_radius,
                    minimum,
                    maximum
                )

            altered_data = altered_data.detach()
            loss = loss_function(model(altered_data), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % noisy == 0 or epoch == 0:
                if too_noisy:
                    perf = Coach.measure_performance(
                        model,
                        *training_data.get_training_batch(100),
                        training_data.get_data_storage_type(),
                        False
                    )
                else:
                    perf = -1
                print(f'Epoch {epoch}: {loss.item()}: {perf}')
        tt = time.time()
        print(f'Training finished at {tt}; lasted {tt - t} seconds.')
