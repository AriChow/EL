# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import uuid
import pathlib
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from EL.utils.callbacks import Callback, ConsoleLogger, Checkpoint, CheckpointSaver, TensorboardLogger, EarlyStopper
from EL.utils.utils import move_to
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _add_dicts(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return result


def _div_dict(d, n):
    result = dict(d)
    for k in result:
        result[k] /= n
    return result


class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            callbacks: Optional[List[Callback]] = None,
            validation_freq=1,
            train_batches_per_epoch=None,
            val_batches_per_epoch=None,
            tensorboard=None,
            patience=3,
            tensorboard_dir=None
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.train_data = train_data
        self.validation_data = validation_data
        self.validation_freq = validation_freq
        self.device = device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer.state = move_to(self.optimizer.state, self.device)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=patience, verbose=True)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches_per_epoch = val_batches_per_epoch
        if tensorboard:
            assert tensorboard_dir, 'tensorboard directory has to be specified'
            tensorboard_logger = TensorboardLogger()
            self.callbacks.append(tensorboard_logger)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        it = iter(self.validation_data)
        if self.val_batches_per_epoch is None:
            self.val_batches_per_epoch = len(self.validation_data)
        progress = tqdm.tqdm(total=self.val_batches_per_epoch, position=1, desc='Val Loss = Inf, Val Acc = 0')
        with torch.no_grad():
            for _ in range(self.val_batches_per_epoch):
                batch = next(it)
                batch = move_to(batch, self.device)
                optimized_loss, rest = self.game(*batch)
                mean_loss += optimized_loss
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
                progress.update(1)
                progress.set_description_str('Val Loss = ' + str(mean_loss.item()/n_batches) + ', Val Acc = ' + str(mean_rest['acc']/n_batches))
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()

        if self.train_batches_per_epoch is None:
            self.train_batches_per_epoch = len(self.train_data)
        progress = tqdm.tqdm(total=self.train_batches_per_epoch, position=1, desc='Train Loss = Inf, Train Acc = 0')
        it = iter(self.train_data)
        for _ in range(self.train_batches_per_epoch):
            batch = next(it)
            self.optimizer.zero_grad()
            batch = move_to(batch, self.device)
            optimized_loss, rest = self.game(*batch)
            mean_rest = _add_dicts(mean_rest, rest)
            optimized_loss.backward()
            self.optimizer.step()
            n_batches += 1
            progress.update(1)
            mean_loss += optimized_loss
            progress.set_description_str('Train Loss = ' + str(mean_loss.item()/n_batches) + ', Train Acc = ' + str(mean_rest['acc']/n_batches))

        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()
            self.scheduler.step(train_loss)
            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

