# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Dict, Any, Union,  NamedTuple, List, Tuple
import pathlib

import torch

from egg.core.util import get_summary_writer


class Callback:
    trainer: 'Trainer'

    def on_train_begin(self, trainer_instance: 'Trainer'):
        self.trainer = trainer_instance

    def on_train_end(self):
        pass

    def on_test_begin(self):
        pass

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        pass


class ConsoleLogger(Callback):

    def __init__(self, print_train_loss=False, as_json=False):
        self.print_train_loss = print_train_loss
        self.as_json = as_json
        self.epoch_counter = 0

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None):
        if self.as_json:
            dump = dict(mode='test', epoch=self.epoch_counter, loss=self._get_metric(loss))
            for k, v in logs.items():
                dump[k] = self._get_metric(v)
            output_message = json.dumps(dump)
        else:
            output_message = f'test: epoch {self.epoch_counter}, loss {loss},  {logs}'
        print(output_message, flush=True)

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        self.epoch_counter += 1

        if self.print_train_loss:
            if self.as_json:
                dump = dict(mode='train', epoch=self.epoch_counter, loss=self._get_metric(loss))
                for k, v in logs.items():
                    dump[k] = self._get_metric(v)
                output_message = json.dumps(dump)
            else:
                output_message = f'train: epoch {self.epoch_counter}, loss {loss},  {logs}'
            print(output_message, flush=True)

    def _get_metric(self, metric: Union[torch.Tensor, float]) -> float:
        if torch.is_tensor(metric) and metric.dim() > 1:
            return metric.mean().item()
        elif torch.is_tensor(metric):
            return metric.item()
        elif type(metric) == float:
            return metric
        else:
            raise TypeError('Metric must be either float or torch.Tensor')


class TensorboardLogger(Callback):

    def __init__(self, writer=None):
        if writer:
            self.writer = writer
        else:
            self.writer = get_summary_writer()
        self.epoch_counter = 0

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None):
        self.writer.add_scalar(tag=f'test/loss', scalar_value=loss, global_step=self.epoch_counter)
        for k, v in logs.items():
            self.writer.add_scalar(tag=f'test/{k}', scalar_value=v, global_step=self.epoch_counter)

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        self.writer.add_scalar(tag=f'train/loss', scalar_value=loss, global_step=self.epoch_counter)
        for k, v in logs.items():
            self.writer.add_scalar(tag=f'train/{k}', scalar_value=v, global_step=self.epoch_counter)
        self.epoch_counter += 1

    def on_train_end(self):
        self.writer.close()


class TemperatureUpdater(Callback):

    def __init__(self, agent, decay=0.9, minimum=0.1, update_frequency=1):
        self.agent = agent
        assert hasattr(agent, 'temperature'), 'Agent must have a `temperature` attribute'
        assert not isinstance(agent.temperature, torch.nn.Parameter), \
            'When using TemperatureUpdater, `temperature` cannot be trainable'
        self.decay = decay
        self.minimum = minimum
        self.update_frequency = update_frequency
        self.epoch_counter = 0

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        if self.epoch_counter % self.update_frequency == 0:
            self.agent.temperature = max(self.minimum, self.agent.temperature * self.decay)
        self.epoch_counter += 1


class Checkpoint(NamedTuple):
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]


class CheckpointSaver(Callback):

    def __init__(
            self,
            checkpoint_path: Union[str, pathlib.Path],
            checkpoint_freq: int = 1,
            prefix: str = ''
    ):
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.checkpoint_freq = checkpoint_freq
        self.prefix = prefix
        self.epoch_counter = 0

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        if self.checkpoint_freq > 0 and (self.epoch_counter % self.checkpoint_freq == 0):
            filename = f'{self.prefix}_{self.epoch_counter}' if self.prefix else str(self.epoch_counter)
            self.save_checkpoint(filename=filename)
        self.epoch_counter += 1

    def on_train_end(self):
        self.save_checkpoint(filename=f'{self.prefix}_final' if self.prefix else 'final')

    def save_checkpoint(self, filename: str):
        """
        Saves the game, agents, and optimizer states to the checkpointing path under `<number_of_epochs>.tar` name
        """
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)
        path = self.checkpoint_path / f'{filename}.tar'
        torch.save(self.get_checkpoint(), path)

    def get_checkpoint(self):
        return Checkpoint(epoch=self.epoch_counter,
                          model_state_dict=self.trainer.game.state_dict(),
                          optimizer_state_dict=self.trainer.optimizer.state_dict())


class BaseEarlyStopper(Callback):
    """
    A base class, supports the running statistic which is could be used for early stopping
    """

    def __init__(self):
        super(BaseEarlyStopper, self).__init__()
        self.train_stats: List[Tuple[float, Dict[str, Any]]] = []
        self.validation_stats: List[Tuple[float, Dict[str, Any]]] = []
        self.epoch: int = 0

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None) -> None:
        self.epoch += 1
        self.train_stats.append((loss, logs))

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None) -> None:
        self.validation_stats.append((loss, logs))
        self.trainer.should_stop = self.should_stop()

    def should_stop(self) -> bool:
        raise NotImplementedError()

class EarlyStopper(BaseEarlyStopper):
    """
    Implements early stopping using patience on validation loss
    """
    def __init__(self, patience: int = 100, delta: int = 0, save: str ='checkpoint.pth', validation: bool = True, verbose : bool = True) -> None:
        super(EarlyStopper, self).__init__()
        self.patience = patience
        self.delta = delta
        self.save = save
        self.validation = validation
        self.counter = 0
        self.best_score = None
        self.verbose = verbose

    def should_stop(self) -> bool:
        if self.validation:
            assert self.validation_stats, 'Validation data must be provided for early stooping to work'
            stats = self.validation_stats
        else:
            assert self.train_stats, 'Training data must be provided for early stooping to work'
            stats = self.train_stats
        loss = stats[-1][0]
        if self.best_score is None:
            self.best_score = loss
            self.save_checkpoint(loss, self.trainer.game)
        else:
            if loss > self.best_score + self.delta:
                self.counter += 1
            else:
                self.save_checkpoint(loss, self.trainer.game)
                self.best_score = loss
                self.counter = 0
        return self.counter >= self.patience

    def save_checkpoint(self, loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save)

class EarlyStopperAccuracy(BaseEarlyStopper):
    """
    Implements early stopping logic that stops training when a threshold on a metric
    is achieved.
    """

    def __init__(self, threshold: float, field_name: str = 'acc', validation: bool = True) -> None:
        """
        :param threshold: early stopping threshold for the validation set accuracy
            (assumes that the loss function returns the accuracy under name `field_name`)
        :param field_name: the name of the metric return by loss function which should be evaluated against stopping
            criterion (default: "acc")
        :param validation: whether the statistics on the validation (or training, if False) data should be checked
        """
        super(EarlyStopperAccuracy, self).__init__()
        self.threshold = threshold
        self.field_name = field_name
        self.validation = validation

    def should_stop(self) -> bool:
        if self.validation:
            assert self.validation_stats, 'Validation data must be provided for early stooping to work'
            stats = self.validation_stats
        else:
            assert self.train_stats, 'Training data must be provided for early stooping to work'
            stats = self.train_stats

        return stats[-1][1][self.field_name] >= self.threshold

