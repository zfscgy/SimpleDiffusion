from typing import List, Dict, Iterator, Callable, Any

import io

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from deep_zf.convert import Convert
from deep_zf.metrics import Metric, ComposedMetric


class ZModel:
    """
    Trainable model
    """
    def __init__(self):
        self.batch_loss_record = []

    def forward(self, xs: torch.Tensor) -> Any:
        raise NotImplemented()

    def compute_loss(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()

    def train_step(self, xs: torch.Tensor, ys: torch.Tensor) -> float:
        raise NotImplemented()

    def save(self):
        raise NotImplemented()

    def load(self, saved_obj):
        raise NotImplemented()

    def set_train(self, mode: bool):
        raise NotImplemented()


class SimpleModel(ZModel):
    def __init__(self,
                 model: nn.Module,
                 loss_func: Callable[[Any, Any], torch.Tensor] = None,
                 get_optimizer: Callable[[Iterator[nn.Parameter]], Optimizer] = None):
        self.model = model
        self.loss_func = loss_func
        if get_optimizer is not None:
            self.optimizer = get_optimizer(model.parameters())
            
        super(SimpleModel, self).__init__()

    def forward(self, xs: torch.Tensor) -> Any:
        return self.model(xs)

    def compute_loss(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        loss = self.loss_func(self.model(xs), ys)
        self.batch_loss_record.append([loss.item()])
        return loss

    def train_step(self, xs: torch.Tensor, ys: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.compute_loss(xs, ys)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        binary_stream = io.BytesIO()
        torch.save(self.model.state_dict(), binary_stream)
        binary_stream.seek(0)
        return torch.load(binary_stream)

    def load(self, saved_obj):
        self.model.load_state_dict(saved_obj)

    def set_train(self, mode: bool):
        self.model.train(mode)


class TrainMonitor:
    def __init__(self, trainer):
        self.trainer = trainer

    def check(self):
        raise NotImplemented()

    def should_stop(self):
        raise NotImplemented()

    def is_new_best(self):
        raise NotImplemented()


class Trainer:
    def __init__(self, z_model: ZModel,
                 train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                 metrics: List[Metric],
                 save_path: str = None):
        self.z_model = z_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics = metrics

        self.save_path = save_path or "train_record_default"

        self.m_train_iterator = iter(train_loader)

        self.best_model = None
        self.metric_vals = []
        self.current_round = None

    def train_n_batches(self, n_batches: int):
        train_losses = []
        for i in range(n_batches):
            try:
                xs, ys = next(self.m_train_iterator)
            except StopIteration:
                self.m_train_iterator = iter(self.train_loader)
                xs, ys = next(self.m_train_iterator)

            xs, ys = Convert.to_tensor([xs, ys])

            loss = self.z_model.train_step(xs, ys)
            train_losses.append(loss)

        return np.mean(train_losses)

    def train_n_epochs(self, n_epochs: int):
        train_losses = []
        for i in range(n_epochs):
            for xs, ys in self.train_loader:
                xs, ys = Convert.to_tensor([xs, ys])
                loss = self.z_model.train_step(xs, ys)
                train_losses.append(loss)

        return np.mean(train_losses)

    def train(self, n_rounds: int, monitor: TrainMonitor = None, mode: str = "1 epoch", save_model: bool = True):
        try:
            inner_rounds, unit = mode.split(" ")
            inner_rounds = int(inner_rounds)
        except:
            print(f"Illegal mode: {mode}, mode must be 'number epoch/round'")
            return


        # Test the initial model
        current_metric_vals = dict()
        current_metric_vals['round'] = 'init'
        # Compute metric on the validation data loader
        if self.val_loader is not None:
            current_metric_vals.update(self.compute_metric_on_loader())
            self.metric_vals.append(current_metric_vals)

        print(current_metric_vals)
        pd.DataFrame(self.metric_vals).to_csv(self.save_path + "--metric.csv")

        assert unit in ["epoch", "batch"], f"Illegal mode: {mode}, mode must be 'number epoch/round'"
        for i in range(n_rounds):
            self.current_round = i

            self.z_model.set_train(True)
            if unit == "batch":
                self.train_n_batches(inner_rounds)
            else:
                self.train_n_epochs(inner_rounds)
            self.z_model.set_train(False)

            current_metric_vals = dict()
            current_metric_vals['round'] = i * inner_rounds
            n_losses = len(self.z_model.batch_loss_record[0])
            batch_losses = np.array(self.z_model.batch_loss_record)
            for j in range(n_losses):
                current_metric_vals[f'loss{j}'] = np.mean(batch_losses[:, j])
            # Clear batch loss record
            self.z_model.batch_loss_record.clear()

            # Compute metric on the validation data loader
            if self.val_loader is not None:
                current_metric_vals.update(self.compute_metric_on_loader())
                self.metric_vals.append(current_metric_vals)

            print(current_metric_vals)
            pd.DataFrame(self.metric_vals).to_csv(self.save_path + "--metric.csv")

            # Monitor check
            if monitor is not None:
                monitor.check()
                should_stop = monitor.should_stop()
                is_best = monitor.is_new_best()
                if is_best:
                    self.best_model = self.z_model.save()
                if should_stop:
                    break

        if monitor is None or self.best_model is None:
            self.best_model = self.z_model.save()

        if save_model:
            torch.save(self.best_model, self.save_path + "--best.pth")
        # Compute metric on the test data loader
        if self.test_loader is not None:
            self.z_model.load(self.best_model)
            current_metric_vals = self.compute_metric_on_loader(data_loader=self.test_loader)
            current_metric_vals['round'] = -1
            self.metric_vals.append(current_metric_vals)
            pd.DataFrame(self.metric_vals).to_csv(self.save_path + "--metric.csv")

    def compute_metric_on_loader(self, metrics: List[Metric] = None, data_loader: DataLoader = None) -> Dict[str, float]:
        metrics = metrics or self.metrics
        data_loader = data_loader or self.val_loader

        output_dict = dict()
        for metric in metrics:
            if metric.use_batch_mean:
                output_dict[metric.name] = [0, 0]
            else:
                output_dict[metric.name] = ([], [])  # concatenated xs/ys

        for xs, ys in data_loader:
            xs, ys = Convert.to_tensor([xs, ys])
            pred_ys = self.z_model.forward(xs)
            for metric in metrics:
                if metric.use_batch_mean:
                    output_dict[metric.name][0] += metric.compute(pred_ys, ys) * ys.shape[0]
                    output_dict[metric.name][1] += ys.shape[0]
                else:
                    if isinstance(metric, ComposedMetric):
                        pred_ys_wrapped, ys_wrapped = metric.input_wrapper(pred_ys, ys)
                    else:
                        pred_ys_wrapped, ys_wrapped = pred_ys, ys
                    output_dict[metric.name][0].append(pred_ys_wrapped)
                    output_dict[metric.name][1].append(ys_wrapped)

        for metric in metrics:
            if metric.use_batch_mean:
                output_dict[metric.name] = output_dict[metric.name][0] / output_dict[metric.name][1]
            else:
                concated_xs = torch.concat(output_dict[metric.name][0])
                concated_ys = torch.concat(output_dict[metric.name][1])
                output_dict[metric.name] = metric.compute(concated_xs, concated_ys)
        return output_dict


class EarlyStopper(TrainMonitor):
    def __init__(self, trainer: Trainer, metric: Metric, start_round: int = 0, mean_wsize: int = 5, larger_better: bool = True,
                 additional_callback: Callable[[Trainer], Any] = None):
        super(EarlyStopper, self).__init__(trainer)
        self.metric = metric
        self.start_round = start_round
        self.mean_wsize = mean_wsize
        self.larger_better = larger_better
        self.main_metric_vals = []

        self.additional_callback = additional_callback

    def check(self):
        if self.metric.name in self.trainer.metric_vals[-1]:
            metric_val = self.trainer.metric_vals[-1][self.metric.name]
        else:
            metric_val = self.trainer.compute_metric_on_loader([self.metric], self.trainer.test_loader)[self.metric.name]
        self.main_metric_vals.append(metric_val)
        if self.additional_callback is not None:
            self.additional_callback(self.trainer)

    def should_stop(self) -> bool:
        if len(self.main_metric_vals) > max(2 * self.mean_wsize, self.start_round):
            mean_pre = np.mean(self.main_metric_vals[-2 * self.mean_wsize: -self.mean_wsize])
            mean_after = np.mean(self.main_metric_vals[-self.mean_wsize:])

            if self.larger_better and (mean_after < mean_pre):
                return True
            if (not self.larger_better) and (mean_after > mean_pre):
                return True

        return False

    def is_new_best(self) -> bool:
        if len(self.main_metric_vals) < self.start_round:
            return False

        if self.larger_better:
            return len(self.main_metric_vals) <= self.start_round + 1 or \
                   self.main_metric_vals[-1] > np.max(self.main_metric_vals[self.start_round: -1])
        else:
            return len(self.main_metric_vals) <= self.start_round + 1 or \
                    self.main_metric_vals[-1] < np.min(self.main_metric_vals[self.start_round: -1])

