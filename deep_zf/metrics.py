from typing import Tuple

from sklearn.metrics import roc_auc_score

import torch


from deep_zf.convert import Convert


class Metric:
    def __init__(self, name: str, use_batch_mean: bool = True):
        self.name = name
        self.use_batch_mean = use_batch_mean

    def compute(self, pred_ys: torch.Tensor, ys: torch.Tensor):
        raise NotImplemented()


class BinaryAcc(Metric):
    def __init__(self):
        super(BinaryAcc, self).__init__("acc", True)

    def compute(self, pred_ys: torch.Tensor, ys: torch.Tensor):
        return torch.mean((torch.round(pred_ys) == ys).float()).item()


class ClassAcc(Metric):
    def __init__(self):
        super(ClassAcc, self).__init__("acc", True)

    def compute(self, pred_ys: torch.Tensor, ys: torch.LongTensor):
        return torch.mean((torch.argmax(pred_ys, dim=-1) == ys).float()).item()


class Auc(Metric):
    def __init__(self):
        super(Auc, self).__init__("auc", False)

    def compute(self, pred_ys: torch.Tensor, ys: torch.Tensor):
        return roc_auc_score(Convert.to_numpy(ys[:, 0]), Convert.to_numpy(pred_ys[:, 0]))


class HitRatio(Metric):
    def __init__(self, n: int):
        super(HitRatio, self).__init__(f"hr@{n}", True)
        self.n = n

    def compute(self, pred_ys: torch.Tensor, ys: torch.Tensor):
        _, indices = torch.sort(pred_ys, dim=-1)
        top_n_indices = indices[:, -self.n:]  # [batch, n]
        return torch.mean(torch.sum((ys.view(-1, 1) == top_n_indices).float(), dim=-1)).item()


class NegL2Loss(Metric):
    def __init__(self):
        super(NegL2Loss, self).__init__("negL2", True)

    def compute(self, pred_ys: torch.Tensor, ys: torch.Tensor):
        return -torch.mean(torch.sum(torch.square(pred_ys - ys), dim=-1)).item()


class ComposedMetric(Metric):
    def __init__(self, original_metric: Metric, input_wrapper):
        super(ComposedMetric, self).__init__(self.original_metric.name, self.original_metric.use_batch_mean)
        self.original_metric = original_metric
        self.input_wrapper = input_wrapper

    def compute(self, pred_ys, ys):
        return self.original_metric.compute(*self.input_wrapper(pred_ys, ys))


if __name__ == '__main__':
    print(HitRatio(2).compute(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), torch.tensor([0, 0])))
