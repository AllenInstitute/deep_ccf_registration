import torch
from torchmetrics import Metric


class PointwiseRMSE(Metric):
    def __init__(self, coordinate_dim=1):
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.add_state("sum_rmse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        squared_errors = (preds - target) ** 2
        per_point_se = squared_errors.sum(dim=self.coordinate_dim)
        per_point_error = torch.sqrt(per_point_se)[mask]

        self.sum_rmse += per_point_error.sum()
        self.total += len(per_point_error)

    def compute(self):
        return self.sum_rmse / self.total


class PointwiseMAE(Metric):
    def __init__(self, coordinate_dim=1):
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.add_state("sum_mae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        absolute_errors = torch.abs(preds - target)
        per_point_error = absolute_errors.sum(dim=self.coordinate_dim)[mask]

        self.sum_mae += per_point_error.sum()
        self.total += len(per_point_error)

    def compute(self):
        return self.sum_mae / self.total
