import torch
import torch.nn as nn

__all__ = ["SparseMax"]


class SparseMax(nn.Module):
    def __init__(self, K):
        super(SparseMax, self).__init__()
        self.K = K

    def forward(self, x):
        # For nurmercial stability
        x = x - torch.max(x, dim=1, keepdim=True)[0].expand_as(x)

        # Sort values in each channel descending order
        zs = torch.sort(x, dim=1, descending=True)[0]

        # Get shape of (1, -1, ...) for base shape for expansion
        ndim = x.ndim
        base_shape = [1 for _ in range(ndim)]
        base_shape[1] = -1

        # Set indexed range
        idx_range = torch.arange(
            start=1, end=self.K + 1, step=1, device=x.device, dtype=x.dtype
        ).view(base_shape)
        idx_range = idx_range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + idx_range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim=1)
        is_gt = (bound > cumulative_sum_zs).type(x.type())
        k = torch.max(is_gt * idx_range, dim=1, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim=1, keepdim=True) - 1) / k
        taus = taus.expand_as(x)

        # Sparsemax operation
        x = torch.max(torch.zeros_like(x), x - taus)

        return x
