import torch


# ERROR FUNCTIONS


class SquaredError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # return (a - b) ** 2
        return torch.mean((a - b) ** 2, dim=-1)


class AbsoluteError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # return torch.abs(a - b)
        return torch.abs(a - b).mean(dim=-1)


class SquaredNormError(torch.nn.Module):
    def __init__(self, dim: int = -1) -> None:
        """_summary_

        Note that

            squared norm error = 3 * squared error

        because the former sums whereas the latter averages over the last dim.

        Parameters
        ----------
        dim : int, optional
            _description_, by default -1
        """
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.sum((a - b) ** 2, self.dim)


class SquaredCosineSimilarityError(torch.nn.CosineSimilarity):
    def __init__(self, dim: int = -1, eps: float = 1e-8) -> None:
        super().__init__(dim, eps)

    def forward(self, a, b):
        return (1.0 - super().forward(a, b)) ** 2


# DECORATORS


def MeanReduction(Loss):
    class MeanReduction(Loss):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        def forward(self, y_pred, y_true, weight=None):
            error = super().forward(y_pred, y_true)
            if weight is None:
                return error.mean()
            else:
                return torch.sum(weight * error) / weight.sum()

    return MeanReduction


def SemiHardReduction(Loss):
    class SemiHardReduction(Loss):
        def __init__(self, upper_split: float, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            assert 0 <= upper_split <= 0.5
            self.upper_split = upper_split

        def forward(self, y_pred, y_true, weight=None):
            B,N = y_pred.shape[:2]
            n = int(self.upper_split * N)
            ix = torch.arange(B, device=y_pred.device)[:, None]

            error = super().forward(y_pred, y_true)
            if weight is not None:
                error = error * weight
            error, error_index = error.sort(dim=1)

            low, high = error[:, :-n], error[:, -n:]
            # just sample indices once and reuse for all samples in batch
            index = low[0].multinomial(num_samples=n, replacement=False)
            low = low[:, index]

            if weight is None:
                loss = 0.5 * low.mean() + 0.5 * high.mean()
            else:
                weight_low = weight[ix, error_index[:, :-n][:, index]]
                weight_high = weight[ix, error_index[:, -n:]]
                loss = (
                    0.5 * torch.sum(low * weight_low) / weight_low.sum()
                    + 0.5 * torch.sum(high * weight_high) / weight_high.sum()
                )

            return loss

    return SemiHardReduction


# LOSSES


MSELoss = MeanReduction(SquaredError)
L1Loss = MeanReduction(AbsoluteError)
MSNELoss = MeanReduction(SquaredNormError)
MSCosSimLoss = MeanReduction(SquaredCosineSimilarityError)

SemiHardSELoss = SemiHardReduction(SquaredError)
SemiHardL1Loss = SemiHardReduction(AbsoluteError)
SemiHardSNELoss = SemiHardReduction(SquaredNormError)
