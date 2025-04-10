import torch

# ERROR FUNCTIONS

class SquaredError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return (a - b) ** 2
        # return torch.mean((a - b) ** 2, dim=-1)

class NormalizedSquaredError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # return (a - b) ** 2 / torch.clamp(b**2, min=1.0/b.abs().amax()**2)
        # return torch.mean((a - b) ** 2 / torch.clamp(b**2, self.tol), dim=-1)
        d = a**2 + b**2
        d = d.clamp(min=1.0/d.amax())
        return (a - b)**2 / d

class AbsoluteError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return torch.abs(a - b)#.mean(dim=-1)

class NormalizedAbsoluteError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # return torch.abs(a - b) / b.clamp(min=1.0/b.abs().amax())
        d = a.abs() + b.abs()
        d = d.clamp(min=1.0/d.amax())
        return torch.abs(a - b) / d


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


def wrap_mean_reduction(cls):
    class MeanReduction(cls):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        def forward(self, y_pred, y_true, weight=None):
            error = super().forward(y_pred, y_true)
            if weight is None:
                return error.mean()
            else:
                return torch.sum(weight * error) / weight.sum()

    return MeanReduction


def wrap_semi_hard_reduction(cls):
    class SemiHardReduction(cls):
        def __init__(self, hard_fraction: float, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            assert 0 <= hard_fraction <= 0.5
            self.hard_fraction = hard_fraction

        def forward(self, y_pred, y_true, weight=None):
            n = int(self.hard_fraction * y_pred.shape[1])

            error = super().forward(y_pred, y_true)
            error = error if weight is None else error * weight
            error, error_index = error.sort(dim=-1)

            low, high = error[:, :-n], error[:, -n:]
            # just sample indices once and reuse for all samples in batch
            index = low[0].multinomial(num_samples=n, replacement=False)
            low = low[:, index]

            if weight is None:
                low_reduc = low.mean()
                high_reduc = high.mean()
            else:
                weight_low = weight.gather(1, error_index[:, :-n][:, index])
                weight_high = weight.gather(1, error_index[:, -n:])
                low_reduc = torch.sum(low * weight_low) / weight_low.sum()
                high_reduc = torch.sum(high * weight_high) / weight_high.sum()

            return 0.5 * low_reduc + 0.5 * high_reduc

    return SemiHardReduction


# LOSSES


MSELoss = wrap_mean_reduction(SquaredError)
NSELoss = wrap_mean_reduction(NormalizedSquaredError)
L1Loss = wrap_mean_reduction(AbsoluteError)
NL1Loss = wrap_mean_reduction(NormalizedAbsoluteError)
MSNELoss = wrap_mean_reduction(SquaredNormError)
MSCosSimLoss = wrap_mean_reduction(SquaredCosineSimilarityError)

SemiHardSELoss = wrap_semi_hard_reduction(SquaredError)
SemiHardL1Loss = wrap_semi_hard_reduction(AbsoluteError)
SemiHardSNELoss = wrap_semi_hard_reduction(SquaredNormError)
