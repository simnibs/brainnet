import torch


# ERROR FUNCTIONS


class SquaredError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return (a - b) ** 2


class SquaredNormError(torch.nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.sum((a - b) ** 2, self.dim)


class AbsoluteError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return torch.abs(a - b)


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
            n = int(self.upper_split * y_pred.shape[1])

            error = super().forward(y_pred, y_true)
            if weight is not None:
                error = error * weight
            error, error_index = error.sort()

            low, high = error[:-n], error[-n:]
            index = low.multinomial(num_samples=n, replacement=False)
            low = low[index]

            if weight is None:
                loss = 0.5 * low.mean() + 0.5 * high.mean()
            else:
                weight_low = weight[error_index[:-n][index]]
                weight_high = weight[error_index[-n:]]
                loss = (
                    0.5 * low * weight_low / weight_low.sum()
                    + 0.5 * high * weight_high / weight_high.sum()
                )

            return loss

    return SemiHardReduction


# LOSSES


MSELoss = MeanReduction(SquaredError)
L1Loss = MeanReduction(AbsoluteError)
MSNELoss = MeanReduction(SquaredNormError)
MSCosSimLoss = MeanReduction(SquaredCosineSimilarityError)

SemihardSELoss = SemiHardReduction(SquaredError)
SemihardL1Loss = SemiHardReduction(AbsoluteError)
SemihardSNELoss = SemiHardReduction(SquaredNormError)
