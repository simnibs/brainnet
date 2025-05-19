import math

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
        d = d.clamp(min=1.0 / d.amax())
        return (a - b) ** 2 / d


class AbsoluteError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return torch.abs(a - b)  # .mean(dim=-1)


class NormalizedAbsoluteError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # return torch.abs(a - b) / b.clamp(min=1.0/b.abs().amax())
        d = a.abs() + b.abs()
        d = d.clamp(min=1.0 / d.amax())
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


class NegLogLikDiagonalMultivariateNormal(torch.nn.Module):
    def __init__(self):
        """Simplified version of torch.distributions.MultivariateNormal that
        assumes a diagonal covariance matrix.
        """
        super().__init__()

    def log_prob(self, loc, scale, value):
        # compute the Mahalanobis distance (x-mu).T @ SIGMA**-1 @ (x-mu) when
        # SIGMA is diagonal
        diff = value - loc
        M = diff.pow(2).div(scale.pow(2)).sum(-1)

        half_log_det = scale.log().sum(-1)
        d = loc.shape[-1]
        return -0.5 * (d * math.log(2 * math.pi) + M) - half_log_det
        # return -0.5 * M - half_log_det
        # return -M - 2.0 * half_log_det

    def forward(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        query_points: torch.Tensor,
    ):
        """Given loc (mean) and scale (standard deviation) of a multivariate
        normal distribution with diagonal covariance structure, estimate the
        negative log probability of the query points.

        Parameters
        ----------
        loc : torch.Tensor
            (n_batch, n_vertices, d)
        scale : torch.Tensor
            The *standard deviation* in each dimension, i.e., the square root
            of the diagonal entries in the covariance matrix.
            (n_batch, n_vertices, d)
        query_points : torch.Tensor
            (n_batch, n_vertices, d)

        Returns
        -------
        neg_log_prob
            (n_batch, n_vertices)
        """
        return self.log_prob(loc, scale, query_points).neg()


class NegLogLikStandardMultivariateNormal(NegLogLikDiagonalMultivariateNormal):
    def log_prob(self, loc, scale, value):
        # force scale = 1.0

        # compute the Mahalanobis distance (x-mu).T @ SIGMA**-1 @ (x-mu) when
        # SIGMA is diagonal
        diff = value - loc
        M = diff.pow(2).div(1.0).sum(-1)

        # half_log_det = 0.0
        # d = loc.shape[-1]

        # return -0.5 * (d * math.log(2 * math.pi) + M)
        # return -0.5 * M
        return -M


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


# Make a mean reduction of the negative log probability loss mimicking the
# input pattern of `wrap_mean_reduction` but where the third arg is passed to
# the super method rather than being used to weigh the error post hoc!
class NegLogLikLoss(NegLogLikDiagonalMultivariateNormal):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, y_pred_scale):
        error = super().forward(y_pred, y_pred_scale, y_true)
        return error.mean()


class NegLogLikStandardLoss(NegLogLikStandardMultivariateNormal):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, y_pred_scale):
        error = super().forward(y_pred, y_pred_scale, y_true)
        return error.mean()
