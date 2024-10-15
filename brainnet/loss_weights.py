import torch

from brainnet import resources_dir

class WeightsMedialWall:
    def __init__(self, weights: torch.Tensor, n_vertices: int | None = None):
        """_summary_

        Parameters
        ----------
        weights : torch.Tensor
            Tensor with two elements where the first and second gives the
            weight of non-medial wall and medial wall vertices, respectively.
        device : str | torch.device, optional
            _description_, by default "cpu"
        """
        medial_wall = torch.load(resources_dir / "medial-wall.pt", map_location=weights.device)
        if n_vertices is not None:
            medial_wall = medial_wall[:n_vertices]
        self.weights = weights[medial_wall.int()]

    def get_weights(self):
        return self.weights


class WeightsFromCurvatureProb:
    def __init__(self, device: str | torch.device = "cpu"):
        n_bins: int = 100
        edges = (-5, 5)
        # Curvature value associated wit each bin
        self.bin_edges = torch.linspace(*edges, n_bins+1, device=device)
        # Negative log probability of curvature value shifted so that the
        # minimum value is 1
        self.neg_log_prob = {
            "white": torch.tensor(
                [
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    9.9035,
                    9.8557,
                    9.7148,
                    9.6459,
                    9.4981,
                    9.3488,
                    9.2343,
                    9.1418,
                    9.0485,
                    8.9031,
                    8.8140,
                    8.7151,
                    8.5630,
                    8.4146,
                    8.3411,
                    8.1593,
                    8.0253,
                    7.8721,
                    7.6944,
                    7.5504,
                    7.3950,
                    7.2435,
                    7.0600,
                    6.8886,
                    6.7141,
                    6.5161,
                    6.3203,
                    6.1122,
                    5.9161,
                    5.7001,
                    5.4802,
                    5.2636,
                    5.0219,
                    4.7791,
                    4.5246,
                    4.2690,
                    3.9932,
                    3.7173,
                    3.4322,
                    3.1289,
                    2.8079,
                    2.4582,
                    2.0616,
                    1.6146,
                    1.1608,
                    1.0000,
                    1.1831,
                    1.6072,
                    2.2114,
                    2.9053,
                    3.6152,
                    4.2864,
                    4.9017,
                    5.4531,
                    5.9435,
                    6.3391,
                    6.7112,
                    7.0213,
                    7.3110,
                    7.5578,
                    7.7866,
                    8.0028,
                    8.2472,
                    8.4211,
                    8.5710,
                    8.7560,
                    8.9456,
                    9.0623,
                    9.1995,
                    9.3488,
                    9.4858,
                    9.5900,
                    9.6954,
                    9.8654,
                    9.9771,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                    10.0000,
                ], device=device
            ),
            "pial": torch.tensor(
                [
                    7.6681,
                    7.6037,
                    7.5850,
                    7.5255,
                    7.4867,
                    7.4426,
                    7.3912,
                    7.3570,
                    7.3155,
                    7.2743,
                    7.2336,
                    7.1781,
                    7.1404,
                    7.0666,
                    7.0333,
                    6.9693,
                    6.9171,
                    6.8744,
                    6.7937,
                    6.7550,
                    6.6798,
                    6.6176,
                    6.5659,
                    6.4852,
                    6.4230,
                    6.3649,
                    6.2698,
                    6.2006,
                    6.1008,
                    6.0194,
                    5.9278,
                    5.8204,
                    5.7129,
                    5.5869,
                    5.4412,
                    5.3098,
                    5.1473,
                    4.9720,
                    4.7807,
                    4.5632,
                    4.3161,
                    4.0573,
                    3.7515,
                    3.4160,
                    3.0363,
                    2.6033,
                    2.1167,
                    1.6060,
                    1.1896,
                    1.0000,
                    1.1840,
                    1.5536,
                    1.9103,
                    2.2423,
                    2.5451,
                    2.8271,
                    3.0823,
                    3.3201,
                    3.5371,
                    3.7322,
                    3.9179,
                    4.0872,
                    4.2468,
                    4.3927,
                    4.5335,
                    4.6669,
                    4.7830,
                    4.8927,
                    5.0069,
                    5.1078,
                    5.1994,
                    5.2936,
                    5.3870,
                    5.4604,
                    5.5460,
                    5.6158,
                    5.6818,
                    5.7614,
                    5.8210,
                    5.8775,
                    5.9458,
                    6.0079,
                    6.0510,
                    6.1077,
                    6.1600,
                    6.2128,
                    6.2657,
                    6.3130,
                    6.3538,
                    6.4038,
                    6.4422,
                    6.4817,
                    6.5418,
                    6.5686,
                    6.6234,
                    6.6598,
                    6.6936,
                    6.7362,
                    6.7532,
                    6.8049,
                ], device=device
            ),
        }

    def bucketize(self, t):
        return torch.bucketize(t, self.bin_edges[1:-1])

    def get_weights(self, t: torch.Tensor, surface: str, normalize: bool = False):
        w = self.neg_log_prob[surface][self.bucketize(t)]
        if normalize:
            w /= w.sum(1)
        return w
