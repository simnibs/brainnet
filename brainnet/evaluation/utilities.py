from pathlib import Path
from typing import Callable

import pandas as pd
import torch

# from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced  # , sync_all_reduce
from ignite.engine import Events

from brainnet.utilities import flatten_dict
from brainnet import event_handlers


def list_of_dicts_to_dataframe(data, index=None):
    # list[dict] to dict where values are lists of the values of the
    # initial list[dict]
    out = {}
    for d in data:
        for k, v in flatten_dict(d).items():
            if k in out:
                out[k].append(v)
            else:
                out[k] = [v]
    # DataFrame: index = iterations; columns = multiindex of keys
    return pd.DataFrame(out, index)

def add_metric_writer(
    engine, out_dir: Path, metric: str = "loss"
):
    if not out_dir.exists():
        out_dir.mkdir()
    engine.add_event_handler(
        Events.EPOCH_COMPLETED, event_handlers.write_metric, metric, out_dir
    )

class MetricAggregator(Metric):

    # required_output_keys: tuple[str,str] = ("raw", "weighted") #("y_pred", "y", "criterion_kwargs")
    # _state_dict_all_req_keys: tuple[str,str] = #("_sum", "_num_examples")

    def __init__(
        self,
        # loss_fn: Callable,
        output_transform: Callable = lambda x: x,
        batch_size: Callable = len,
        device: str | torch.device = torch.device("cpu"),
    ):
        """This "metric" is based on ignite.metrics.Loss but works with a dict of
        (averaged) losses rather than computing a single loss from y_pred and
        y. All entries (losses) are averaged separately.
        """
        super().__init__(output_transform, device=device)
        self._batch_size = batch_size

    @reinit__is_reduced
    def reset(self) -> None:
        self._aggregated = []

    @reinit__is_reduced
    def update(self, output: tuple) -> None:
        if len(output) == 4:
            loss, _, _, _ = output  # out signature: loss, x, y_pred, y_true
        else:
            ValueError(
                f"Wrong output signature from engine for CriterionAggregator. Expected (loss, x, y_pred, y_true), got output of length {len(output)}."
            )

        # batch_size = self._batch_size(x)
        # if batch_size > 1:
        #     brainnet.utilities.recursive_dict_multiply(loss, batch_size)
        # brainnet.utilities.add_dict(self._sum, loss)
        # brainnet.utilities.increment_dict_count(self._num_examples, loss, batch_size)

        self._aggregated.append(loss)

    # @sync_all_reduce("_sum", "_num_examples")
    def compute(self) -> pd.DataFrame:
        return list_of_dicts_to_dataframe(self._aggregated)
