import json
import os
import time
import warnings
from copy import copy

import sys
import numpy as np
import opacus
import pandas as pd
import torch
from mimic3models import metrics
from mimic3models.phenotyping.load_preprocessed import load_cached_data
from mimic3models.phenotyping.logistic.main import load_process_data
from mimic3models.stats_utils import write_stats
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLPMultiLabel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPMultiLabel, self).__init__()

        hidden_dim = (input_dim + output_dim) // 2
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(output))

    @staticmethod
    def get_num_params(input_dim, output_dim):
        hidden_dim = (input_dim + output_dim) // 2
        return (input_dim + 1) * hidden_dim + (hidden_dim + 1) * output_dim


def get_loader(X, y, batch_size, device=DEVICE):
    return DataLoader(
        TensorDataset(
            torch.tensor(X).float().to(device), torch.tensor(y).float().to(device)
        ),
        batch_size=batch_size,
    )


def train_with_params(
    custom_params: dict = None,
    seed: int = 472368,
    sample_run=False,
    cached_data=True,
    stats_path=None,
):
    args = dict()
    args["data"] = "/mimic3/data/phenotyping/"
    args["period"] = "all"
    args["features"] = "all"

    torch.manual_seed(seed)

    if custom_params is None:
        custom_params = dict()

    dp_params = dict()
    # Learning rate for training
    dp_params["learning_rate"] = 1e-3

    # Number of epochs
    dp_params["num_epochs"] = 20

    if sample_run:
        dp_params["num_epochs"] = 0

    # batch size
    dp_params["batch_size"] = 64

    # Ratio of the standard deviation to the clipping norm
    dp_params["noise_multiplier"] = 2

    # Clipping norm
    dp_params["clip_bound"] = 1

    # Batch size as a fraction of full data size
    dp_params["sample_rate"] = "default"

    # delta
    dp_params["delta"] = 1e-5

    # weight decay
    dp_params["l2-reg"] = 1e-3

    dp_params.update(custom_params)

    dp_params["batch_size"] = int(dp_params.get("batch_size"))
    dp_params["num_epochs"] = int(dp_params.get("num_epochs"))

    if cached_data:
        train_X, train_y, val_X, val_y, test_X, test_y = load_cached_data(
            args.get("data")
        )
    else:
        train_X, train_y, val_X, val_y, test_X, test_y = load_process_data(args)

    train_loader = get_loader(train_X, train_y, dp_params.get("batch_size"))

    # print("Loading and parsing data took", time.time() - start_data, "seconds.")

    # print("Starting model fitting")
    start_model = time.time()
    model = MLPMultiLabel(train_X.shape[1], train_y.shape[1])
    model.to(DEVICE)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=dp_params.get("learning_rate"),
        weight_decay=dp_params.get("l2-reg"),
    )

    # make private
    warnings.filterwarnings("ignore", message=r".*Secure RNG turned off.*")
    privacy_engine = opacus.PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=dp_params.get("noise_multiplier"),
        max_grad_norm=dp_params.get("clip_bound"),
    )

    gradient_norms = list()

    step = 0
    for epoch in range(dp_params.get("num_epochs")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.train()
            train_epoch_loss = 0

            for data, target in train_loader:
                optimizer.zero_grad()
                y_pred = model(data)

                loss = criterion(y_pred, target)
                train_epoch_loss += loss.item()

                loss.backward()
                for p in model.parameters():
                    for grad_single_sample in p.grad_sample:
                        gradient_norms.append(
                            [
                                grad_single_sample.detach().data.norm(2).cpu().item(),
                                step,
                                epoch,
                            ]
                        )
                optimizer.step()
                step += 1

    # compute gradient clipping value
    quantile_clipping = stats.percentileofscore(
        gradient_norms, dp_params.get("clip_bound")
    )

    model.eval()

    df = pd.DataFrame(
        [[x[0], x[1], x[2], dp_params.get("clip_bound")] for x in gradient_norms]
    )
    df.columns = ["gradient_norm", "clipping_bound", "optim_step", "epoch"]
    df.to_csv(
        "/mimic3experiments/mlp_dp_gradients_new_exp_cb_{}.csv".format(
            sys.argv[1]
        ),
        mode="a",
        header=not os.path.exists(
            "/mimic3experiments/mlp_dp_gradients_new_exp_cb_{}.csv".format(
                sys.argv[1]
            )
        ),
        index=False,
    )
    with torch.no_grad():
        test_activations = model(torch.tensor(test_X).to(DEVICE)).cpu()

        auc_dict = metrics.print_metrics_multilabel(test_y, test_activations, verbose=0)

    epsilon = privacy_engine.get_epsilon(delta=dp_params.get("delta"))

    columns = [
        "title",
        "ave_auc_micro",
        "ave_auc_macro",
        "ave_auc_weighted",
        "epsilon",
        "learning_rate",
        "l2-reg",
        "num_epochs",
        "batch_size",
        "noise_multiplier",
        "clip_bound",
        "quantile_clipping",
        "delta",
        "seed",
        "timestamp",
        "elapsed_training_time",
    ]

    # write stats each epoch
    stats_dict = copy(dp_params)
    stats_dict.update(auc_dict)
    stats_dict["quantile_clipping"] = quantile_clipping
    stats_dict["num_epochs"] = epoch + 1
    stats_dict["seed"] = seed
    stats_dict["title"] = "mimic3_phenotyping_mlp_dp"
    stats_dict["epsilon"] = epsilon
    write_stats(stats_dict, path=stats_path, columns=columns)

    return auc_dict, epsilon, model


if __name__ == "__main__":
    csv_path = "/mimic3/mimic3models/phenotyping/mlp/best_for_time_measurements/cb.csv"

    df = pd.read_csv(csv_path)

    list_of_dicts = list()

    for i, r in df.iterrows():
        current_dict = dict()
        for c in df.columns:
            current_dict[c] = r[c]
        list_of_dicts.append(current_dict)

    all_runs = np.ravel([list_of_dicts] * 1)

    step = int(sys.argv[1])
    for i, c in enumerate(all_runs[step : step + 20]):
        print(
            "Looking at config {}/{} of interval {}-{}".format(
                i, len(all_runs[step : step + 20]), step, step + 20
            ),
            flush=True,
        )
        auc_dict, epsilon, model = train_with_params(
            c,
            stats_path="/mimic3experiments/mlp_dp_cb_quantile_test.csv",
            seed=int(c["seed"]),
        )
