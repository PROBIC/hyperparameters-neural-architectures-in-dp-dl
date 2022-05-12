import os
import time
import warnings
from copy import copy

import numpy as np
import opacus
import torch
from joblib import dump
from mimic3models import metrics
from mimic3models.activation_functions.tempered_sigmoid import Temperedsigmoid
from mimic3models.phenotyping.load_preprocessed import load_cached_data
from mimic3models.phenotyping.logistic.main import load_process_data
from mimic3models.stats_utils import write_stats
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLPMultiLabel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation_function="tempered",
        scale=2.0,
        inverse_temperature=2.0,
        offset=1.0,
    ):
        super(MLPMultiLabel, self).__init__()

        hidden_dim = (input_dim + output_dim) // 2

        # tempered sigmoid params
        self.scale = scale
        self.inverse_temp = inverse_temperature
        self.offset = offset

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        if activation_function == "tempered":
            self.activation = Temperedsigmoid(
                scale=self.scale,
                inverse_temperature=self.inverse_temp,
                offset=self.offset,
            )
        elif activation_function == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation_function == "relu":
            self.activation = torch.nn.ReLU()

    def forward(self, x):
        output = self.activation(self.linear1(x))
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
    stop_epsilon=10,
):
    args = dict()
    args["data"] = os.path.join(os.path.dirname(__file__), "../../../data/phenotyping/")
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

    if cached_data:
        train_X, train_y, _, _, test_X, test_y = load_cached_data(args.get("data"))

    train_loader = get_loader(train_X, train_y, dp_params.get("batch_size"))

    model = MLPMultiLabel(
        train_X.shape[1],
        train_y.shape[1],
        activation_function=dp_params.get("activation"),
        scale=dp_params.get("scale"),
        inverse_temperature=dp_params.get("inverse_temperature"),
        offset=dp_params.get("offset"),
    )
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
                optimizer.step()

            model.eval()
            with torch.no_grad():
                test_activations = model(torch.tensor(test_X).to(DEVICE)).cpu()

                auc_dict = metrics.print_metrics_multilabel(
                    test_y, test_activations, verbose=0
                )

        epsilon = privacy_engine.get_epsilon(delta=dp_params.get("delta"))

        if epsilon > stop_epsilon:
            break

        columns = [
            "title",
            "activation",
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
            "sample_rate",
            "inverse_temperature",
            "scale",
            "offset",
            "delta",
            "seed",
            "timestamp",
        ]

        # write stats each epoch
        stats_dict = copy(dp_params)
        stats_dict.update(auc_dict)
        stats_dict["num_epochs"] = epoch + 1
        stats_dict["seed"] = seed
        stats_dict["title"] = "mimic3_phenotyping_mlp_dp"
        stats_dict["epsilon"] = epsilon
        write_stats(stats_dict, path=stats_path, columns=columns)

    return auc_dict, epsilon
