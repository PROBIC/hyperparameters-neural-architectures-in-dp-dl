import os
import sys
import time
from tracemalloc import stop
import warnings
from copy import copy

import joblib
import numpy as np
import opacus
import optuna
import torch
from mimic3models import metrics
from mimic3models.phenotyping.load_preprocessed import load_cached_data
from mimic3models.phenotyping.logistic.main import load_process_data
from mimic3models.stats_utils import write_stats
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CACHED_DATA = "/mimic3/data/phenotyping/"

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
    stop_epsilon=1,
    cached_data_path = None,
):
    args = dict()
    if cached_data_path is None:
        args["data"] = os.path.join(os.path.dirname(__file__), "../../../data/phenotyping/")
    else:
        args["data"] = cached_data_path
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

    start_all = time.time()
    start_data = time.time()

    if cached_data:
        train_X, train_y, val_X, val_y, test_X, test_y = load_cached_data(
            args.get("data")
        )
    else:
        train_X, train_y, val_X, val_y, test_X, test_y = load_process_data(args)

    train_loader = get_loader(train_X, train_y, dp_params.get("batch_size"))
    val_loader = get_loader(val_X, val_y, dp_params.get("batch_size"))
    test_loader = get_loader(test_X, test_y, dp_params.get("batch_size"))

    start_model = time.time()
    model = get_model(train_X.shape[1], train_y.shape[1], dp_params)
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

                # print("\nTest data:")
                auc_dict = metrics.print_metrics_multilabel(
                    test_y, test_activations, verbose=0
                )

        epsilon = privacy_engine.get_epsilon(delta=dp_params.get("delta"))

        if epsilon > float(stop_epsilon):
            break

        # write stats each epoch
        stats_dict = copy(dp_params)
        stats_dict.update(auc_dict)
        stats_dict["num_epochs"] = epoch + 1
        stats_dict["seed"] = seed
        stats_dict["title"] = "mimic3_phenotyping_mlp_dp_layers"
        stats_dict["epsilon"] = epsilon
        stats_dict["num_params"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
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
            "sample_rate",
            "n_layers",
            "num_params",
            "delta",
            "seed",
            "timestamp",
        ]

        write_stats(stats_dict, path=stats_path, columns=columns)

    return auc_dict, epsilon


def get_model(input, output, dp_params):
    class MLPMultiLabel(torch.nn.Module):
        def __init__(self, input_dim, output_dim, n_layers, out_dims_list):
            super(MLPMultiLabel, self).__init__()

            layers = list()
            for i in range(n_layers - 1):
                layers.append(torch.nn.Linear(input_dim, out_dims_list[i]))
                layers.append(torch.nn.ReLU())
                input_dim = out_dims_list[i]

            layers.append(torch.nn.Linear(input_dim, output_dim))
            self.layers = torch.nn.Sequential(*layers)

        def forward(self, x):
            return torch.sigmoid(self.layers(x))

    return MLPMultiLabel(
        input, output, dp_params.get("n_layers"), dp_params.get("out_dims_list")
    )


def return_objective(paramvalue):
    def objective(trial):
        noise_multiplier = trial.suggest_uniform("noise_multiplier", 0.5, 5)
        clip_bound = trial.suggest_uniform("clip_bound", 0.2, 2)
        learning_rate = trial.suggest_uniform("learning_rate", 0.0009, 0.011)
        l2_regularizer = trial.suggest_uniform("l2_regularizer", 0.0009, 0.0011)
        num_epochs = trial.suggest_int("num_epochs", 1, 200)
        batch_size = trial.suggest_int("batch_size", 32, 1024)
        n_layers = int(sys.argv[1])
        out_dims_list = list()
        for i in range(n_layers - 1):
            out_dims_list.append(trial.suggest_int("n_units_l{}".format(i), 4, 300))
        seeds = [472368, 374148, 521365]
        scores = list()

        for seed in seeds:
            auc_dict, epsilon = train_with_params(
                {
                    "noise_multiplier": noise_multiplier,
                    "clip_bound": clip_bound,
                    "learning_rate": learning_rate,
                    "l2-reg": l2_regularizer,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "n_layers": n_layers,
                    "out_dims_list": out_dims_list,
                },
                stats_path="/mimic3experiments/mlp_optuna_architecture_layer_exp_more_epsilon_{}_{}.csv".format(
                    str(sys.argv[1]), sys.argv[2]
                ),
                seed=seed,
                stop_epsilon=sys.argv[2],
                cached_data_path=CACHED_DATA,
            )

            scores.append(auc_dict.get("ave_auc_micro"))
        return np.mean(scores)

    return objective


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="mlp_layers_more_epsilon_{}_{}.pkl".format(
            str(sys.argv[1]), sys.argv[2]
        ),
        direction="maximize",
    )
    objective = return_objective(None)
    joblib.dump(
        study,
        str(
            "/mimic3experiments/mlp_layers_{}_{}.pkl".format(
                str(sys.argv[1]), sys.argv[2]
            )
        ),
    )
    for _ in range(25):
        study.optimize(objective, n_trials=20)
        joblib.dump(
            study,
            str(
                "/mimic3experiments/mlp_layers_{}_{}.pkl".format(
                    str(sys.argv[1]), sys.argv[2]
                )
            ),
        )
