import sys
import time
from copy import copy

import joblib
import numpy as np
import optuna
import torch
from mimic3models import metrics
from mimic3models.phenotyping.load_preprocessed import load_cached_data
from mimic3models.stats_utils import write_stats
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_loader(X, y, batch_size, device=DEVICE):
    return DataLoader(
        TensorDataset(
            torch.tensor(X).float().to(device), torch.tensor(y).float().to(device)
        ),
        batch_size=batch_size,
    )


def train_with_params(
    custom_params: dict,
    train_X,
    train_y,
    test_X,
    test_y,
    seed: int = 472368,
    sample_run=False,
    stats_path=None,
):
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

    # weight decay
    dp_params["l2-reg"] = 1e-3

    dp_params.update(custom_params)

    dp_params["batch_size"] = int(dp_params.get("batch_size"))
    dp_params["num_epochs"] = int(dp_params.get("num_epochs"))

    train_loader = get_loader(train_X, train_y, dp_params.get("batch_size"))

    start_model = time.time()
    model = get_model(train_X.shape[1], train_y.shape[1], dp_params)
    model.to(DEVICE)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=dp_params.get("learning_rate"),
        weight_decay=dp_params.get("l2-reg"),
    )

    for epoch in range(dp_params.get("num_epochs")):
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

        # write stats each epoch
        stats_dict = copy(dp_params)
        stats_dict.update(auc_dict)
        stats_dict["num_epochs"] = epoch + 1
        stats_dict["seed"] = seed
        stats_dict["title"] = "mimic3_phenotyping_mlp_width_bn"
        stats_dict["width_hidden_layer"] = sys.argv[1]

        columns = [
            "title",
            "ave_auc_micro",
            "ave_auc_macro",
            "ave_auc_weighted",
            "learning_rate",
            "l2-reg",
            "num_epochs",
            "batch_size",
            "n_layers",
            "num_params",
            "width_hidden_layer",
            "seed",
            "timestamp",
        ]

        write_stats(stats_dict, path=stats_path, columns=columns)

    return auc_dict


def get_model(input, output, dp_params):
    class MLPMultiLabel(torch.nn.Module):
        def __init__(self, input_dim, output_dim, n_layers, out_dims_list):
            super(MLPMultiLabel, self).__init__()

            layers = list()
            layers.append(torch.nn.Linear(input_dim, out_dims_list[0]))
            layers.append(torch.nn.BatchNorm1d(out_dims_list[0]))
            layers.append(torch.nn.ReLU())
            input_dim = out_dims_list[0]

            layers.append(torch.nn.Linear(input_dim, output_dim))
            self.layers = torch.nn.Sequential(*layers)

        def forward(self, x):
            return torch.sigmoid(self.layers(x))

    return MLPMultiLabel(
        input, output, dp_params.get("n_layers"), dp_params.get("out_dims_list")
    )


def return_objective(train_X, train_y, test_X, test_y):
    def objective(trial):
        noise_multiplier = trial.suggest_uniform("noise_multiplier", 0.5, 5)
        clip_bound = trial.suggest_uniform("clip_bound", 0.2, 2)
        learning_rate = trial.suggest_uniform("learning_rate", 0.0009, 0.011)
        l2_regularizer = trial.suggest_uniform("l2_regularizer", 0.0009, 0.0011)
        num_epochs = trial.suggest_int("num_epochs", 1, 200)
        batch_size = trial.suggest_int("batch_size", 32, 2048)
        n_layers = 2
        out_dims_list = list()
        out_dims_list.append(int(sys.argv[1]))
        seeds = [472368, 374148, 521365]
        scores = list()

        for seed in seeds:
            auc_dict = train_with_params(
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
                stats_path="/mimic3experiments/mlp_non_dp_optuna_architecture_width_batchnorm_exp_{}.csv".format(
                    str(sys.argv[1])
                ),
                seed=seed,
                train_X=train_X,
                train_y=train_y,
                test_X=test_X,
                test_y=test_y,
            )

            scores.append(auc_dict.get("ave_auc_micro"))
        return np.mean(scores)

    return objective


if __name__ == "__main__":
    cached_data_path = "//mimic3/data/phenotyping/"
    train_X, train_y, val_X, val_y, test_X, test_y = load_cached_data(cached_data_path)
    study = optuna.create_study(
        study_name="mlp_non_dp_width_batchnorm_{}".format(str(sys.argv[1])),
        direction="maximize",
    )
    objective = return_objective(train_X, train_y, test_X, test_y)
    joblib.dump(
        study,
        str(
            "/mimic3experiments/mlp_non_dp_width_batchnorm_{}.pkl".format(
                str(sys.argv[1])
            )
        ),
    )
    for _ in range(20):
        study.optimize(objective, n_trials=20)
        joblib.dump(
            study,
            str(
                "/mimic3experiments/mlp_non_dp_width_batchnorm_{}.pkl".format(
                    str(sys.argv[1])
                )
            ),
        )
