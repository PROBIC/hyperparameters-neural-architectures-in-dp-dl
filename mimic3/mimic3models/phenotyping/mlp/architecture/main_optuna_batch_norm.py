import os
import sys
import time
import warnings
from copy import copy

import joblib
import numpy as np
import torch
import optuna
from mimic3models import metrics
from mimic3models.phenotyping.load_preprocessed import load_cached_data
from mimic3models.phenotyping.logistic.main import load_process_data
from mimic3models.stats_utils import write_stats
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLPMultiLabel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPMultiLabel, self).__init__()

        hidden_dim = (input_dim + output_dim) // 2  # 369
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.batchnorm = torch.nn.BatchNorm1d(hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.linear1(x)
        output = self.relu(self.batchnorm(output))
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
    data_shuffle_seed=None,
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

    # batch size>
    dp_params["batch_size"] = 64

    # Batch size as a fraction of full data size
    dp_params["sample_rate>"] = "default"

    # weight decay
    dp_params["l2-reg"] = 1e-3

    dp_params.update(custom_params)

    start_all = time.time()
    start_data = time.time()

    if cached_data:
        train_X, train_y, val_X, val_y, test_X, test_y = load_cached_data(
            args.get("data")
        )
    else:
        train_X, train_y, val_X, val_y, test_X, test_y = load_process_data(args)

    if data_shuffle_seed is not None:
        combined_X = np.concatenate((train_X, test_X))
        combined_y = np.concatenate((train_y, test_y))
        proportion_train = len(train_X) / len(combined_X)
        proportion_test = len(test_X) / len(combined_X)
        train_X, test_X, train_y, test_y = train_test_split(
            combined_X,
            combined_y,
            train_size=proportion_train,
            test_size=proportion_test,
            random_state=data_shuffle_seed,
        )

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
                # save_gradients_to_csv(
                #    model,
                #    epoch + 1,
                #    "/mimic3experiments/mlp_not_dp_RELU_gradients.csv",
                # )
                optimizer.step()

            model.eval()
            with torch.no_grad():
                test_activations = model(torch.tensor(test_X).to(DEVICE)).cpu()

                # print("\nTest data:")
                auc_dict = metrics.print_metrics_multilabel(
                    test_y, test_activations, verbose=0
                )

        # write stats each epoch
        stats_dict = copy(dp_params)
        stats_dict.update(auc_dict)
        stats_dict["num_epochs"] = epoch + 1
        stats_dict["seed"] = seed
        stats_dict["title"] = "mimic3_phenotyping_mlp_bn"

        columns = [
            "title",
            "ave_auc_micro",
            "ave_auc_macro",
            "ave_auc_weighted",
            "learning_rate",
            "l2-reg",
            "num_epochs",
            "batch_size",
            "seed",
            "timestamp",
        ]

        write_stats(stats_dict, path=stats_path, columns=columns)

    return auc_dict, model


def return_objective(paramvalue):
    def objective(trial):
        learning_rate = trial.suggest_uniform("learning_rate", 0.00001, 0.01)
        l2_regulizer = trial.suggest_uniform("l2_regulizer", 0.00001, 0.01)
        num_epochs = trial.suggest_int("num_epochs", 1, 400)
        batch_size = trial.suggest_int("batch_size", 32, 2048)

        seeds = [472368, 374148, 521365]
        scores = list()

        for seed in seeds:
            auc_dict, _ = train_with_params(
                {
                    "learning_rate": learning_rate,
                    "l2-reg": l2_regulizer,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                },
                stats_path=str(
                    "/mimic3experiments/mlp_batchrnorm_before_optuna.csv"
                ),
                seed=seed,
            )

            scores.append(auc_dict.get("ave_auc_micro"))
        return np.mean(scores)

    return objective


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=str("mlp_layernorm_optuna"), direction="maximize"
    )
    objective = return_objective(None)
    joblib.dump(
        study, "/mimic3experiments/mlp_batchrnorm_before_optuna.pkl",
    )
    for _ in range(50):
        study.optimize(objective, n_trials=20)
        joblib.dump(
            study, "/mimic3experiments/mlp_batchrnorm_before_optuna.pkl",
        )
