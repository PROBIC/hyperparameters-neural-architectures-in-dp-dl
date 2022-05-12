import sys

import joblib
import numpy as np
import optuna
from mimic3models.phenotyping.mlp.dpmain_tempered import train_with_params


def return_objective(paramvalue):
    def objective(trial):
        noise_multiplier = trial.suggest_uniform("noise_multiplier", 0.5, 5)
        clip_bound = trial.suggest_uniform("clip_bound", 0.2, 2)
        learning_rate = trial.suggest_uniform("learning_rate", 0.00001, 0.009)
        l2_regulizer = trial.suggest_uniform("l2_regulizer", 0.0009, 0.0011)
        num_epochs = trial.suggest_int("num_epochs", 1, 200)
        batch_size = trial.suggest_int("batch_size", 32, 2048)

        # tempered sigmoid values
        offset = trial.suggest_uniform("offset", -1, 4)
        inverse_temperature = trial.suggest_uniform("inverse_temperature", 0.01, 8)
        scale = trial.suggest_uniform("scale", -1, 5)

        seeds = [472368, 374148, 521365]
        scores = list()

        for seed in seeds:
            if len(sys.argv) > 1:
                stop_epsilon = float(sys.argv[1])
            auc_dict, epsilon = train_with_params(
                {
                    "noise_multiplier": noise_multiplier,
                    "clip_bound": clip_bound,
                    "learning_rate": learning_rate,
                    "l2-reg": l2_regulizer,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "offset": offset,
                    "inverse_temperature": inverse_temperature,
                    "scale": scale,
                    "activation": sys.argv[2],
                },
                stats_path=str(
                    "/mimic3experiments/mlp_tempered_optuna_all_{}_{}.csv".format(
                        sys.argv[1], sys.argv[2]
                    )
                ),
                seed=seed,
                stop_epsilon=stop_epsilon,
            )

            scores.append(auc_dict.get("ave_auc_micro"))
        return np.mean(scores)

    return objective


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=str(
            "mlp_tempered_optuna_all_{}_{}.pkl".format(sys.argv[1], sys.argv[2])
        ),
        direction="maximize",
    )
    objective = return_objective(None)
    joblib.dump(
        study,
        str(
            "/mimic3experiments/mlp_tempered_optuna_all_{}_{}.pkl".format(
                sys.argv[1], sys.argv[2]
            )
        ),
    )
    for _ in range(40):
        study.optimize(objective, n_trials=20)
        joblib.dump(
            study,
            str(
                "/mimic3experiments/mlp_tempered_optuna_all_{}_{}.pkl".format(
                    sys.argv[1], sys.argv[2]
                )
            ),
        )
