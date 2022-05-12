import sys
import time
from copy import copy

path = "/python_packages"
sys.path.append(path)


import joblib
from sklearn.metrics import auc
import numpy as np
import optuna
import tensorflow as tf
from mimic3models import metrics
from mimic3models.phenotyping.load_preprocessed import load_cached_data
from mimic3models.stats_utils import write_stats
from tensorflow_privacy.privacy.analysis.rdp_accountant import (
    compute_rdp,
    get_privacy_spent,
)
from tensorflow_privacy.privacy.dp_query.quantile_adaptive_clip_sum_query import (
    QuantileAdaptiveClipSumQuery,
)
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamOptimizer


def compute_epsilon(steps, dp_params, dataset_size):
    """Computes epsilon value for given hyperparameters."""
    if dp_params.get("noise_multiplier") == 0.0:
        return float("inf")
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = dp_params.get("batch_size") / dataset_size
    rdp = compute_rdp(
        q=sampling_probability,
        noise_multiplier=dp_params.get("noise_multiplier"),
        steps=steps,
        orders=orders,
    )
    # Delta is set to 1e-5
    return get_privacy_spent(orders, rdp, target_delta=dp_params.get("delta"))[0]


def get_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))


def train_with_params(
    dp_params: dict = None,
    seed: int = 472368,
    stats_path=None,
    cached_data_path=None,
):
    tf.random.set_seed(seed)
    # Load training and test data.
    train_X, train_y, val_X, val_y, test_X, test_y = load_cached_data(cached_data_path)

    train_dataset = get_dataset(train_X, train_y)
    # Define the model using tf.keras.layers
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(train_X.shape[1])),
            tf.keras.layers.Dense(81, activation="relu"),
            tf.keras.layers.Dense(train_y.shape[1], activation="sigmoid"),
        ]
    )

    dp_adaptive_query = QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=dp_params.get("initial_clipbound"),
        noise_multiplier=dp_params.get("noise_multiplier"),
        target_unclipped_quantile=dp_params.get("target_unclipped_quantile"),
        learning_rate=0.2,
        clipped_count_stddev=1,
        expected_num_records=20,
        geometric_update=True,
    )
    optimizer = DPAdamOptimizer(
        learning_rate=dp_params.get("learning_rate"), dp_sum_query=dp_adaptive_query
    )

    train_dataset = train_dataset.shuffle(len(test_X), reshuffle_each_iteration=True)

    @tf.function
    def training_step(data, target):
        with tf.GradientTape(persistent=True) as gradient_tape:
            # This dummy call is needed to obtain the var list.
            logits = model(data, training=True)
            var_list = model.trainable_variables

            # In Eager mode, the optimizer takes a function that returns the loss.
            @tf.function
            def loss_fn():
                criterion = tf.keras.losses.BinaryCrossentropy(
                    from_logits=False, reduction="none"
                )
                loss = criterion(target, logits)
                return loss

            grads_and_vars = optimizer.compute_gradients(
                loss_fn, var_list, gradient_tape=gradient_tape
            )

        optimizer.apply_gradients(grads_and_vars)

    # Training loop.
    for epoch in range(dp_params.get("num_epochs")):
        # Train the model for one epoch.
        for data, target in train_dataset.batch(
            dp_params.get("batch_size"), drop_remainder=True
        ):
            auc_dict, epsilon = training_step(data, target)

        # Evaluate the model and print results
        test_activations = model(test_X, training=False)
        auc_dict = metrics.print_metrics_multilabel(test_y, test_activations, verbose=0)

        # Compute the privacy budget expended.
        epsilon = compute_epsilon(
            (epoch + 1) * len(train_X) // dp_params.get("batch_size"),
            dp_params,
            len(train_X),
        )
        
        if epsilon > float(sys.argv[2]):
            break

        # write stats each epoch
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
            "initial_clipbound",
            "target_unclipped_quantile",
            "delta",
            "seed",
            "timestamp",
        ]
        stats_dict = copy(dp_params)
        stats_dict.update(auc_dict)
        stats_dict["num_epochs"] = epoch + 1
        stats_dict["seed"] = seed
        stats_dict["title"] = "mimic3_phenotyping_mlp_dp_tf_adaptive"
        stats_dict["epsilon"] = epsilon
        write_stats(stats_dict, path=stats_path, columns=columns)

    return auc_dict


if __name__ == "__main__":

    def objective(trial):
        noise_multiplier = trial.suggest_uniform("noise_multiplier", 0.5, 5)
        learning_rate = trial.suggest_uniform("learning_rate", 0.0009, 0.011)
        l2_regularizer = trial.suggest_uniform("l2_regularizer", 0.0009, 0.0011)
        num_epochs = trial.suggest_int("num_epochs", 1, 300)
        batch_size = trial.suggest_int("batch_size", 32, 3072)
        initial_clipbound = trial.suggest_uniform("initial_clipbound", 0.2, 3)
        target_unclipped_quantile = float(sys.argv[1])
        seeds = [472368, 374148, 521365]
        scores = list()
        for seed in seeds:
            auc_dict = train_with_params(
                {
                    "noise_multiplier": noise_multiplier,
                    "learning_rate": learning_rate,
                    "l2-reg": l2_regularizer,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "initial_clipbound": initial_clipbound,
                    "target_unclipped_quantile": target_unclipped_quantile,
                    "delta": 1e-5,
                },
                stats_path="/mimic3experiments/mlp_tf_adaptive_clipping_{}_{}.csv".format(
                    str(sys.argv[1]), sys.argv[2]
                ),
                seed=seed,
                cached_data_path="/mimic3/data/phenotyping/",
            )

            scores.append(auc_dict.get("ave_auc_micro"))
        return np.mean(scores)

    study = optuna.create_study(
        study_name="mlp_tf_adaptive_clipping_{}_{}".format(
            str(sys.argv[1]), str(sys.argv[2])
        ),
        direction="maximize",
    )
    joblib.dump(
        study,
        str(
            "/mimic3experiments/mlp_tf_adaptive_clipping_{}_{}.pkl".format(
                str(sys.argv[1]), sys.argv[2]
            )
        ),
    )
    for _ in range(20):
        study.optimize(objective, n_trials=20)
        joblib.dump(
            study,
            str(
                "/mimic3experiments/mlp_tf_adaptive_clipping_{}_{}.pkl".format(
                    str(sys.argv[1]), sys.argv[2]
                )
            ),
        )
