import os
import pandas as pd
import datetime


def _timestamp():
    return "{:%Y-%b-%d %H:%M:%S}".format(datetime.datetime.now())


def write_stats(stats_dict: dict, path: str=None, columns=None):
    if path is None:
        path = "/proj/tobaben/mimic3experiments/dp_stats.csv"

    if columns is None:
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
            "delta",
            "seed",
            "timestamp",
        ]

    stats_list = []

    for c in columns:
        if c == "timestamp":
            stats_list.append(_timestamp())
        else:
            stats_list.append(stats_dict.get(c))

    df = pd.DataFrame([stats_list])
    df.columns = columns
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

