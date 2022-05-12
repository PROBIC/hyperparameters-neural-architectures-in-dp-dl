from sklearn.metrics import auc
from mimic3models.phenotyping.mlp.main_train_func import train_with_params
import numpy as np

if __name__ == "__main__":
    for data_shuffle_seed in [472368, 374148, 993935, 402060, 521365]:
        macro_auc = list()
        micro_auc = list()
        for seed in [472368, 374148, 993935, 402060, 521365]:
            print(data_shuffle_seed, seed, flush=True)
            auc_dict, model = train_with_params(
                custom_params={
                    "learning_rate": 0.00022101564344668978,
                    "l2_regulizer": 0.000506456911848878,
                    "num_epochs": 200,
                    "batch_size": 1345,
                },
                seed=seed,
                data_shuffle_seed=data_shuffle_seed,
                stats_path="/mimic3experiments/mlp_data_shuffle.csv",
            )
            macro_auc.append(auc_dict.get("ave_auc_macro"))
            micro_auc.append(auc_dict.get("ave_auc_micro"))
        print(
            data_shuffle_seed,
            "ave_auc_macro mean:",
            np.mean(macro_auc),
            "(std:",
            str(np.std(macro_auc)) + ")",
            "ave_auc_micro mean:",
            np.mean(micro_auc),
            "(std:",
            str(np.std(micro_auc)) + ")",
            flush=True,
        )

