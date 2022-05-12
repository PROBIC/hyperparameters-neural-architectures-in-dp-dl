import argparse
import os

from mimic3models.phenotyping.logistic.main import load_process_data
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the data of phenotyping task",
        default=os.path.join(os.path.dirname(__file__), "../../data/phenotyping/"),
    )
    parser.add_argument(
        "--period",
        type=str,
        default="all",
        help="specifies which period extract features from",
        choices=[
            "first4days",
            "first8days",
            "last12hours",
            "first25percent",
            "first50percent",
            "all",
        ],
    )
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        help="specifies what features to extract",
        choices=["all", "len", "all_but_len"],
    )
    args = parser.parse_args()
    train_X, train_y, val_X, val_y, test_X, test_y = load_process_data(args)

    print("Saving data to ", args.data)
    np.save(os.path.join(args.data, "train_X.npy"), train_X)
    np.save(os.path.join(args.data, "train_y.npy"), train_y)
    np.save(os.path.join(args.data, "val_X.npy"), val_X)
    np.save(os.path.join(args.data, "val_y.npy"), val_y)
    np.save(os.path.join(args.data, "test_X.npy"), test_X)
    np.save(os.path.join(args.data, "test_y.npy"), test_y)


if __name__ == "__main__":
    main()
