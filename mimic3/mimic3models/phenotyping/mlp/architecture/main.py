import argparse
import copy
import os
import time

import torch
from joblib import dump
from mimic3models.phenotyping.dp_analysis.save_gradients import save_gradients_to_csv
from mimic3models import metrics
from mimic3models.phenotyping.load_preprocessed import load_cached_data
from mimic3models.phenotyping.logistic.main import load_process_data
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


def fit(model, train_loader, val_loader, criterion, optimizer, n_epochs, verbose=True):
    model.train()
    best_model = None
    best_val_loss = 10e5
    best_train_loss = 0
    increase = 0

    for epoch in range(n_epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            y_pred = model(data)

            loss = criterion(y_pred, target)
            train_epoch_loss += loss.item()

            loss.backward()
            #save_gradients_to_csv(model, epoch+1, "/mimic3experiments/mlp_gradients.csv")
            optimizer.step()

        with torch.no_grad():
            for data, target in val_loader:
                y_pred = model(data)
                val_loss = criterion(y_pred, target)
                val_epoch_loss += val_loss.item()

        if val_epoch_loss / len(val_loader) < best_val_loss:
            best_val_loss = val_epoch_loss / len(val_loader)
            best_train_loss = train_epoch_loss / len(train_loader)
            best_model = copy.deepcopy(model)
            increase = 0
        else:
            increase += 1

        if verbose:
            print('Epoch: {}. Train loss: {}. Validation loss: {}'.format(
                epoch + 1, train_epoch_loss / len(train_loader), val_epoch_loss / len(val_loader)
            ))

        if increase == 5:
            if verbose:
                print('\nBest training loss {}. Best validation loss {}'.format(best_train_loss, best_val_loss))
            break

    return best_model


def get_loader(X, y, batch_size, device=DEVICE):
    return DataLoader(
        TensorDataset(torch.tensor(X).float().to(device), torch.tensor(y).float().to(device)),
        batch_size=batch_size
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/phenotyping/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument(
        "--cached_data", default=True, help="Load the data from cached npy files"
    )
    args = parser.parse_args()
    print(args)

    start_all = time.time()
    start_data = time.time()

    if args.cached_data:
        train_X, train_y, val_X, val_y, test_X, test_y = load_cached_data(args.data)
    else:
        train_X, train_y, val_X, val_y, test_X, test_y = load_process_data(args)

    train_loader = get_loader(train_X, train_y, args.batch_size)
    val_loader = get_loader(val_X, val_y, args.batch_size)
    test_loader = get_loader(test_X, test_y, args.batch_size)

    print('Loading and parsing data took', time.time() - start_data, 'seconds.')

    print('Starting model fitting')
    start_model = time.time()
    model = MLPMultiLabel(train_X.shape[1], train_y.shape[1])
    model.to(DEVICE)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model = fit(model, train_loader, val_loader, criterion, optimizer, args.epochs)
    #dump(model, './models/phenotyping_mlp.joblib')

    model.eval()
    with torch.no_grad():
        train_activations = model(torch.tensor(train_X).to(DEVICE)).cpu()
        val_activations = model(torch.tensor(val_X).to(DEVICE)).cpu()
        test_activations = model(torch.tensor(test_X).to(DEVICE)).cpu()

        print('\nTraining data:')
        metrics.print_metrics_multilabel(train_y, train_activations)
        print('\nValidation data:')
        metrics.print_metrics_multilabel(val_y, val_activations)
        print('\nTest data:')
        metrics.print_metrics_multilabel(test_y, test_activations)

    print('Fitting and evaluating model took', time.time() - start_model, 'seconds')
    print('Time elapsed:', time.time() - start_all)
