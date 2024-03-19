import argparse
import time
import torch


def validate(model, val_dataloader, epoch):
    """
    Validate given model.

        Args:
            model: Current quantum model.
            val_dataloader: Validation data.
            epoch: Index of current epoch for logging.
        Return:
            Runtime of validation.
    """
    start_time = time.perf_counter()
    # get 1000 random samples and flatten
    data_val, target_val = next(iter(val_dataloader))
    data_val = torch.reshape(data_val, (data_val.shape[0], -1))
    # no gradients required for validation
    with torch.no_grad():
        prediction_val = torch.nn.functional.softmax(model(data_val), dim=1)
        print('VALIDATE [Epoch {}] Accuracy: {:.1f}%'
              .format(epoch,
                      # determine accuracy, i.e. number of correctly classified samples
                      100 * torch.sum(torch.eq(torch.argmax(prediction_val, dim=1), target_val)) / len(data_val)))
    return time.perf_counter() - start_time


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=48,
                        help='Number of data samples for each update step.')
    parser.add_argument('--episodes', type=int, default=12000,
                        help='Maximum number of episodes to train for.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set seed for reproducibility.')
    parser.add_argument('--use_qml', action='store_true',
                        help='Train using qiskit-machine-learning instead of qiskit-torch-module.')
    return parser.parse_args()
