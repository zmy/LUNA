import logging
import random
from os import path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils import resample
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
import wandb

from .args import define_main_parser

logger = logging.getLogger(__name__)
log = logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def get_args():
    parser = define_main_parser()
    # Prepares specific arguments for Fraud Detection task.
    parser.add_argument('--model_name_or_path', type=str, default=None, help='directory of the TabFormer model')
    parser.add_argument('--cached_feature_dir', type=str, default=None, help='directory of the cached feature')
    parser.add_argument('--lr', type=str, default=1e-3, help='learning rate in fine-tuning')
    parser.add_argument('--upsample', action='store_true', help='upsample training data')

    args = parser.parse_args()

    return args


class TransactionFeatureDataset(Dataset):
    """Transaction Feature Dataset for Fraud Detection task."""

    def __init__(self, data, label, with_upsample=False):
        """Args:
            - data: sample feature extracted from TabBERT.
            - label: label in sample (window) level.
            - with_upsample: if True, upsample fraudulent data to have the same amount with non-fraudulent data.
        """
        self.data = data
        self.label = label
        if with_upsample:
            self._upsample()

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)

    def _upsample(self):
        logger.info('Upsample fraudulent samples.')
        non_fraud = self.data[self.label == 0]
        fraud = self.data[self.label == 1]
        fraud_upsample = resample(fraud, replace=True, n_samples=non_fraud.shape[0], random_state=2022)
        self.data = torch.cat((fraud_upsample, non_fraud))
        self.label = torch.cat((torch.ones(fraud_upsample.shape[0]), torch.zeros(non_fraud.shape[0])))


class LSTMPredictionHead(nn.Module):
    """LSTM prediction head for binary classification."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, dropout=0.1,
                            batch_first=True)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (h_n, h_c) = self.lstm(x)
        # Use the last output for prediction
        out = self.linear(out[:, -1, :])

        pred = self.sigmoid(out)

        return pred


def evaluate(model, data_loader):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for (data, label) in tqdm(data_loader):
            data, label = data.cuda(), label.cuda()
            pred = model(data).squeeze()
            pred = torch.round(pred)
            predictions.append(pred)
            labels.append(label)
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy())

    return f1


def main():
    args = get_args()

    # Initialize wandb and log config.
    config = dict(
        learning_rate=args.lr,
        batch_size=args.per_device_train_batch_size,
        epoch=args.num_train_epochs,
        use_numtok=args.use_numtok,
        number_model_config=args.number_model_config,
        model_name_or_path=args.model_name_or_path,
    )
    wandb.init(
        config=config,
    )

    # random seeds
    seed = args.seed
    logger.info(f'Using seed = {seed}')
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

    logger.info(f'load features and labels from {args.cached_feature_dir}')

    train_feature_fname = path.join(args.cached_feature_dir, 'train_feature.pth')
    train_label_fname = path.join(args.cached_feature_dir, 'train_label.pth')
    valid_feature_fname = path.join(args.cached_feature_dir, 'valid_feature.pth')
    valid_label_fname = path.join(args.cached_feature_dir, 'valid_label.pth')
    test_feature_fname = path.join(args.cached_feature_dir, 'test_feature.pth')
    test_label_fname = path.join(args.cached_feature_dir, 'test_label.pth')

    train_features = torch.load(train_feature_fname)
    train_labels = torch.load(train_label_fname)
    valid_features = torch.load(valid_feature_fname)
    valid_labels = torch.load(valid_label_fname)
    test_features = torch.load(test_feature_fname)
    test_labels = torch.load(test_label_fname)

    train_dataset = TransactionFeatureDataset(train_features, train_labels, with_upsample=args.upsample)
    valid_dataset = TransactionFeatureDataset(valid_features, valid_labels)
    test_dataset = TransactionFeatureDataset(test_features, test_labels)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size)

    logger.info(f'Building LSTM prediction head.')
    model = LSTMPredictionHead()
    model = model.cuda()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_f1 = 0
    final_result_f1 = 0

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        for (data, labels) in tqdm(train_dataloader):
            optimizer.zero_grad()
            data, labels = data.cuda(), labels.float().cuda()
            pred = model(data).squeeze()
            loss = loss_fn(pred, labels)
            total_loss += loss.item() * data.shape[0]
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_dataset)

        # Evaluate on validation set.
        valid_f1 = evaluate(model, valid_dataloader)
        test_f1 = evaluate(model, test_dataloader)

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            final_result_f1 = test_f1
        logger.info(
            f'Epoch: {epoch}, loss = {avg_loss}, valid f1 = {valid_f1}, final result f1 = {final_result_f1}')
        wandb.log({
            'loss': avg_loss,
            'valid_f1': valid_f1,
            'test_f1': test_f1,
            'final_result_f1': final_result_f1
        })


if __name__ == '__main__':
    main()
