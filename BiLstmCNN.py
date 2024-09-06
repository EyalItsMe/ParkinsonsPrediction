import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim, nmfcc):
        super(CNNBiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=nmfcc, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bi_lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(512, 128)  # BiLSTM outputs twice the hidden size (256*2)
        self.fc2 = nn.Linear(128, 2)  # Output dimension is 2 (Healthy or Parkinson's)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)

        # Prepare for LSTM (batch_size, seq_length, feature_size)
        x = x.permute(0, 2, 1)

        # BiLSTM layer
        x, _ = self.bi_lstm(x)

        # Take the final hidden state output of the BiLSTM
        x = x[:, -1, :]  # Get the output from the last time step

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
