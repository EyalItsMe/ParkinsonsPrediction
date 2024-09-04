import torch
import torch.nn as nn
import torch.nn.functional as F


class MFCCCNN(nn.Module):
    def __init__(self):
        super(MFCCCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=13, out_channels=32, kernel_size=5, stride=3, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=3, padding=2)
        self.fc1 = nn.Linear(235520, 128)  # Adjust the dimension according to the output size
        self.fc2 = nn.Linear(128, 2)  # Output dimension is 2 (Healthy or Parkinson's)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
