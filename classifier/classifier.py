import torch

import torch.nn as nn
import torch.nn.functional as F


class WormClassifier(nn.Module):
    # IMG SIZE 24 x 24
    def __init__(self):
        super(WormClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 5, 1, 2)
        self.conv2 = nn.Conv2d(12, 24, 5, 1, 2)
        self.conv3 = nn.Conv2d(24, 36, 5, 1, 2)

        self.fc1 = nn.Linear(36, 1)

    def forward(self, x, encode=False):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 4)
        encoded = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(encoded))

        if encode:
            return encoded, x

        return x
