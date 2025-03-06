import torch
import torch.nn as nn
import torch.nn.functional as F


class OwlNet(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(150, 64), stride=(2, 2), padding=2)  # (256,70) → (128,35)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(8, 8), stride=(2, 2), padding=1)  # (128,35) → (64,18)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=1)  # (64,18) → (32,9)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)  # (32,9) → (16,5)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 256)    , #Flatten → 128-D embedding
            nn.BatchNorm1d(256)          ,
            nn.ReLU()                     ,
            nn.Linear(256, 512),  # Flatten → 128-D embedding
            nn.ReLU()                     ,
            nn.Linear(512, embedding_dim),  # Flatten → 128-D embedding
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)  # Flatten before FC layer
        x = self.dropout(x)
        x = self.fc(x)

        return x