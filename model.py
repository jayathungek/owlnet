import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the ResNet-50 basic block
class Bottleneck(nn.Module):
    expansion = 4  # ResNet-50 uses bottleneck blocks with expansion

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # Skip connection
        out = self.relu(out)

        return out

# Define the full ResNet-50 model
class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=32, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


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

        
class FrequencyAwareOwnNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers for narrow frequency bands (vertical filters)
        self.freq_conv1 = nn.Conv2d(1, 32, kernel_size=(21, 1), stride=(2, 1), padding=(2, 0))  # 1x5 kernel
        self.freq_conv2 = nn.Conv2d(32, 64, kernel_size=(31, 1), stride=(2, 1), padding=(3, 0))  # 1x7 kernel
        self.freq_conv3 = nn.Conv2d(64, 128, kernel_size=(41, 1), stride=(2, 1), padding=(5, 0))  # 1x11 kernel

        # Temporal Feature Extractors (horizontal filters)
        self.temp_conv1 = nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))  # 3x1 kernel
        self.temp_conv2 = nn.Conv2d(128, 128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2))  # 5x1 kernel

        self.gap = nn.AdaptiveAvgPool2d(1)
        # Squeeze-and-Excitation (SE) Block for Frequency Attention
        self.se_block = nn.Sequential(
            nn.Linear(128, 128 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(128 // 4, 128),
            nn.Sigmoid()
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 144, 256)  # Adjust for your spectrogram dimensions
        self.fc2 = nn.Linear(256, 128)  # Adjust for your spectrogram dimensions

    def forward(self, x):
        # Frequency-sensitive convolution layers
        # print(x.shape)
        x = F.relu(self.freq_conv1(x))
        x = F.relu(self.freq_conv2(x))
        x = F.relu(self.freq_conv3(x))

        # Temporal feature extraction
        x = F.relu(self.temp_conv1(x))
        x = F.relu(self.temp_conv2(x))

        # x = x.mean(dim=(2, 3))
        x_gap = self.gap(x).view(x.shape[0], x.shape[1])
        # Apply Frequency Attention (SE Block)
        se_weight = self.se_block(x_gap)  # Global channel pooling
        x = x * se_weight.view(x.shape[0], x.shape[1], 1, 1)  # Apply attention weights

        # Flatten & Fully Connected Layers
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

        

class AudioTransformer(nn.Module):
    def __init__(self, dim=128, depth=4, heads=8, mlp_dim=256, max_seq_len=512):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=(16, 16), stride=(16, 16))
        
        # Fixed positional embedding defined once
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=mlp_dim, activation='gelu'
            ),
            num_layers=depth
        )
        self.fc = nn.Linear(dim, 128)

    def forward(self, x):
        # Create patch embeddings
        x = self.patch_embed(x)  # (B, dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim) where N = variable length

        # Use positional embedding up to sequence length
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        x = self.transformer(x)
        x = x.mean(dim=1)
        return F.normalize(self.fc(x), p=2, dim=1) 