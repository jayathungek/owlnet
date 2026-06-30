import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Model needs to take into account the following features as 
# described by A. Dreiss et al.:

# • Call duration
# • Absolute loudness (biased by distance from mic)
# • Loudness deviation (0-1),which represents whether the first or second half of the call is loudest (>0.5: call is louder at the end than the beginning) 
# • Mean frequency
# • Upper frequency (25% of call loudness is above this).
# • Frequency variation (SD over time within the call)

class OwlNet(nn.Module):
    def __init__(
        self,
        encoder_dim,
        embedding_dim,
        dropout,
        num_dreiss_features,
        use_attention=False
    ):
        super().__init__()

        self.out_dim = 2 * encoder_dim
        self.num_dreiss_features = num_dreiss_features
        self.use_attention = use_attention
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(128, 16), stride=(2, 2), padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(8, 8), stride=(2, 2), padding=1)  
        self.conv3 = nn.Conv2d(64, encoder_dim, kernel_size=(5, 5), stride=(2, 2), padding=1) 
        self.conv4 = nn.Conv2d(encoder_dim, encoder_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(encoder_dim)
        self.bn4 = nn.BatchNorm2d(encoder_dim)

        # TODO: for some reason, using attention breaks the validation 
        # visualisation tool. It appears to reflect the embeddings on the y axis??
        if use_attention:
            encoder_layer = TransformerEncoderLayer(self.out_dim, 8, batch_first=True)
            self.attention_module = TransformerEncoder(encoder_layer, num_layers=6, enable_nested_tensor=False)

        self.dropout = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_head = nn.Sequential(
            nn.LayerNorm(self.num_dreiss_features),
            nn.Linear(self.num_dreiss_features, 32),
            nn.ReLU(),
            nn.Linear(32, encoder_dim),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x, dreiss_features):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.gap(x)
        x = torch.flatten(x, start_dim=1) 
        if self.use_attention:
            x = self.attention_module(x)
        z_spec = self.dropout(x)
        z_feat = self.feat_head(dreiss_features)
        z = torch.concat([z_spec, z_feat], dim=-1)
        out = self.fc(z)
        return out