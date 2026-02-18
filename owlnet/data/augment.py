import torch.nn as nn
import torchaudio.transforms as T



class Specaugment(nn.Module):
    def __init__(self, tmask_param=0, fmask_param=0) -> None:
        super().__init__()
        self.time_mask = T.TimeMasking(time_mask_param=tmask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=fmask_param)

    def forward(self, x):
        x = self.time_mask(x)
        x = self.freq_mask(x)
        return x
        