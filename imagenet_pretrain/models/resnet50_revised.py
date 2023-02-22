import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchinfo import summary
import torchaudio


class IRNet(nn.Module):
    def __init__(self,
                 pretrained=True,
                 feature_dim=2048,
                 n_fft=1024,
                 hop_len=512,
                 ch_expand='copy',
                 norm_mel=True,
                 interpolate=True,
                 discretize=False,
                 ):
        super(IRNet, self).__init__()
        weights = 'ResNet50_Weights.DEFAULT' if pretrained else None

        self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=22050,
                                                             n_fft=n_fft,
                                                             hop_length=hop_len,
                                                             # f_max=int(22050 / 2),
                                                             n_mels=128)

        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.interpolator = nn.Upsample(size=224, mode='bilinear', align_corners=True)

        self.backbone = torchvision.models.resnet50(weights=weights)
        self.backbone.fc = nn.Identity()
        self.bn = nn.BatchNorm1d(feature_dim)
        self.head = nn.Linear(feature_dim, 11)

        self.ch_expand = ch_expand
        self.normalize = norm_mel
        self.interpolate = interpolate
        self.discretize = discretize

    def forward(self, x):
        # computer acoustic features
        x = self.mel_spec(x)
        x = self.to_db(x)

        if self.normalize:
            x = self._normalize(x)

        x = x.unsqueeze(1)

        if self.ch_expand == 'copy':
            x = torch.cat((x, x, x), 1)

        x = self.interpolator(x)

        if self.discretize:
            x = self._discretize(x)

        # encode neural features
        emb = self.backbone(x)
        # classify
        x = self.bn(emb)
        x = self.head(x)

        return emb, x

    def _normalize(self, mel_spec):
        """
        perform normalization on mel-spectrogram (B x M_BIN X T)
        https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122/5
        """

        bs, m_bin, t = mel_spec.shape[0], mel_spec.shape[1], mel_spec.shape[2]
        mel_spec = mel_spec.contiguous().view(bs, -1)
        mel_spec -= mel_spec.min(1, True)[0]
        mel_spec /= mel_spec.max(1, True)[0]
        mel_spec = mel_spec.view(bs, m_bin, t)

        return mel_spec

    def _discretize(self, mel_spec):
        '''
        mapping mel-spectrogram to discrete values.
        :param mel_spec:
        :return: discrete mel-spec
        '''

        mel_spec = torch.clamp(mel_spec, 0.0, 1.0)
        boundaries = torch.linspace(start=0.0, end=1.0, steps=256).type_as(mel_spec)
        idx = torch.bucketize(mel_spec, boundaries).type_as(mel_spec)
        mel_spec = idx * (1 / 255)

        return mel_spec

if __name__ == '__main__':
    model = IRNet()
    summary(model, input_size=(2, 22050))