import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torchaudio


class IRNet(nn.Module):
    def __init__(self, num_classes=11):
        super(IRNet, self).__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=22050,
                                                             n_fft=1024,
                                                             hop_length=512,
                                                             f_max=int(22050 / 2),
                                                             n_mels=128)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(0.25),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(0.25)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(0.25)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(7, 10))
        )

        self.emb = nn.Linear(256, 1024)
        self.fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )

        # initialize
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_uniform_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x / (x.max(1, True)[0])
        mel = self.mel_spec(x)[:, :, :43]
        x = torch.log(mel + 1e-6)
        x = x.unsqueeze(1)
        x = x.transpose(-1, -2)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.shape[0], -1)
        emb = self.emb(x)
        x = self.fc(emb)
        return emb, x


if __name__ == '__main__':
    model = IRNet()
    summary(model, input_size=(2, 22050))
