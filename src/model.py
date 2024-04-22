from torch import nn
from torchvision.models import resnet18


class ResNet(nn.Module):
    def __init__(self, resmodel=resnet18):
        super().__init__()
        self.model = resmodel()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=2, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)