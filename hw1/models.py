import torch.nn as nn


def conv_relu(cin, cout, k, stride=1):
    return [
        nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.ReLU(inplace=True),
    ]


class SmallCNN(nn.Module):
    """3xSxS -> 100 logits. Output resolutions: S/4, S/4, S/8, S/8, S/16, S/16."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            *conv_relu(3, 32, 7, stride=2),
            nn.MaxPool2d(3, stride=2, padding=1),
            *conv_relu(32, 64, 5),
            *conv_relu(64, 128, 3, stride=2),
            *conv_relu(128, 256, 1),
            *conv_relu(256, 256, 3, stride=2),
            *conv_relu(256, 512, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


def build_model():
    return SmallCNN(num_classes=100)
