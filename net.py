import torchvision
import torch.nn as nn


class LastLayer(nn.Module):
    """the last layer for inception_v3"""

    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=256),
            nn.ELU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=16),
        )
        self.classifier = nn.Linear(in_features=16, out_features=1)
        self.detector = nn.Linear(in_features=16, out_features=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        label = self.sigmoid(self.classifier(x))
        # box coordinates are numbers netween 0 and 299:
        box = 299 * self.sigmoid(self.detector(x))
        return label, box


class PretrainedNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.inception = torchvision.models.inception_v3(pretrained=True)
        # freezing the weights:
        for parameter in self.inception.parameters():
            parameter.requires_grad = False

        in_features = self.inception.AuxLogits.fc.in_features
        self.inception.AuxLogits.fc = LastLayer(in_features)

        in_features = self.inception.fc.in_features
        self.inception.fc = LastLayer(in_features)

    def forward(self, x):
        label, box = self.inception(x)
        return label, box
