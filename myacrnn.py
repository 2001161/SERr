import torch
import torch.nn.functional as F
from torch import nn


class acrnn(nn.Module):
    def __init__(self):
        super(acrnn, self).__init__()

        # 1. 3D Convolution layers with LeakyReLU activation
        self.convs = nn.ModuleList([
            nn.Conv2d(3, 128, (5, 3), padding=(2, 1)),
            nn.Conv2d(128, 256, (5, 3), padding=(2, 1)),
            nn.Conv2d(256, 256, (5, 3), padding=(2, 1)),
            nn.Conv2d(256, 256, (5, 3), padding=(2, 1)),
            nn.Conv2d(256, 256, (5, 3), padding=(2, 1)),
            nn.Conv2d(256, 256, (5, 3), padding=(2, 1))
        ])
        self.leaky_relu = nn.LeakyReLU()

        # Max-pooling layer after the first convolutional layer
        self.maxpool = nn.MaxPool2d((2, 2))

        # Dimensionality reduction linear layer after CNN
        self.linear = nn.Linear(5120, 768)

        # LSTM layer
        self.lstm = nn.LSTM(768, 128, bidirectional=True)

        # 3. Attention layer
        self.attention = nn.Linear(256, 1)

        # 4. Fully connected layer
        self.fully_connected = nn.Linear(256, 64)
        self.batchnorm = nn.BatchNorm1d(64)

        # 5. Output layer (Softmax is applied in the forward method)
        self.classifier = nn.Linear(64, 4)

    def forward(self, x):
        # CRNN part
        for i, conv in enumerate(self.convs):
            x = self.leaky_relu(conv(x))
            if i == 0:  # After the first convolution layer
                x = self.maxpool(x)

        x = x.permute(0, 2, 3, 1)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])  # Flatten the tensor
        x = self.leaky_relu(self.linear(x))
        x = x.reshape(-1, 150, 768)

        x, _ = self.lstm(x)

        # Attention part
        attn_scores = self.attention(x)
        attn_scores = torch.softmax(attn_scores, dim=1)
        x = torch.sum(attn_scores * x, dim=1)

        # Fully connected layers and output
        x = self.leaky_relu(self.fully_connected(x))
        x = self.batchnorm(x)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)

        return x
