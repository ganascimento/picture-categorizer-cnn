import torch
import torch.nn as nn


class ImageClassificationNN(nn.Module):
    def __init__(self):
        super().__init__()

        # in_channels: Number of input channels. This corresponds to the number of channels in the input image. For example, an RGB image has 3 channels (red, green and blue)
        # out_channels: defines the output channels (or amount of filters). Each filter generates a feature map.
        # kernel_size: specifies the size of the kernel, which is the weight matrix that will slide across the image to extract features
        # padding: specifies the number of pixels that the image will be padded with zeros around the edges to maintain the image size
        # stride: defines the step (or shift) of the window as it moves across the input

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)  # Number of output classes, as the base is CIFAR-10, there are 10 classes

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)  # Flattening
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        return x
