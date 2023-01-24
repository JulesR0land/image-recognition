import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a convolution neural network
class Network(nn.Module):
    def __init__(self, input_shape):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)

        self.linear_shape = self.get_linear_shape(input_shape)
        self.fc1 = nn.Linear(self.linear_shape[0] * self.linear_shape[1] * self.linear_shape[2], 10)


    def get_linear_shape(self, input_shape):
        input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2])

        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))

        return ((output.shape[1], output.shape[2], output.shape[3]))


    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, self.linear_shape[0] * self.linear_shape[1] * self.linear_shape[2])
        output = self.fc1(output)

        return output