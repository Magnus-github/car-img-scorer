import torch
from torch import nn
import torch.nn.functional as F



class CNNImageScorer(nn.Module):
    def __init__(self, img_size: int = 224, num_channels: int = 64, num_scores: int = 2):
        super(CNNImageScorer, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(num_channels, 2*num_channels, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        fc_size = 2*num_channels * (img_size // 4) * (img_size // 4)
        self.fc1 = nn.Linear(fc_size, 1024)
        self.fc2 = nn.Linear(1024, num_scores)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    
    
if __name__ == "__main__":
    model = CNNImageScorer()
    print(model)