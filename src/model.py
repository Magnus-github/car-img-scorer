import torch
from torch import nn
import torch.nn.functional as F



class CNNImageScorer(nn.Module):
    def __init__(self, img_size: int = 224, num_scores: int = 2):
        super(CNNImageScorer, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        fc_size = 256 * (img_size // 8) * (img_size // 8)
        self.fc1 = nn.Linear(fc_size, 1024)
        self.fc2 = nn.Linear(1024, num_scores)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
if __name__ == "__main__":
    model = CNNImageScorer()
    print(model)