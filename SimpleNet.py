import torch.nn as nn
import torch.nn.functional as F
import torch

#Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
# class SimpleNet(nn.Module):
#     def __init__(self) -> None:
#         super(SimpleNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
# #         x = x.view(-1, 16 * 5 * 5)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
     
# class SimpleNet(nn.Module):
#     def __init__(self) -> None:
#         super(SimpleNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.bn1 = nn.BatchNorm2d(6)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.bn3 = nn.BatchNorm1d(120)
#         self.fc2 = nn.Linear(120, 84)
#         self.bn4 = nn.BatchNorm1d(84)
#         self.fc3 = nn.Linear(84, 10)

#     # pylint: disable=arguments-differ,invalid-name
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Compute forward pass."""
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.bn3(self.fc1(x)))
#         x = F.relu(self.bn4(self.fc2(x)))
#         x = self.fc3(x)
#         return x
    
class SimpleNet(nn.Module):
    """SimpleNet."""
    def __init__(self) -> None:
        """SimpleNet Builder."""
        super(SimpleNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            self.bn1,
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            self.bn2,
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            self.bn3,
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Fully Connected Layer
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward."""
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x    
    

   