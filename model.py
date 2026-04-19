import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessNet(nn.Module):
    def __init__(self, num_blocks=10, num_filters=128):
        super(ChessNet, self).__init__()
        # Input layer (14 channels -> num_filters)
        self.conv_in = nn.Conv2d(14, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)
        
        # Build residual blocks (Depth for learning complex patterns)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])
        
        # Policy / Action output head
        self.conv_policy = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.bn_policy = nn.BatchNorm2d(32)
        
        # 64 squares from -> 64 squares to (4096 possible encoded actions)
        self.fc_policy = nn.Linear(32 * 8 * 8, 64 * 64)
        
    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
            
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1) # flatten
        p = self.fc_policy(p) # Q-values for actions
        return p
