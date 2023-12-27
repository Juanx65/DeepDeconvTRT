import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Downsampling path
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Upsampling path
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        
        # Output path
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(8, 1, kernel_size=1)
        
    def forward(self, x):
        # Downsampling
        x1 = F.relu(self.conv1(x))
        x1p = self.pool(x1)
        x2 = F.relu(self.conv2(x1p))
        x2p = self.pool(x2)
        x3 = F.relu(self.conv3(x2p))
        x3p = self.pool(x3)
        x4 = F.relu(self.conv4(x3p))
        
        # Upsampling
        x5 = self.upconv1(x4)
        # Skip connection from x3 to x5
        x5 = torch.cat((x5, x3), dim=1)
        x5 = F.relu(self.conv5(x5))
        
        x6 = self.upconv2(x5)
        # Skip connection from x2 to x6
        x6 = torch.cat((x6, x2), dim=1)
        x6 = F.relu(self.conv6(x6))
        
        x7 = self.upconv3(x6)
        # Skip connection from x1 to x7
        x7 = torch.cat((x7, x1), dim=1)
        x7 = F.relu(self.conv7(x7))
        
        # Output
        output = self.output_conv(x7)
        #output = F.relu(output)
        output = torch.tanh(output)

        return output