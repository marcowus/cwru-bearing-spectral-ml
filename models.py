import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierFC(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassifierFC, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048*128)
        self.fc2 = nn.Linear(2048*16, 2048)
        self.fc3 = nn.Linear(2048, 128)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x
    
class ClassifierCNN_binary(nn.Module):
    def __init__(self):
        super(ClassifierCNN_binary, self).__init__()
        # Input: 1 channel, 128x2048
        # Output: 8 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 5), padding=(1, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # After conv1 and pool1: shape is (batch_size, 8, 64, 1024)
        # Output: 16 channels
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 5), padding=(1, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # After conv2 and pool2: shape is (batch_size, 16, 32, 512)
        
        # Calculate the size of the flattened tensor after the last pooling layer
        # 16 * (128//4) * (2048//4) = 16 * 32 * 512 = 262144
        # 16 * (N // 4) * (win_width // 4)
        # You may need to adjust these dimensions based on your specific pooling/stride choices
        self.flattened_size = 16 * 32 * 512
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1) # 1 output for binary classification

    def forward(self, x):
        # Input is (batch_size, 128, 2048). Need to add a channel dimension.
        x = x.view(-1, 1, 128, 2048)
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
    

class ClassifierCNN_multi_class(nn.Module):
    def __init__(self, N, win_width):
        super(ClassifierCNN_multi_class, self).__init__()
        self.N = N
        self.win_width = win_width
        # Input: 1 channel, 128x2048
        # Output: 8 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 5), padding=(1, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # After conv1 and pool1: shape is (batch_size, 8, 64, 1024)
        # Output: 16 channels
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 5), padding=(1, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # After conv2 and pool2: shape is (batch_size, 16, 32, 512)
        
        # Calculate the size of the flattened tensor after the last pooling layer
        # 16 * (128//4) * (2048//4) = 16 * 32 * 512 = 262144
        # You may need to adjust these dimensions based on your specific pooling/stride choices
        # self.flattened_size = 16 * 32 * 512
        self.flattened_size = 16 * (self.N // 4) * (self.win_width // 4)
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        # Input is (batch_size, 128, 2048). Need to add a channel dimension.
        x = x.view(-1, 1, self.N, self.win_width)
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x