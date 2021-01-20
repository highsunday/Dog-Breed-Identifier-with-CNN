import torch
import torch.nn as nn
import torch.nn.functional as F     # 激励函数都在这
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 100, 100)
            nn.Conv2d(
                in_channels=3,      # input height
                out_channels=64,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (32, 100, 100)
            #nn.BatchNorm2d(64),
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (64, 50, 50)
        )
        self.conv2 = nn.Sequential(  # input shape (64, 50, 50)
            nn.Conv2d(64, 128, 3, 1, 1,bias=False),      # output shape (128, 50, 50)
            #nn.BatchNorm2d(128),
            nn.ReLU(),    # activation
            nn.Conv2d(128, 128, 3, 1, 1,bias=False),      # output shape (128, 50, 50)
            #nn.BatchNorm2d(128),
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (128, 25, 25)
        )
        self.out1 = nn.Linear(128 * 25 * 25, 240)   # fully connected layer, output 120 classes
        self.out2 = nn.Linear(240, 120)   # fully connected layer, output 120 classes

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 50 * 50)
        x = self.out1(x)
        output = self.out2(x)
        return output
    
    
