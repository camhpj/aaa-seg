from typing import Any

import torch.nn as nn
from torch.nn.common_types import _size_2_t


class BasicBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t,
            padding: _size_2_t,
            dilation: _size_2_t,
        ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, x) -> Any:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out


class Down(nn.Module):
    def __init__(self, kernel_size: int):
        super(Down, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=kernel_size)

    def forward(self, x) -> Any:
        out = self.down(x)
        return out


class Up(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x) -> Any:
        out = self.up(x)
        return out



class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            n_classes: int,
            kernel_size: _size_2_t | 3,
            stride: _size_2_t | 1,
            padding: _size_2_t | 1,
            dilation: _size_2_t | 1,
        ) -> None:
        # encoder
        self.conv1 = BasicBlock(in_channels, 64, kernel_size, stride, padding, dilation)
        self.pool1 = Down(2)
        self.conv2 = BasicBlock(64, 128, kernel_size, stride, padding, dilation)
        self.pool2 = Down(2)
        self.conv3 = BasicBlock(128, 256, kernel_size, stride, padding, dilation)
        self.pool3 = Down(2)
        self.conv4 = BasicBlock(256, 512, kernel_size, stride, padding, dilation)
        self.pool4 = Down(2)

        self.conv5 = BasicBlock(512, 1024, kernel_size, stride, padding, dilation)

        # decoder
        self.scale1 = Up(1024)
        self.conv6 = BasicBlock(1024, 512, kernel_size, stride, padding, dilation)
        self.scale2 = Up(512)
        self.conv7 = BasicBlock(512, 256, kernel_size, stride, padding, dilation)
        self.scale3 = Up(256)
        self.conv8 = BasicBlock(256, 128, kernel_size, stride, padding, dilation)
        self.scale4 = Up(128)
        self.conv9 = BasicBlock(128, 64, kernel_size, stride, padding, dilation)

        self.conv10 = nn.Conv2d(64, n_classes, kernel_size=1)


    def forward(self, x) -> Any:
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.conv4(out)
        out = self.pool4(out)
        out = self.conv5(out)
        out = self.scale1(out)
        out = self.conv6(out)
        out = self.scale2(out)
        out = self.conv7(out)
        out = self.scale3(out)
        out = self.conv8(out)
        out = self.scale4(out)
        out = self.conv9(out)
        out = self.conv10(out)
        return out
