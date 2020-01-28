import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

with open('C:\\Users\\brain\\PycharmProjects\\Alpha-Zero-Neural-Network\\config.json') as json_data_file:
    config = json.load(json_data_file)


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()

        self.action_size = config["boardSize"] * config["boardSize"]
        self.board_x = config["boardSize"]
        self.board_y = config["boardSize"]
        self.conv1 = nn.Conv2d(1, 512, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=512, planes=512, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.action_size = config["boardSize"] * config["boardSize"]
        self.board_x = config["boardSize"]
        self.board_y = config["boardSize"]
        self.conv = nn.Conv2d(512, 3, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * self.board_x * self.board_y, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(512, 32, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(self.board_x * self.board_y * 32, self.action_size)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 3 * self.board_x * self.board_y)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, self.board_x * self.board_y * 32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
