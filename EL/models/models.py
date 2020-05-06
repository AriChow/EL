import torch.nn as nn
import torch
from aviation.models.CVAE_parts import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SenderChexpertFull(nn.Module):
    def __init__(self, first_channel=64, input_channels=3, conv_layers=4, lin_layers=2, num_ftrs=2048, output_size=512):
        super(SenderChexpertFull, self).__init__()
        self.channel_in = first_channel
        self.nc = input_channels
        self.num_ftrs = num_ftrs
        self.output_size = output_size

        self.inc = inconv(self.nc, self.channel_in)
        self.downs = []
        for i in range(conv_layers-1):
            self.downs.append(inconv(self.channel_in, self.channel_in // 2).to(device))
            self.channel_in //= 2
        # self.pool = nn.AvgPool2d(14)
        # self.linear_downs = []
        # for i in range(lin_layers):
        #     self.linear_downs.append(nn.Linear(self.channel_in, self.channel_in // 2).to(device))
        #     self.channel_in //= 2

        self.embed = nn.Linear(1568, self.num_ftrs)
        self.fc = nn.Linear(self.num_ftrs, self.output_size)

    def forward(self, x):
        x = self.inc(x)
        for i in range(len(self.downs)):
            x = self.downs[i](x)
        # x = self.pool(x)

        x = x.view(x.shape[0], -1)
        # for i in range(len(self.linear_downs)):
        #     x = self.linear_downs[i](x)

        x = self.embed(x)
        x = self.fc(x)
        return x


class SenderChexpert(nn.Module):
    def __init__(self, model, num_ftrs=2048, output_size=512):
        super(SenderChexpert, self).__init__()
        self.fc = nn.Linear(num_ftrs, output_size)
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        x = self.fc(x)
        return x


class ReceiverChexpert(nn.Module):
    def __init__(self, input_size=512):
        super(ReceiverChexpert, self).__init__()
        self.fc = nn.Linear(input_size, 2)
        self.out = nn.Softmax()

    def forward(self, channel_input, receiver_input=None):
        x = self.fc(channel_input)
        return self.out(x)


class SenderOncoFeat(nn.Module):
    def __init__(self, input_size=28, hidden_size=20):
        super(SenderOncoFeat, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.out = nn.ReLU()

    def forward(self, input):
        return self.out(self.fc(input))


class ReceiverOncoFeat(nn.Module):
    def __init__(self, input_size=20, output_size=4):
        super(ReceiverOncoFeat, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.out = nn.Softmax()

    def forward(self, input, receiver_input=None):
        return self.out(self.fc(input))


class SenderOncoImages(nn.Module):
    def __init__(self, hidden_size=20):
        super(SenderOncoImages, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(4, 10),
                                  nn.BatchNorm2d(),
                                  nn.ReLU(),
                                  nn.AvgPool2d(3),
                                  nn.ReLU)

    def forward(self, input):
        return self.conv(input)


class ReceiverOncoImages(nn.Module):
    def __init__(self):
        super(ReceiverOncoImages, self).__init__()
        pass

    def forward(self):
        pass
