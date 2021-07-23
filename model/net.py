import torch.nn as nn


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, k=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, k=2),

            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3)
        )
        self.fcnet = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2)
        )

    def forward_once(self, ipimg):
        X = self.convnet(ipimg)
        X = X.view(X.size()[0], -1)
        op_val = self.fcnet(X)

        return op_val

    def forward(self, ip1, ip2):
        op1 = self.forward_once(ip1)
        op2 = self.forward_once(ip2)

        return op1, op2

