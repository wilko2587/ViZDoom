import torch
from torch import nn

class SmallNet(nn.Module):
    """
    For input [3 x 128 x 160]

    conv2d: kernel size 3, 16 out channels, padding 1
    [16 x 128 x 160]
    maxpool: kernel size 4
    [16 x 32 x 40]
    ReLU
    [16 x 32 x 40]

    conv2d: kernel size 3, 16 our channels, padding 1
    [16 x 32 x 40]
    maxpool: kernel size 4
    [16 x 8 x 10]
    ReLU
    [16 x 8 x 10]

    linear: in (8 * 10 * 16)
    [num_actions]
    """
    def __init__(self,
                 in_height: int,
                 in_width: int,
                 num_classes: int
                 ):

        h = in_height
        w = in_width

        super(SmallNet, self).__init__()

        self.c1 = nn.Conv2d(kernel_size=3,
                            in_channels=3,
                            out_channels=16,
                            stride=1,
                            padding=1)
        self.m1 = nn.MaxPool2d(kernel_size=4)
        self.r1 = nn.ReLU()

        self.c2 = nn.Conv2d(kernel_size=3,
                            in_channels=16,
                            out_channels=16,
                            stride=1,
                            padding=1)

        self.m2 = nn.MaxPool2d(kernel_size=4)
        self.r2 = nn.ReLU()

        self.fc1 = nn.Linear(int(h/16 * w/16 * 16), num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.m1(x)
        x = self.r1(x)
        x = self.c2(x)
        x = self.m2(x)
        x = self.r2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x.flatten()


class MedNet(nn.Module):
    """
    For input [3 x 128 x 160]

    -----
    conv2d: kernel size 3, 16 out channels, padding 1
    [16 x 128 x 160]
    maxpool: kernel size 2
    [16 x 64 x 80]
    ReLU
    [16 x 64 x 80]
    ------

    ------
    conv2d: kernel size 3, 16 our channels, padding 1
    [16 x 32 x 40]
    maxpool: kernel size 2
    [16 x 32 x 40]
    ReLU
    [16 x 32 x 40]
    ------

    ------
    conv2d: kernel size 3, 16 our channels, padding 1
    [16 x 32 x 40]
    maxpool: kernel size 2
    [16 x 16 x 20]
    ReLU
    [16 x 16 x 20]
    ------

    ------
    conv2d: kernel size 3, 16 our channels, padding 1
    [16 x 16 x 20]
    maxpool: kernel size 2
    [16 x 8 x 10]
    ReLU
    [16 x 8 x 10]
    ------

    linear: in (8 * 10 * 16): out ( 10 * 16 )
    dropout
    linear: in ( 10 * 16): out ( 10 * 16 )
    dropout
    linear: in ( 10 * 16): out [num_actions]
    [num_actions]
    """

    def __init__(self,
                 in_height: int,
                 in_width: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 ):

        h = in_height
        w = in_width

        super(MedNet, self).__init__()

        self.c1 = nn.Conv2d(kernel_size=3,
                            in_channels=3,
                            out_channels=16,
                            stride=1,
                            padding=1)
        self.m1 = nn.MaxPool2d(kernel_size=2)
        self.r1 = nn.ReLU()

        self.c2 = nn.Conv2d(kernel_size=3,
                            in_channels=16,
                            out_channels=16,
                            stride=1,
                            padding=1)

        self.m2 = nn.MaxPool2d(kernel_size=2)
        self.r2 = nn.ReLU()

        self.c3 = nn.Conv2d(kernel_size=3,
                            in_channels=16,
                            out_channels=16,
                            stride=1,
                            padding=1)

        self.m3 = nn.MaxPool2d(kernel_size=2)
        self.r3 = nn.ReLU()

        self.c4 = nn.Conv2d(kernel_size=3,
                            in_channels=16,
                            out_channels=16,
                            stride=1,
                            padding=1)

        self.m4 = nn.MaxPool2d(kernel_size=2)
        self.r4 = nn.ReLU()

        h = int(h/16)
        w = int(w/16)
        self.fc1 = nn.Linear(h * w * 16, 256)
        self.d1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, 128)
        self.d2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(128, 128)
        self.d3 = nn.Dropout(p=dropout)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.m1(x)
        x = self.r1(x)
        x = self.c2(x)
        x = self.m2(x)
        x = self.r2(x)
        x = self.c3(x)
        x = self.m3(x)
        x = self.r3(x)
        x = self.c4(x)
        x = self.m4(x)
        x = self.r4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.d1(x)
        x = self.fc2(x)
        x = self.d2(x)
        x = self.fc3(x)
        x = self.d3(x)
        x = self.fc4(x)
        return x.flatten()


class VGG(nn.Module):
    def __init__(
            self,
            architecture,
            in_channels=3,
            in_height=224,
            in_width=224,
            num_hidden=4096,
            num_classes=1000
    ):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.convs = self.init_convs(architecture)
        self.fcs = self.init_fcs(architecture)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcs(x)
        return torch.nn.functional.softmax(x.flatten(), dim=0)

    def init_fcs(self, architecture):
        pool_count = architecture.count("M")
        factor = (2 ** pool_count)
        if (self.in_height % factor) + (self.in_width % factor) != 0:
            raise ValueError(
                f"`in_height` and `in_width` must be multiples of {factor}"
            )
        out_height = self.in_height // factor
        out_width = self.in_width // factor
        last_out_channels = next(
            x for x in architecture[::-1] if type(x) == int
        )
        return nn.Sequential(
            nn.Linear(
                last_out_channels * out_height * out_width,
                self.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_hidden, self.num_classes)
        )

    def init_convs(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    ]
                )
                in_channels = x
            else:
                layers.append(
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                )

        return nn.Sequential(*layers)

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

if __name__ == '__main__':
    VGG16 = VGG(
        in_channels=3,
        in_height=160,
        in_width=128, # width needs 8 padding to fit architecture (divisible by 32)
        architecture=VGG_types["VGG11"]
    )