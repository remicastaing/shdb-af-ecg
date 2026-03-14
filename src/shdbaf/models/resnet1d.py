import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, k=7):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=k, stride=stride, padding=p, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=k, stride=1, padding=p, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = F.relu(out + identity)
        return out


class ResNet1D(nn.Module):
    def __init__(self, in_ch=2, num_classes=4, base=32, blocks=(2, 2, 2), k=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=k, stride=2, padding=k // 2, bias=False),
            nn.BatchNorm1d(base),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        ch = base
        self.layer1, ch = self._make_layer(ch, base, blocks[0], stride=1, k=k)
        self.layer2, ch = self._make_layer(ch, base * 2, blocks[1], stride=2, k=k)
        self.layer3, ch = self._make_layer(ch, base * 4, blocks[2], stride=2, k=k)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, num_classes),
        )

    def _make_layer(self, in_ch, out_ch, n, stride, k):
        down = None
        if stride != 1 or in_ch != out_ch:
            down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = [BasicBlock1D(in_ch, out_ch, stride=stride, downsample=down, k=k)]
        for _ in range(1, n):
            layers.append(BasicBlock1D(out_ch, out_ch, stride=1, downsample=None, k=k))
        return nn.Sequential(*layers), out_ch

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)
