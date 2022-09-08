import torch.nn as NN
import torch

# EEGNet
# Input 格式(B, C, H, W)
# 采样频率 128 HZ (若采样频率变化，调整卷积核大小以及池化尺寸)


# 定义参数
C = 16  # 脑电通道数
T = 15360  # 采样个数（采样频率*时间）
F1 = 8  # number of temporal filters
D = 2  # depth multiplier
F2 = 16  # number of pointwise filters   F2 = D * F1
num_classes = 2  # 分类数
p = 0.25  # Dropout参数 跨被试使用0.25


# EEGNet
class EEGNet(NN.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.C = C  # 脑电通道数
        self.T = T  # 采样个数（采样频率*时间）
        self.F1 = F1  # number of temporal filters
        self.D = D  # depth multiplier
        self.F2 = F2  # number of pointwise filters
        self.num_classes = num_classes  # 分类数
        self.p = p  # Dropout参数

        self.Block1 = NN.Sequential(
            NN.ZeroPad2d(padding=(49, 50, 0, 0)),  # padding (left, right, up, bottom)

            NN.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, 100), bias=False),  # (B, F1, C, T)

            NN.BatchNorm2d(num_features=self.F1)  # (B, F1, C, T)
        )

        # 深度卷积
        self.DepthwiseConv2D = NN.Conv2d(in_channels=self.F1, out_channels=self.D * self.F1,
                                         kernel_size=(self.C, 1), groups=self.F1, bias=False)  # (B, D*F1, 1, T)

        self.Block2 = NN.Sequential(
            NN.BatchNorm2d(num_features=self.D * self.F1),  # (B, D*F1, 1, T)

            NN.ELU(),

            NN.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),  # (B, D*F1, 1, T/4)

            NN.Dropout(p)
        )

        # 可分离卷积
        self.SeparableConv2D = NN.Sequential(
            NN.ZeroPad2d(padding=(12, 12, 0, 0)),

            NN.Conv2d(in_channels=self.D * self.F1, out_channels=self.D * self.F1,
                      kernel_size=(1, 25), groups=self.D * self.F1, bias=False),  # (B, D*F1, 1, T/4)

            NN.Conv2d(in_channels=self.D * self.F1, out_channels=self.F2,
                      kernel_size=(1, 1), bias=False)  # (B, F2, 1, T/4)
        )

        self.Block3 = NN.Sequential(
            NN.BatchNorm2d(self.F2),
            NN.ELU(),
            NN.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),  # (B, F2, 1, T/32)
            NN.Dropout(p)
        )

        # 全连接层
        self.fc1 = NN.Linear(int(self.F2 * int(self.T / 32)), self.num_classes)
        self.softmax = NN.Softmax(dim=-1)

    def forward(self, x):
        x = self.Block1(x)
        x = self.DepthwiseConv2D(x)
        x = self.Block2(x)
        x = self.SeparableConv2D(x)
        y = self.Block3(x)
        y = y.view(y.size(0), -1)
        x = self.fc1(y)
        output = self.softmax(torch.sigmoid(x))
        return output


