import torch.nn as nn
import torch
import torch.nn.functional as F

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)

class ConvBlockRes(nn.Module):
    # momentum value was set as per the original implementation
    # https://github.com/haoheliu/voicefixer/blob/main/voicefixer/restorer/model_kqq_bn.py
    def __init__(self, in_channels, out_channels, size, stride_val, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        pad = size//2
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(size, size),
            stride=(stride_val, stride_val),
            padding=(pad, pad),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(size, size),
            stride=(stride_val, stride_val),
            padding=(pad, pad),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        
        if stride_val != 1:
            self.residual = False
        else:
            self.residual = True

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, x):
        origin = x
        x = self.conv1(F.leaky_relu_(self.bn1(x), negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))

        if self.residual:
            return origin + x
        else:
            return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(in_channels, 32, kernel_size=3,)),
            nn.LeakyReLU(0.01, True),
        )
        self.convres1 = ConvBlockRes(in_channels=32, out_channels=32, size=3, stride_val=1)
        self.convres2 = ConvBlockRes(in_channels=32, out_channels=32, size=3, stride_val=1)
        self.convres3 = ConvBlockRes(in_channels=32, out_channels=64, size=3, stride_val=2)
        self.convres4 = ConvBlockRes(in_channels=64, out_channels=64, size=3, stride_val=1)
        self.convres5 = ConvBlockRes(in_channels=64, out_channels=32, size=3, stride_val=2)
        self.convres6 = ConvBlockRes(in_channels=32, out_channels=32, size=3, stride_val=1)
        self.convres7 = ConvBlockRes(in_channels=32, out_channels=32, size=3, stride_val=2)
        self.convres8 = ConvBlockRes(in_channels=32, out_channels=32, size=3, stride_val=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.convres1(x)
        x = self.convres2(x)
        x = self.convres3(x)
        x = self.convres4(x)
        x = self.convres5(x)
        x = self.convres6(x)
        x = self.convres7(x)
        x = self.convres8(x)
        return x



class FrequencyDiscriminator(nn.Module):
    def __init__(self, in_channel = 2, fft_size = 2048, hop_length = 441, window="hann_window"):
        super(FrequencyDiscriminator, self).__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = fft_size
        self.window = getattr(torch, window)(self.win_length)
        self.stft_channel = fft_size // 2 + 1
        self.disc =Discriminator(in_channel)

    def forward(self, x):
        self.window = self.window.to(x.device)
        x_stft = torch.stft(x, self.fft_size, self.hop_length, self.win_length, self.window, return_complex=False)
        x_stft = x_stft.permute(0, 3, 1, 2)
        x = self.disc(x_stft)

        return x


if __name__ == '__main__':
    model = FrequencyDiscriminator()

    x = torch.randn(4,176400)
    print(x.shape)

    y = model(x)
    print(y.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)