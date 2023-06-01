import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from voicefixer.vocoder.config import Config



class UpsampleNet(nn.Module):
    def __init__(self, input_size, output_size, upsample_factor, hp=None, index=0):

        super(UpsampleNet, self).__init__()
        self.up_type = Config.up_type
        self.use_smooth = Config.use_smooth
        self.use_drop = Config.use_drop
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor
        self.skip_conv = nn.Conv1d(input_size, output_size, kernel_size=1)
        self.index = index
        if self.use_smooth:
            window_lens = [5, 5, 4, 3]
            self.window_len = window_lens[index]

        if self.up_type != "pn" or self.index < 3:
            # if self.up_type != "pn":
            layer = nn.ConvTranspose1d(
                input_size,
                output_size,
                upsample_factor * 2,
                upsample_factor,
                padding=upsample_factor // 2 + upsample_factor % 2,
                output_padding=upsample_factor % 2,
            )
            self.layer = nn.utils.weight_norm(layer)
        else:
            self.layer = nn.Sequential(
                nn.ReflectionPad1d(1),
                nn.utils.weight_norm(
                    nn.Conv1d(input_size, output_size * upsample_factor, kernel_size=3)
                ),
                nn.LeakyReLU(),
                nn.ReflectionPad1d(1),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        output_size * upsample_factor,
                        output_size * upsample_factor,
                        kernel_size=3,
                    )
                ),
                nn.LeakyReLU(),
                nn.ReflectionPad1d(1),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        output_size * upsample_factor,
                        output_size * upsample_factor,
                        kernel_size=3,
                    )
                ),
                nn.LeakyReLU(),
            )

        if hp is not None:
            self.org = Config.up_org
            self.no_skip = Config.no_skip
        else:
            self.org = False
            self.no_skip = True

        if self.use_smooth:
            self.mas = nn.Sequential(
                # LowpassBlur(output_size, self.window_len),
                MovingAverageSmooth(output_size, self.window_len),
                # MovingAverageSmooth(output_size, self.window_len),
            )

    def forward(self, inputs):

        if not self.org:
            inputs = inputs + torch.sin(inputs)
            B, C, T = inputs.size()
            res = inputs.repeat(1, self.upsample_factor, 1).view(B, C, -1)
            skip = self.skip_conv(res)
            if self.up_type == "repeat":
                return skip

        outputs = self.layer(inputs)
        if self.up_type == "pn" and self.index > 2:
            B, c, l = outputs.size()
            outputs = outputs.view(B, -1, l * self.upsample_factor)

        if self.no_skip:
            return outputs

        if not self.org:
            outputs = outputs + skip

        if self.use_smooth:
            outputs = self.mas(outputs)

        if self.use_drop:
            outputs = F.dropout(outputs, p=0.05)

        return outputs


class ResStack(nn.Module):
    def __init__(self, channel, kernel_size=3, resstack_depth=4, hp=None):
        super(ResStack, self).__init__()

        self.use_wn = Config.use_wn
        self.use_shift_scale = Config.use_shift_scale
        self.channel = channel

        def get_padding(kernel_size, dilation=1):
            return int((kernel_size * dilation - dilation) / 2)

        if self.use_shift_scale:
            self.scale_conv = nn.utils.weight_norm(
                nn.Conv1d(
                    channel, 2 * channel, kernel_size=kernel_size, dilation=1, padding=1
                )
            )

        if not self.use_wn:
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LeakyReLU(),
                        nn.utils.weight_norm(
                            nn.Conv1d(
                                channel,
                                channel,
                                kernel_size=kernel_size,
                                dilation=3 ** (i % 10),
                                padding=get_padding(kernel_size, 3 ** (i % 10)),
                            )
                        ),
                        nn.LeakyReLU(),
                        nn.utils.weight_norm(
                            nn.Conv1d(
                                channel,
                                channel,
                                kernel_size=kernel_size,
                                dilation=1,
                                padding=get_padding(kernel_size, 1),
                            )
                        ),
                    )
                    for i in range(resstack_depth)
                ]
            )
        else:
            self.wn = WaveNet(
                in_channels=channel,
                out_channels=channel,
                cin_channels=-1,
                num_layers=resstack_depth,
                residual_channels=channel,
                gate_channels=channel,
                skip_channels=channel,
                # kernel_size=5,
                # dilation_rate=3,
                causal=False,
                use_downup=False,
            )

    def forward(self, x):
        if not self.use_wn:
            for layer in self.layers:
                x = x + layer(x)
        else:
            x = self.wn(x)

        if self.use_shift_scale:
            m_s = self.scale_conv(x)
            m_s = m_s[:, :, :-1]

            m, s = torch.split(m_s, self.channel, dim=1)
            s = F.softplus(s)

            x = m + s * x[:, :, 1:]  # key!!!
            x = F.pad(x, pad=(1, 0), mode="constant", value=0)

        return x

class WaveNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_layers=10,
        residual_channels=64,
        gate_channels=64,
        skip_channels=64,
        kernel_size=3,
        dilation_rate=2,
        cin_channels=80,
        hp=None,
        causal=False,
        use_downup=False,
    ):
        super(WaveNet, self).__init__()

        self.in_channels = in_channels
        self.causal = causal
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cin_channels = cin_channels
        self.kernel_size = kernel_size
        self.use_downup = use_downup

        self.front_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.residual_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        if self.use_downup:
            self.downup_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.residual_channels,
                    out_channels=self.residual_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=self.residual_channels,
                    out_channels=self.residual_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                UpsampleNet(self.residual_channels, self.residual_channels, 4, hp),
            )

        self.res_blocks = nn.ModuleList()
        for n in range(self.num_layers):
            self.res_blocks.append(
                ResBlock(
                    self.residual_channels,
                    self.gate_channels,
                    self.skip_channels,
                    self.kernel_size,
                    dilation=dilation_rate**n,
                    cin_channels=self.cin_channels,
                    local_conditioning=(self.cin_channels > 0),
                    causal=self.causal,
                    mode="SAME",
                )
            )
        self.final_conv = nn.Sequential(
            nn.ReLU(),
            Conv(self.skip_channels, self.skip_channels, 1, causal=self.causal),
            nn.ReLU(),
            Conv(self.skip_channels, self.out_channels, 1, causal=self.causal),
        )

    def forward(self, x, c=None):
        return self.wavenet(x, c)

    def wavenet(self, tensor, c=None):

        h = self.front_conv(tensor)
        if self.use_downup:
            h = self.downup_conv(h)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            h, s = f(h, c)
            skip += s
        out = self.final_conv(skip)
        return out

    def receptive_field_size(self):
        num_dir = 1 if self.causal else 2
        dilations = [2 ** (i % self.num_layers) for i in range(self.num_layers)]
        return (
            num_dir * (self.kernel_size - 1) * sum(dilations)
            + 1
            + (self.front_channels - 1)
        )

    def remove_weight_norm(self):
        for f in self.res_blocks:
            f.remove_weight_norm()
