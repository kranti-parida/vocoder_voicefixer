import torch
import torch.nn as nn
from torchsubband import SubbandDSP


# from models.vocoder_freq_discriminator import FrequencyDiscriminator
try:
    from models.vocoder_time_discriminator import timeDiscriminator
except:
    from vocoder_time_discriminator import timeDiscriminator
from torchsubband import SubbandDSP

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_D = 4, ndf = 128, n_layers = 3, 
                downsampling_factor = 4, disc_out = 1):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = timeDiscriminator(
                ndf, n_layers, downsampling_factor, disc_out
            )

        # downsample by 2
        self.downsample = nn.AvgPool1d(2, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)


    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results


class MultiBandDiscriminator(nn.Module):
    def __init__(self, ndf = 128, n_layers = 3, 
                downsampling_factor = 4, disc_out = 1,
                num_subband=4):
        super().__init__()
        self.num_subband = num_subband
        self.subband = SubbandDSP(subband=num_subband)
        ## freeze the parameter for subband decomposition as it is implemented as a filter
        for param in self.subband.parameters():
            param.requires_grad = False

        self.model = nn.ModuleDict()
        for i in range(num_subband):
            self.model[f"disc_{i}"] = timeDiscriminator(
                ndf, n_layers, downsampling_factor, disc_out
            )

        self.apply(weights_init)

    def forward(self, x):
        import pdb; pdb.set_trace()
        x = self.subband.wav_to_sub(x)
        results = []
        for i in range(self.num_subband):
            results.append(self.model[f"disc_{i}"](x[:,i,:].unsqueeze(1)))

        return results

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiscale_discriminator = MultiScaleDiscriminator()
        self.multiband_discriminator = MultiBandDiscriminator()
        # self.freq_discriminator = FrequencyDiscriminator()

    def forward(self, x):
        res_multi_scale = self.multiscale_discriminator(x)
        res_multi_band = self.multiband_discriminator(x)
        # res_freq = self.freq_discriminator(x)

        # return res_multi_scale, res_multi_band, res_freq
        return res_multi_scale, res_multi_band


if __name__ == '__main__':
    model = Discriminator()
    '''
    Length of features :  5
    Length of score :  3
    torch.Size([3, 16, 25600])
    torch.Size([3, 64, 6400])
    torch.Size([3, 256, 1600])
    torch.Size([3, 512, 400])
    torch.Size([3, 512, 400])
    torch.Size([3, 1, 400]) -> score
    '''

    x = torch.randn(3, 1, 3*22050)
    print(x.shape)
    mScale, mBand = model(x)
    import pdb; pdb.set_trace()