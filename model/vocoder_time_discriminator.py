import torch
import torch.nn as nn


class timeDiscriminator(nn.Module):
    def __init__(self, ndf = 128, n_layers = 3, downsampling_factor = 4, disc_out = 512):
        super(timeDiscriminator, self).__init__()
        discriminator = nn.ModuleDict()
        discriminator["layer_0"] = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(1, ndf, kernel_size=15, stride=1)),
            nn.LeakyReLU(0.2, True),
        )

        stride = downsampling_factor
        groups = stride
        for n in range(1, n_layers + 1):
            groups = 2*groups
            discriminator["layer_%d" % n] = nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(
                    ndf,
                    ndf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=groups,
                )),
                nn.LeakyReLU(0.2, True),
            )

        discriminator["layer_%d" % (n_layers + 1)] = nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(ndf, 1, kernel_size=3, stride=1, padding=1)),
                nn.LeakyReLU(0.2, True),
        )
        self.discriminator = discriminator

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for key, module in self.discriminator.items():
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]


if __name__ == '__main__':
    model = timeDiscriminator()

    x = torch.randn(3, 1, 66048)
    print(x.shape)

    features, score = model(x)
    import pdb; pdb.set_trace()
    print("Length of features : ", len(features))
    print("Length of score : ", len(score))
    for feat in features:
        print(feat.shape)
    print(score.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)