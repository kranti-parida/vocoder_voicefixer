import torch
import os

from dataset import wav_mel_dataset, get_dataset
from config import Config

class Vocoder(nn.Module):
    def __init__(self, sample_rate):
        super(Vocoder, self).__init__()
        Config.refresh(sample_rate)
        self.rate = sample_rate
        if(not os.path.exists(Config.ckpt)):
            raise RuntimeError("Model not found. Please download the model")
        self._load_pretrain(Config.ckpt)
        self.weight_torch = Config.get_mel_weight_torch(percent=1.0)[None, None, None, ...]

    def _load_pretrain(self, pth):
        self.model = Generator(Config.cin_channels)
        checkpoint = load_checkpoint(pth, torch.device("cpu"))
        load_try(checkpoint["generator"], self.model)
        
    def forward(self, mel, cuda=False):
        """
        :param non normalized mel spectrogram: [batchsize, 1, t-steps, n_mel]
        :return: [batchsize, 1, samples]
        """
        assert mel.size()[-1] == 128
        check_cuda_availability(cuda=cuda)
        self.model = try_tensor_cuda(self.model, cuda=cuda)
        mel = try_tensor_cuda(mel, cuda=cuda)
        self.weight_torch = self.weight_torch.type_as(mel)
        # mel = mel / self.weight_torch
        mel = tr_normalize(tr_amp_to_db(torch.abs(mel)) - 20.0)
        mel = tr_pre(mel[:, 0, ...])
        wav_re = self.model(mel)
        return wav_re


def train(config):
    torch.cuda.manual_seed(h.seed)
    device = torch.device("cuda")

    # load model
    model = Vocoder(config.sample_rate)

    ## load pre-trained model here
    
    if os.path.isdir(config.checkpoint_path):
        cp_voc = scan_checkpoint(config.checkpoint_path, 'ttu_')

    # load last checkpoint if exists
    steps = 0
    if cp_voc is None:
        state_dict = None
        last_epoch = -1
    else:
        state_dict = load_checkpoint(cp_voc, device)
        print(f"Loading state dict from {cp_voc}...")
        translator.load_state_dict(state_dict['model'])
        try:
            steps = state_dict['steps'] + 1
            last_epoch = state_dict['epoch']
        except:
            steps = int(input(f"Enter steps for {cp_voc}"))
            last_epoch = int(input(f"Enter epochs for {cp_voc}"))
    
    # optimizer

    optim = torch.optim.Adam(model.parameters(), config.learning_rate, betas=[config.adam_b1, config.adam_b2])

    if state_dict is not None:
        try:
            optim.load_state_dict(state_dict['optim'])
        except:
            print("Warning: Did not find optim state in checkpoint!")
            pass


