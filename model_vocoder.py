from model.generator import Generator
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
        self.model.eval()
        self.model.remove_weight_norm()
        self.model.remove_weight_norm()
        for p in self.model.parameters():
            p.requires_grad = False

    # def vocoder_mel_npy(self, mel, save_dir, sample_rate, gain):
    #     mel = mel / Config.get_mel_weight(percent=gain)[...,None]
    #     mel = normalize(amp_to_db(np.abs(mel)) - 20)
    #     mel = pre(np.transpose(mel, (1, 0)))
    #     with torch.no_grad():
    #         wav_re = self.model(mel) # torch.Size([1, 1, 104076])
    #         save_wave(tensor2numpy(wav_re)*2**15,save_dir,sample_rate=sample_rate)

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