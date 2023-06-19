import torch 
import torch.nn as nn
import math

def spectrogram(y, n_fft, hop_size, win_size, center=False):
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, 
                window=torch.hann_window(win_size).to(y.device),
                center=center, pad_mode='reflect', normalized=False, 
                onesided=True, return_complex=False)

    spec = spec.pow(2).sum(-1)+(1e-9)

    return spec


class specLoss(nn.Module):
    def __init__(self, 
            loss_type = 'l1',
            win_len = [4096, 2048, 1024, 512, 256, 128, 64],
            hop_len = [2048, 1024, 512, 256, 128, 64, 32],
            fft_size = [8192, 4096, 2048, 1024, 512, 256, 128],
            lam_mag = 1.0, 
            lam_sc = 1.0):
        super(specLoss, self).__init__()
        '''
            loss_type: l1/l2, Whether to perform l1 or l2 loss
        '''
        assert len(win_len) == len(hop_len), f'window length:{len(win_len)} and hop length:{len(hop_len)} should be equal'
        assert len(win_len) == len(fft_size), f'window length:{len(win_len)} and fft size:{len(fft_size)} should be equal'
        
        if loss_type == 'l1':
            self.loss = torch.nn.L1Loss()
        if loss_type == 'l2':
            self.loss = torch.nn.MSELoss()

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_size = fft_size

        self.lam_mag = lam_mag
        self.lam_sc = lam_sc

    def __call__(self, output, target):
        '''
            output: ouput time domain waveform
            target: target time domain waveform
        '''
        loss_spec = 0.0
        for Wlen, Hlen, Fsize in zip(self.win_len, self.hop_len, self.fft_size):
            spec_output = spectrogram(output, Fsize, Wlen, Hlen)
            spec_target = spectrogram(target, Fsize, Wlen, Hlen)

            loss_mag = self.loss(torch.log(spec_output),torch.log(spec_target))
            
            ## check if this is required
            # loss_mag /= math.sqrt(Wlen)

            loss_sc = torch.norm(spec_output-spec_target, p='fro') / torch.norm(spec_target, p='fro')
            # loss_sc /= math.sqrt(Wlen)

            _loss_spec = self.lam_mag * loss_mag + self.lam_sc * loss_sc

            loss_spec += _loss_spec

        return loss_spec

class timeLoss(nn.Module):
    def __init__(self, 
            loss_type = 'l1',
            frame_len = [240, 480, 960],
            hop_len = [120, 240, 480],
            lam_seg = 1.0, 
            lam_energy = 1.0,
            lam_phase = 1.0):
        super(timeLoss, self).__init__()
        '''
            loss_type: l1/l2, Whether to perform l1 or l2 loss
        '''
        assert len(frame_len) == len(hop_len), f'No. of frames{len(frame_len)} and hop length:{len(hop_len)} should be equal'
        
        if loss_type == 'l1':
            self.loss = torch.nn.L1Loss()
        if loss_type == 'l2':
            self.loss = torch.nn.MSELoss()

        self.frame_len = frame_len
        self.hop_len = hop_len

        self.lam_seg = lam_seg
        self.lam_energy = lam_energy
        self.lam_phase = lam_phase

    def __call__(self, output, target):
        '''
            output: ouput time domain waveform
            target: target time domain waveform
        '''
        loss_time = 0.0
        sq_output = output**2
        sq_target = target**2
        diff_output = torch.diff(sq_output, n=1)
        diff_target = torch.diff(sq_target, n=1)

        loss_seg = self.loss(output, target)
        loss_energy = self.loss(sq_output, sq_target)
        loss_phase = self.loss(diff_output, diff_target)

        for fLen, hLen in zip(self.frame_len, self.hop_len):
            win_out = output.unfold(-1, fLen, hLen).mean(-1)
            win_tar = target.unfold(-1, fLen, hLen).mean(-1)
            loss_seg += self.loss(win_out, win_tar)

            win_sq_out = sq_output.unfold(-1, fLen, hLen).mean(-1)
            win_sq_tar = sq_target.unfold(-1, fLen, hLen).mean(-1)
            loss_energy += self.loss(win_sq_out, win_sq_tar)

            win_diff_out = torch.diff(win_sq_out, n=1)
            win_diff_tar = torch.diff(win_sq_tar, n=1)
            loss_phase += self.loss(win_diff_out, win_diff_tar)
        
        loss_time = self.lam_seg*loss_seg + self.lam_energy*loss_energy + self.lam_phase*loss_phase

        return loss_time