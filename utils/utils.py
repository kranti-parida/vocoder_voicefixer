import os
import torch
import glob
import shutil
import json

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_mel_pytorch(wav, NFFT, HOP_LEN, WIN_SIZE, WIN, MEL_BASIS):
    stft_torch = torch.stft(wav, n_fft=NFFT, hop_length=HOP_LEN, 
                    win_length=WIN_SIZE, window=WIN, center=True, return_complex=True)
    
    stft_torch = torch.abs(stft_torch)
    mel_torch = MEL_BASIS @ stft_torch
    mel_torch = mel_torch.unsqueeze(1).permute(0, 1, 3, 2)

    return mel_torch


def load_config(filepath):
    with open(filepath) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h