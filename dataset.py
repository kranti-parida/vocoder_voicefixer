import torch
import os
import librosa
import numpy as np

def get_dataset(path, mode):
    with open(os.path.join(path, mode+'.txt'), 'r') as f:
        audios = f.read().splitlines()
    
    audios_path = [os.path.join(path, 'audio_processed_44.1k_128kbps', audio) for audio in audios]

    return audios_path

class wav_mel_dataset(torch.utils.data.Dataset):
    def __init__(self,config, file_list):
        self.audio_files = file_list
        self.config = config
    
    def __getitem__(self, index):
        audio_path = self.audio_files[index]
        wav, _ = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        nsamples = self.config.sample_rate * self.config.duration
        
        ## ignoring first 10% and last 5% of audio as it is usually contains introductory music
        start = np.random.randint(int(0.1*wav.shape[0]), (wav.shape[0]-int(0.05*wav.shape[0])) - nsamples)

        # If audio is too short (less than 10 minutes) ignore that
        if (start + nsamples > wav.shape[0]) or (wav.shape[0] < 500*44100):
            return self.__getitem__(index)
        wav = wav[start : start + nsamples]
        wav = wav / np.max(np.abs(wav))

        return {
                'wav': wav,
                'path': audio_path,
        }

    def __len__(self):
        return len(self.audio_files)