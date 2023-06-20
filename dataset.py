import torch
import os
import librosa
import numpy as np

def get_dataset(path, mode):
    with open(os.path.join(path, mode+'.txt'), 'r') as f:
        audios = f.read().splitlines()
    
    all_audios_path = os.listdir(os.path.join(path, 'audio_processed_44.1k_128kbps_10s_split'))
    audios_path = []
    for audio in all_audios_path:
        audio_id = audio.split('/')[-1][:-12]
        if audio_id+'.wav' in audios:
            audios_path.append(os.path.join(path, 'audio_processed_44.1k_128kbps_10s_split', audio))

    return audios_path

class wav_mel_dataset(torch.utils.data.Dataset):
    def __init__(self,config, file_list):
        self.audio_files = file_list
        self.config = config
    
    def __getitem__(self, index):
        audio_path = self.audio_files[index]
        wav, _ = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        nsamples = self.config.sample_rate * self.config.duration
        
        # ## ignoring first 10% and last 5% of audio as it is usually contains introductory music
        # init = int(0.1*wav.shape[0])
        # final = wav.shape[0]-int(0.05*wav.shape[0])
        # start = np.random.randint(init, final - nsamples)

        # # If audio is too short (~ less than 9 minutes) ignore that
        # if (start + nsamples > wav.shape[0]) or (wav.shape[0] < 500*44100):
        #     return self.__getitem__(index)


        start = np.random.randint(0, wav.shape[0] - nsamples)
        wav = wav[start : start + nsamples]
        wav = wav / np.max(np.abs(wav))

        return {
                'wav': wav,
                'path': audio_path,
        }

    def __len__(self):
        return len(self.audio_files)

if __name__ == '__main__':
    config_path = './configs/vocoder.json'
    mode = 'val'
    from utils.utils import load_config
    config = load_config(config_path)
    file_list = get_dataset(config.data_path, mode)
    print(len(file_list))
    import pdb; pdb.set_trace()
    datast = wav_mel_dataset(config, file_list)
    print(len(datast))
    print(datast[0]['wav'].shape)
    print(datast[0]['path'])