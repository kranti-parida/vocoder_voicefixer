## This script is to check the length of different examplses in the dataset

import librosa
import os

path = './data'
mode = 'train'

with open(os.path.join(path, mode+'.txt'), 'r') as f:
        audios = f.read().splitlines()
    
audios_path = [os.path.join(path, 'audio_processed_44.1k_128kbps', audio) for audio in audios]

lengths = {}
for audio_path in audios_path:
    wav, _ = librosa.load(audio_path, sr=44100, mono=True)
    lengths[audio_path] = wav.shape[0]/44100
    print(wav.shape[0]/44100)

sorted_length = sorted(lengths.items(), key=lambda x: x[1])

