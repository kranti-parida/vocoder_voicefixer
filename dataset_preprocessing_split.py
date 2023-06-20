## Split the long sequence into short sequences of 10s
## remove the first 10% and last 5% of the audio as it usually contains introductory music


import os
import librosa
from tqdm import tqdm
import soundfile as sf

def split_audio(audio_path, duration=10, sample_rate=44100, output_path_root='./data/audio_processed_44.1k_128kbps_10s_split'):
    wav, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    ## ignoring first 10% and last 5% of audio as it is usually contains introductory/end music
    start = int(0.1*wav.shape[0])
    end = wav.shape[0]-int(0.05*wav.shape[0])

    start_time_in_sec = start/sample_rate

    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)

    ## split the audio into short sequences of 10s
    for i in range(start, end, duration*sample_rate):
        if i+duration*sample_rate > end:
            break
        split_wav = wav[i:i+duration*sample_rate]
        aud_name = audio_path.split('/')[-1][:-4]
        output_path = f'{aud_name}_{start_time_in_sec:07.2f}.wav'
        sf.write(os.path.join(output_path_root, output_path), split_wav, sample_rate)
        start_time_in_sec += duration



src_path = './data/audio_processed_44.1k_128kbps'
for file in tqdm(os.listdir(src_path)):
    if file.endswith('.wav') and not file.startswith('.'):
        audio_path = os.path.join(src_path, file)
        split_audio(audio_path)
    
