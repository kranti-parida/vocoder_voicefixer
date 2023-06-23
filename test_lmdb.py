import lmdb
import io
import pydub
import numpy as np
import time
import soundfile as sf

lmdb_path = './data/audio_processed_44.1k_128kbps_10s_split_val_sr_44100.lmdb'

db = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

def _cvtBytesToAudio(bytes):
    sr = 44100
    bytes = io.BytesIO(bytes)
    audio = pydub.AudioSegment.from_file(bytes)
    sample_width, channels, rate = audio.sample_width, audio.channels, audio.frame_rate
    if channels != 1:
        print(f"Channel is not 1, rather no. of channels is: {channels}")
        return None
    if rate != sr:
        print(f"Sampling rate is not as desire {sr}, rather it has a value {rate}")
        return None
    audio = np.array(audio.get_array_of_samples())
    audio= np.float32(audio) / 2**((sample_width*8)-1)
    
    return audio

def _cvtBytesToAudio(bytes):
    sr = 44100
    bytes = io.BytesIO(bytes)
    audio = pydub.AudioSegment.from_file(bytes)
    sample_width, channels, rate = audio.sample_width, audio.channels, audio.frame_rate
    if channels != 1:
        print(f"Channel is not 1, rather no. of channels is: {channels}")
        return None
    if rate != sr:
        print(f"Sampling rate is not as desire {sr}, rather it has a value {rate}")
        return None
    audio = np.array(audio.get_array_of_samples())
    audio= np.float32(audio) / 2**((sample_width*8)-1)
    
    return audio

def _cvtBytesToAudioSF(bytes):
    sr = 44100
    bytes = io.BytesIO(bytes)
    audio, rate = sf.read(bytes)
    if rate != sr:
        print(f"Sampling rate is not as desire {sr}, rather it has a value {rate}")
        return None    
    return audio



with db.begin() as txn:
    for key, value in txn.cursor():
        # print(key)
        # print(value)
        st = time.time()
        audio_sf = _cvtBytesToAudioSF(value)
        print(f"Time taken for SF: {time.time()-st}")
        st = time.time()
        audio = _cvtBytesToAudio(value)
        print(f"Time taken for Pydub: {time.time()-st}")

        # import pdb; pdb.set_trace()