import torch
import os
import librosa
import numpy as np
import lmdb
import pydub
import io

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
    def __init__(self,config, file_list, mode='train'):
        self.audio_files = file_list
        self.mode = mode
        self.root_dir = '/'.join(file_list[0].split('/')[:-2])
        self.dataset_type = file_list[0].split('/')[-2]
        self.sr = config.sample_rate
        self.duration = config.duration
        self.use_lmdb = config.use_lmdb

        if self.use_lmdb:
            if self.root_dir is not None:
                lmdbfile = os.path.join(self.root_dir, f'{self.dataset_type}_{self.mode}_sr_{self.sr}.lmdb')
                if not os.path.exists(lmdbfile):
                    print(f"Preparing LMDB...")
                    self.write_lmdb(lmdbfile)

                self.db = lmdb.open(lmdbfile, readonly=True, lock=False, readahead=False, meminit=False)
    
    def _read_audio_resample(self, audiofile):
        try:
            inp = pydub.AudioSegment.from_file(audiofile)
            sample_width, channels, rate = inp.sample_width, inp.channels, inp.frame_rate

            if (channels == 2 or rate != self.sr):
                audio = np.array(inp.get_array_of_samples())
                audio= np.float32(audio) / 2**((sample_width*8)-1)
            
                ## for stereo audio, convert to mono
                if channels == 2:
                    audio = audio.reshape((-1, 2))
                    audio = (audio[:,0] + audio[:,1])/2
                    channels = 1

                ## resample to specified rate
                if rate != self.sr:
                    audio = librosa.resample(audio, orig_sr = rate, target_sr = self.sr)
                    rate = self.sr
                
                ##re-encode tha audio
                output_sig = self.convert_float_to_bytes(audio, Nbytes=sample_width)
                inp = pydub.AudioSegment(output_sig.tobytes(), 
                                frame_rate=rate, 
                                sample_width=sample_width, channels=channels)
            file_handle = inp.export(format='wav')
            fl_bytes = file_handle.read()

            return fl_bytes

        except:
            print(f"Error reading audio file {audiofile}")
            return None
        
    def convert_float_to_bytes(self, data, Nbytes):
        '''
        data: np.array of float data type
        Nbytes: no. of bytes for encoding integer data
        '''
        d_type = {}
        max_val = 2**((Nbytes*8)-1)
        d_type = f'np.int{Nbytes*8}'

        data = (data*max_val).astype(eval(d_type))

        return data

    def write_lmdb(self, lmdbfile):
        '''
        Write the LMDB file corresponding to this dataset
        '''
        db = lmdb.open(lmdbfile, map_size=500 * 1e9, readonly=False, meminit=False, map_async=True)
        write_frequency = 100
        print('***** Starting to write LMDB ' + lmdbfile)
        print(f"Num: {len(self.audio_files)}")
        
        max_idx = len(self.audio_files)

        txn = db.begin(write=True)
        all_audiofiles = []
        for idx, audiofile in  enumerate(self.audio_files):
            ## read and save audio here
            fl_bytes = self._read_audio_resample(audiofile)
            if fl_bytes is None:
                continue
            key_str = u'audio_{}'.format(os.path.basename(audiofile)).encode('ascii') 
            if not txn.get(key_str):
                txn.put(key_str, fl_bytes)

            if idx>0 and idx % write_frequency == 0:
                print("Writing LMDB txn to disk [%d/%d]" % (idx, max_idx))
                txn.commit()
                txn = db.begin(write=True)

        # finish remaining transactions 
        print("Writing final LMDB txn to disk [%d/%d]" % (max_idx, max_idx))
        txn.commit()

        print("Flushing LMDB database")
        db.sync()
        db.close()
        print("**** LMDB writing complete")

    def _cvtBytesToAudio(self, bytes):
        bytes = io.BytesIO(bytes)
        audio = pydub.AudioSegment.from_file(bytes)
        sample_width, channels, rate = audio.sample_width, audio.channels, audio.frame_rate
        if channels != 1:
            print(f"Channel is not 1, rather no. of channels is: {channels}")
            return None
        if rate != self.sr:
            print(f"Sampling rate is not as desire {self.sr}, rather it has a value {rate}")
            return None
        audio = np.array(audio.get_array_of_samples())
        audio= np.float32(audio) / 2**((sample_width*8)-1)
        
        return audio

    def __getitem__(self, index):
        audio_path = self.audio_files[index]
        
        if self.use_lmdb:
            with self.db.begin(write=False) as txn:
                wav = self._cvtBytesToAudio(txn.get(u'audio_{}'.format(audio_path).encode('ascii')))
                if audio is None:
                    print(f"Error reading audio file {audio_path}")
                    return None
        else:
            wav, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        
        nsamples = self.sr * self.duration
        
        # ## ignoring first 10% and last 5% of audio as it is usually contains introductory music
        # init = int(0.1*wav.shape[0])
        # final = wav.shape[0]-int(0.05*wav.shape[0])
        # start = np.random.randint(init, final - nsamples)

        # # If audio is too short (~ less than 9 minutes) ignore that
        # if (start + nsamples > wav.shape[0]) or (wav.shape[0] < 500*44100):
        #     return self.__getitem__(index)


        if self.mode == 'val':
            start = 0
        else:
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