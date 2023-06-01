import wave
import os
import numpy as np
import soundfile as sf
import librosa


def save_wave(frames: np.ndarray, fname, sample_rate=44100):
    shape = list(frames.shape)
    if len(shape) == 1:
        frames = frames[..., None]
    in_samples, in_channels = shape[-2], shape[-1]
    if in_channels >= 3:
        if len(shape) == 2:
            frames = np.transpose(frames, (1, 0))
        elif len(shape) == 3:
            frames = np.transpose(frames, (0, 2, 1))
        msg = (
            "Warning: Save audio with "
            + str(in_channels)
            + " channels, save permute audio with shape "
            + str(list(frames.shape))
            + " please check if it's correct."
        )
        # print(msg)
    if (
        np.max(frames) <= 1
        and frames.dtype == np.float32
        or frames.dtype == np.float16
        or frames.dtype == np.float64
    ):
        frames *= 2**15
    frames = frames.astype(np.short)
    if len(frames.shape) >= 3:
        frames = frames[0, ...]
    sf.write(fname, frames, samplerate=sample_rate)


def get_duration(fname):
    with wave.open(fname) as f:
        params = f.getparams()
    return params[3] / params[2]


def read_wave(
    fname,
    sample_rate,
    portion_start=0,
    portion_end=1,
):  # Whether you want raw bytes
    """
    :param fname: wav file path
    :param sample_rate:
    :param portion_start:
    :param portion_end:
    :return: [sample, channels]
    """
    # sr = get_sample_rate(fname)
    # if(sr != sample_rate):
    #     print("Warning: Sample rate not match, may lead to unexpected behavior.")
    if portion_end > 1 and portion_end < 1.1:
        portion_end = 1
    if portion_end != 1:
        duration = get_duration(fname)
        wav, _ = librosa.load(
            fname,
            sr=sample_rate,
            offset=portion_start * duration,
            duration=(portion_end - portion_start) * duration,
            mono=False,
        )
    else:
        wav, _ = librosa.load(fname, sr=sample_rate, mono=False)
    if len(list(wav.shape)) == 1:
        wav = wav[..., None]
    else:
        wav = np.transpose(wav, (1, 0))
    return wav
