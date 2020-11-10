import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from librosa.filters import mel as librosa_mel_fn
from librosa.core import load
from hparams import hparams
import pyworld
from pysptk import sptk

class Audio2Mel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(hparams.win_size).float()
        mel_basis = librosa_mel_fn(
            hparams.sample_rate, hparams.fft_size, hparams.num_mels, hparams.fmin, hparams.fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = hparams.fft_size
        self.hop_length = hparams.hop_size
        self.win_length = hparams.win_size
        self.sampling_rate = hparams.sample_rate
        self.n_mel_channels = hparams.num_mels

    def forward(self, audio):
        # original padding
        p = (self.n_fft) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        # audio = audio.squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)

        if hparams.dynamic_range_compression:
            log_mel_spec = self.dynamic_range_compression(mel_output, eps=hparams.eps).squeeze()
        else:
            log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-2)).squeeze()

        return log_mel_spec

    def dynamic_range_compression(self, x, eps=0.0001):
        return torch.log(x + eps)

    def dynamic_range_decompression(self, x, eps=0.0001):
        return torch.exp(x) - eps


def trim_silence(wav, trim_top_db=hparams.trim_top_db, trim_fft_size=hparams.trim_fft_size, trim_hop_size=hparams.trim_hop_size):
	return librosa.effects.trim(wav, top_db=trim_top_db, frame_length=trim_fft_size, hop_length=trim_hop_size)[0]

def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size

def _stft(y, hparams):
    return librosa.stft(y=y, n_fft=hparams.fft_size, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

def pitch(wav, hparams, pitch_func="harvest"):
    frame_period = (hparams.hop_size / (0.001 * hparams.sample_rate))

    if isinstance(wav[0], np.float32):
        wav = wav.astype(np.double)

    if pitch_func == "harvest":
        f0, timeaxis = pyworld.harvest(wav, hparams.sample_rate, frame_period=frame_period)
    elif pitch_func == "dio":
        f0, timeaxis = pyworld.dio(wav, hparams.sample_rate, frame_period=frame_period)
    else:
        print("Invalid pitch function.")
        exit(-1)

    return np.nan_to_num(f0)

def energy(wav, hparams):
    #### energy
    D = _stft(wav, hparams)
    E = np.sum(np.abs(D) ** 2, axis=0) / len(D)

    return E

def audio_preprocess(wav, hparams):
    # rescaling
    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # trimming silence
    if hparams.trim_silence:
        wav = trim_silence(wav, trim_top_db=hparams.trim_top_db, trim_fft_size=hparams.trim_fft_size,
                           trim_hop_size=hparams.trim_hop_size)
    return wav



def load_audio(audio_path, hparams):
    data = load(audio_path, sr=hparams.sample_rate)[0]
    data = audio_preprocess(data, hparams)
    return data # Mel 생성 전에 reflect padding 하기 위한 밑작업