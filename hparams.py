# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(
    cleaners = ['korean_cleaners'],  # 'korean_cleaners'   or 'english_cleaners'
    use_phoneme = True,

    # Audio
    sample_rate = 24000,  # original : 22050
    
    # shift can be specified by either hop_size or frame_shift_ms
    hop_size = 300,    # frame_shift_ms = 12.5ms, original = 256
    fft_size = 2048,   # original = 1024
    win_size = 1200,   # 50ms, original = 1024
    num_mels = 128,     # original = 1024

    #Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
    preemphasize = False, #whether to apply filter
    preemphasis = 0.97,

    # min_level_db = -100,
    # ref_level_db = 20,
    # signal_normalization = True, #Whether to normalize mel spectrograms to some predefined range (following below parameters)
    # allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
    # symmetric_mels = True, #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    # max_abs_value = 1., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, not too small for fast convergence)

    eps = 0.0001, # for dynamic range compression
        
    rescaling=True,
    rescaling_max=0.999, 
    
    trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    #M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
    trim_fft_size = 1024,
    trim_hop_size = 256,
    trim_top_db = 20,
    
    
    # mel-basis parameters
    fmin = 20,      # original = 55
    fmax = 12000,    # original = 7600
 
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
