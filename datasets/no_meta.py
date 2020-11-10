import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from text import text_to_sequence
import pkbar
import glob
import torch
import random

from preprocess_utils import Audio2Mel, load_audio, audio_preprocess

def build_from_path(hparams, in_dir, out_dir, num_workers=16):

    executor = ProcessPoolExecutor(max_workers=num_workers)

    #spk_paths = [p for p in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(p) and os.path.basename(p) in hparams.used_spks]
    #files_path = glob.glob(os.path.join(data_root, '**/*.wav')) # 경로 부분은 알아서 자기 상황에 맞춰서 바꾸어야 함
    with open(os.path.join(in_dir, "metadata.csv"), 'rt') as f:
        all_meta = [row.rstrip() for row in f.readlines()]

    mel_gen = Audio2Mel(hparams)

    os.makedirs(out_dir, exist_ok=True)

    pbar = pkbar.Pbar(name='loading and processing dataset', target=len(all_meta))
    futures = []
    for i, row in enumerate(all_meta):
        wav_path, text, spk_label, spk_name = row.split('|')
        full_path = os.path.join(in_dir, wav_path)

        # make speaker directory
        os.makedirs(os.path.join(out_dir, spk_name), exist_ok=True)


        wav_name = os.path.basename(wav_path)
        npz_name = os.path.join(out_dir, spk_name, wav_name[:-4] + ".npz")

        futures.append(executor.submit(partial(_processing_data, hparams, full_path, text, spk_label, npz_name, mel_gen, pbar, i)))

    results = [future.result() for future in futures if future.result() is not None]

    all_mel = np.concatenate(results, axis=1)
    mel_mean = np.mean(all_mel, axis=1)
    mel_std = np.std(all_mel, axis=1)

    np.save(os.path.join(out_dir, "mel_mean_std.npy"),
            [mel_mean, mel_std])

    print('Finish Preprocessing')


def _processing_data(hparams, full_path, text, spk_label, npz_name, mel_gen, pbar, i):

    if os.path.isfile(full_path):
        wav = load_audio(full_path, hparams)
    else:
        print(full_path, "not exists.")
        return None

    mel_gen_input = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
    mel = mel_gen(mel_gen_input).numpy()

    data = {}
    data['audio'] = wav
    data['tokens'] = text_to_sequence(text)
    data['mel'] = mel.T
    data['spk_label'] = spk_label

    np.savez(npz_name, **data)
    pbar.update(i)

    return mel

