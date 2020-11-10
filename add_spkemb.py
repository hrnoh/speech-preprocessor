import argparse
import os
from shutil import move, copy
import warnings
import glob
import random
import numpy as np
import pickle
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import pkbar
from functools import partial

warnings.simplefilter(action='ignore', category=FutureWarning)

def build_from_path(root, spk_emb_path):
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    with open(spk_emb_path, 'rb') as f:
        spk_meta = pickle.load(f)

    all_spk_path = [p for p in glob.glob(os.path.join(root, '*')) if os.path.isdir(p)]
    for spk_path in all_spk_path:
        spk_name = os.path.basename(spk_path)
        spk_emb = None
        for meta in spk_meta:
            if meta[0] == spk_name:
                spk_emb = meta[1]
                break

        # speaker embedding이 없으면 continue
        if spk_emb is None:
            continue

        all_npz_path = glob.glob(os.path.join(spk_path, '*.npz'))
        pbar = pkbar.Pbar(name='loading and processing dataset', target=len(all_npz_path))

        futures = []
        for i, npz_path in enumerate(all_npz_path):
            futures.append(executor.submit(partial(_add_spk_emb, npz_path, spk_emb, pbar, i)))

def _add_spk_emb(npz_path, spk_emb, pbar, i):
    npz = np.load(npz_path)
    npz_dict = dict(npz)
    npz_dict['spk_emb'] = spk_emb
    np.savez(npz_path, **npz_dict)

    pbar.update(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/hd0/korean_seq2seq_vc/preprocessed/ActorClassScandle/')
    parser.add_argument('--spk_emb_path', type=str, default='/sd0/git/speaker_verification')
    args = parser.parse_args()

    build_from_path(args.root, args.spk_emb_path)