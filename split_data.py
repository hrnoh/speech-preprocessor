import argparse
import os
from shutil import move, copy
import warnings
import glob
import random
warnings.simplefilter(action='ignore', category=FutureWarning)


def split_data(root, train, test):
    all_spks = [p for p in glob.glob(os.path.join(root, '*')) if os.path.isdir(p)]
    for spk in all_spks:
        all_npz = glob.glob(os.path.join(spk, '*.npz'))

        total_num = len(all_npz)
        train_num = int(total_num * train)
        val_num = total_num - train_num - test

        os.makedirs(os.path.join(spk, 'train'), exist_ok=True)
        os.makedirs(os.path.join(spk, 'val'), exist_ok=True)
        os.makedirs(os.path.join(spk, 'test'), exist_ok=True)

        all_npz.sort()
        test_npz = all_npz[:test]

        rest_npz = all_npz[test:]
        random.shuffle(rest_npz)
        for idx, npz in enumerate(rest_npz):
            if idx < train_num:
                move(npz, os.path.join(spk, 'train', os.path.basename(npz)))
            else:
                move(npz, os.path.join(spk, 'val', os.path.basename(npz)))

        for idx, npz in enumerate(test_npz):
            move(npz, os.path.join(spk, 'test', os.path.basename(npz)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/hd0/korean_seq2seq_vc/preprocessed/ActorClassScandle/')
    parser.add_argument('--train', type=float, default=0.95)
    parser.add_argument('--test', type=int, default=10)
    args = parser.parse_args()

    split_data(args.root, args.train, args.test)