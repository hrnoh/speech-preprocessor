import argparse
import glob
import os
import numpy as np
import tqdm

def relabeling(in_dir):
    all_dataset = [p for p in glob.glob(os.path.join(in_dir, '*')) if os.path.isdir(p)]
    label = 0
    print(all_dataset)
    for dataset in all_dataset:
        all_spk_path = [p for p in glob.glob(os.path.join(dataset, '*')) if os.path.isdir(p)]
        for spk_path in all_spk_path:
            spk_name = os.path.basename(spk_path)
            all_npz_path = glob.glob(os.path.join(spk_path, '*.npz'))

            print('relabeling {} : {}'.format(spk_name, label))
            for npz_path in tqdm.tqdm(all_npz_path):
                npz = np.load(npz_path)
                npz_dict = dict(npz)
                npz_dict['spk_label'] = label
                np.savez(npz_path, **npz_dict)
            label += 1

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='/hd0/speech-preprocessor/')
    args = parser.parse_args()

    relabeling(args.in_dir)