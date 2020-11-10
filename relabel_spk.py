import argparse

def relabeling(in_dir):
    pass

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='/hd0/dataset/korean_all/')
    args = parser.parse_args()