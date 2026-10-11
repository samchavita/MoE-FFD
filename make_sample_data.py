"""
Generates a tiny synthetic dataset matching the FaceForensics++ layout in the README,
so the data pipeline (dataset.py) can be smoke-tested without the real data.

Usage: python make_sample_data.py --out sample_data --videos 2 --frames 25
"""
import os
from os.path import join
import argparse
import numpy as np
import cv2


METHODS = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NueralTextures']
PHASES = ['train', 'valid', 'test']


def make_video_frames(video_path, num_frames, size, rng):
    os.makedirs(video_path, exist_ok=True)
    for i in range(num_frames):
        img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        # Frame names must be numeric: dataset.py sorts by int(name[:-4])
        cv2.imwrite(join(video_path, '{:04d}.png'.format(i)), img)


def main(args):
    rng = np.random.default_rng(2024)
    root = join(args.out, 'FaceForensics++', args.compression)
    for phase in PHASES:
        for method in METHODS:
            for v in range(args.videos):
                make_video_frames(join(root, phase, method, '{:03d}'.format(v)),
                                  args.frames, args.size, rng)
    print('Sample data written to {}'.format(root))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='sample_data')
    parser.add_argument('--compression', type=str, default='c23')
    parser.add_argument('--videos', type=int, default=2, help="videos per method")
    parser.add_argument('--frames', type=int, default=25, help="frames per video")
    parser.add_argument('--size', type=int, default=256, help="frame width/height")
    main(parser.parse_args())
