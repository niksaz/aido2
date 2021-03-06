# Author: Mikita Sazanovich

import argparse
import subprocess
import os
import pathlib
import numpy as np


def run_cmd(cmd):
    ps = subprocess.Popen(cmd, shell=True)
    ps.wait()


def compile_training_cmd_string(logs_dir: pathlib.Path, data_dir: pathlib.Path, seed) -> str:
    sub_logs_dir = logs_dir / f'{data_dir.name}-{seed}'
    cmd = f'python -m enet.train --logs_dir {sub_logs_dir} --data_dir {data_dir} --seed {seed}'
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', type=str, help='The directory to write logs to', default='logs')
    parser.add_argument('--datasets_dir', type=str, help='The directory where the datasets to run on are stored',
                        default='/data/sazanovich/aido2/')
    parser.add_argument('--seeds', type=int, help='The number of seeds to use for each dataset', default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    logs_dir = pathlib.Path(args.logs_dir).absolute()
    datasets_dir = pathlib.Path(args.datasets_dir).absolute()
    seeds = args.seeds

    os.makedirs(str(logs_dir), exist_ok=True)

    for filepath in datasets_dir.glob('*'):
        if 'distscapes' not in filepath.name:
            continue
        np.random.seed(27)
        seed_values = np.random.random_integers(0, 2*10**9, seeds)
        for seed in seed_values:
            cmd = compile_training_cmd_string(logs_dir, data_dir=filepath, seed=seed)
            print(cmd)
            run_cmd(cmd)


if __name__ == '__main__':
    main()
