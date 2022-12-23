from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from pprint import pprint
from typing import List
import os

import nmmo
import numpy as np

from data_process import replay_utils


@dataclass
class Stat:
    n_sample: int = 0
    n_move: int = 0
    n_attack: int = 0
    n_sell: int = 0
    n_use: int = 0
    n_buy: int = 0


class Summary:

    def __init__(self, dataset_dir: str, part_size: int = 10000):
        self.statistic = Stat()
        self.part_idx = 0
        self.part_size = part_size
        self.dataset_dir = dataset_dir
        self.dataset = []

    def summarize(self, filepaths: List[str]):
        for path in filepaths:
            obs_traj, actions_traj = replay_utils.parse_lzma(path)
            for obs, action in zip(obs_traj, actions_traj):
                for pid in obs:
                    if pid not in action:
                        continue
                    o, a = obs[pid], action[pid]
                    res = self.filter_and_augment(o, a)
                    if res is None:
                        continue
                    for o, a in res:
                        self.dataset.append([o, a])
                        self.statistic.n_sample += 1
                        if nmmo.io.action.Attack in a:
                            self.statistic.n_attack += 1
                        if nmmo.io.action.Move in a:
                            self.statistic.n_move += 1
                        if nmmo.io.action.Use in a:
                            self.statistic.n_use += 1
                        if nmmo.io.action.Sell in a:
                            self.statistic.n_sell += 1
                        if nmmo.io.action.Buy in a:
                            self.statistic.n_buy += 1
            if len(self.dataset) >= self.part_size:
                self.save()
        pprint(self.statistic)
        self.save()

    @staticmethod
    def parse_replay(filepath: str, save_path: str):
        dataset = []
        obs_traj, actions_traj = replay_utils.parse_lzma(filepath)
        for obs, action in zip(obs_traj, actions_traj):
            for pid in obs:
                if pid not in action:
                    continue
                o, a = obs[pid], action[pid]
                dataset.append([o, a])
        print(f"save dataset: {save_path}, samples num: {len(dataset)}")
        np.savez_compressed(save_path, data=dataset)

    @staticmethod
    def filter_by_timealive(obs_traj, actions_traj, time_thres=1000):
        pass
        
    @staticmethod
    def shuffle_inventory(o, a):
        o2, a2 = deepcopy(o), deepcopy(a)
        N = int(o2["Item"]["N"])
        permutation = np.random.permutation(N)
        o2["Item"]["Continuous"][:N] = o2["Item"]["Continuous"][permutation]
        if nmmo.io.action.Use in a2:
            prev = a2[nmmo.io.action.Use][nmmo.io.action.Item]
            if prev < N:
                a2[nmmo.io.action.Use][nmmo.io.action.Item] = np.where(
                    permutation == prev)[0][0]
        if nmmo.io.action.Sell in a2:
            prev = a2[nmmo.io.action.Sell][nmmo.io.action.Item]
            if prev < N:
                a2[nmmo.io.action.Sell][nmmo.io.action.Item] = np.where(
                    permutation == prev)[0][0]
        return o2, a2

    @staticmethod
    def filter_and_augment(o, a):
        if nmmo.io.action.Sell in a or nmmo.io.action.Use in a:
            o2, a2 = Summary.shuffle_inventory(o, a)
            o3, a3 = Summary.shuffle_inventory(o, a)
            return [(o, a), (o2, a2), (o3, a3)]

        if nmmo.io.action.Buy in a and np.random.random() < 0.2:
            return [(o, a)]

        if nmmo.io.action.Attack in a and np.random.random() < 0.1:
            return [(o, a)]

        if np.random.random() < 0.05:
            return [(o, a)]

        return None

    def save(self):
        path = Path(self.dataset_dir) / f"part-{self.part_idx}.npz"
        print(f"save dataset: {path}")
        np.savez_compressed(path, data=self.dataset)
        self.dataset.clear()
        self.part_idx += 1


if __name__ == "__main__":
    s = Summary(dataset_dir="./dataset/npz50")
    paths = sorted(glob("./dataset/replays/*.lzma"))
    print(f"num of replay: {len(paths)}")
    s.summarize(paths)
