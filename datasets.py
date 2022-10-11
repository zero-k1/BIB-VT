import json
import os

import cv2
import torch
import torch.utils.data
from PIL import Image
import numpy as np

from bib_vt.settings import IMG_SZ_RAW, NR_EPS, MAX_LENGTH, IMG_SZ, IMG_CHANNELS, IMG_SZ_COMPRESSED, SUBSAMPLE_FACTOR

""" This file is a slightly modified version of the datasets.py script in Kanishk Gandhi's 
bib-baselines repo (https://github.com/kanishkg/bib-baselines)"""

class FrameDataset(torch.utils.data.Dataset):

    def __init__(self, path, device, types=None, mode="train", process_data=1, train_split=0.8, val_split=0.05,
                 test_split=0.15):
        self.device = device
        self.path, self.types, self.mode = path, types, mode
        self.json_list, self.path_list = [], []
        self.train_split, self.val_split, self.test_split = train_split, val_split, test_split
        assert train_split + val_split + test_split == 1
        # read video files
        self._read_video_files()
        self.data_tuples = []
        # process json files to extract frame indices for training bib_data
        if process_data:
            for t in types:
                self._process_data(t)
        else:
            for t in types:
                with open(os.path.join(self.path, f'index_dict_{mode}_{t}.json'), 'r') as fp:
                    index_dict = json.load(fp)
                self.data_tuples.extend(index_dict['data_tuples'])
            print(len(self.data_tuples))

    def _process_data(self, t):
        # index videos to make frame retrieval easier
        print('processing files')
        for j, v in zip(self.json_list, self.path_list):
            print(j, v)
            try:
                with open(j, 'r') as f:
                    state = json.load(f)
            except UnicodeDecodeError as e:
                print(f'file skipped {j} with {e}')
                continue
            ep_lens = [len(state[str(x)]) for x in range(NR_EPS)]
            all_eps = []
            first_actions = []
            for e in range(NR_EPS):
                this_ep = []
                first_action = -1
                past_len = sum(ep_lens[:e])
                for f in range(ep_lens[e] - 83):
                    f0y, f0x = state[str(e)][str(f)]
                    f1y, f1x = state[str(e)][str(f + 1)]
                    dx = (f1x - f0x)
                    dy = (f1y - f0y)
                    action = (dx, dy)
                    if action != (0, 0) and first_action == -1:
                        first_action = f + past_len
                        this_ep.append((f0x + 6, f0y + 6))
                    if first_action != -1:
                        this_ep.append(action)
                if len(this_ep) == 0:
                    this_ep.append((f0x + 6, f0y + 6))
                all_eps.append(this_ep)
                first_actions.append(first_action)
            self.data_tuples.append((v, first_actions, all_eps))
        index_dict = {'data_tuples': self.data_tuples}
        with open(os.path.join(self.path, f'index_dict_{self.mode}_{t.split("/")[-1]}.json'), 'w') as fp:
            json.dump(index_dict, fp)

    def _fill_lists(self, t, start, stop):
        self.path_list += [os.path.join(self.path, t, x) for x in sorted(os.listdir(os.path.join(self.path, t)))
                           if
                           x.endswith(f'e.mp4')][start:stop]
        self.json_list += [os.path.join(self.path, t, x) for x in sorted(os.listdir(os.path.join(self.path, t)))
                           if
                           x.endswith(f'e.json')][start:stop]

    def _read_video_files(self):
        for t in self.types:
            print(f'reading files of type {t} in {self.mode}')
            type_length = len(os.listdir(os.path.join(self.path, t))) // 2
            if self.mode == 'eval':
                self.path_list += [os.path.join(self.path, t, x) for x in sorted(os.listdir(os.path.join(self.path, t)))
                                   if
                                   x.endswith(f'e.mp4') or x.endswith(f'u.mp4')]
                self.json_list += [os.path.join(self.path, t, x) for x in sorted(os.listdir(os.path.join(self.path, t)))
                                   if
                                   x.endswith(f'e.json') or x.endswith(f'u.json')]
                continue
            elif self.mode == 'train':
                start = 0
                stop = int(type_length * self.train_split)
            elif self.mode == 'test':
                start = int(type_length * self.train_split)
                stop = int(type_length * (self.train_split + self.test_split))
            elif self.mode == 'val':
                start = int(type_length * (self.train_split + self.test_split))
                stop = type_length
            self._fill_lists(t, start, stop)

    @staticmethod
    def _get_frames(data_tuples):
        sub_sampled_traces = []
        for ep_i, ep in enumerate(data_tuples[2]):
            agent_pos = torch.Tensor(ep)
            trace_length = min(MAX_LENGTH, max(2, agent_pos.shape[0] // SUBSAMPLE_FACTOR + 1))
            ls = torch.linspace(0, agent_pos.shape[0] - 1, trace_length).int().unique()
            x_coords = (agent_pos[:, 0].cumsum(dim=0) / IMG_SZ_RAW * IMG_SZ_COMPRESSED).long()
            y_coords = (agent_pos[:, 1].cumsum(dim=0) / IMG_SZ_RAW * IMG_SZ_COMPRESSED).long()
            path = y_coords * IMG_SZ_COMPRESSED + x_coords
            sub_sampled_trace = torch.index_select(path, 0, ls)
            sub_sampled_traces.append(sub_sampled_trace)

        video_filename = data_tuples[0]
        #print(video_filename)
        cap = cv2.VideoCapture(video_filename)
        frames = []
        for ep_i, ep in enumerate(data_tuples[2]):
            steps = torch.Tensor(sub_sampled_traces[ep_i].shape[0], IMG_CHANNELS, IMG_SZ, IMG_SZ)
            counter = 0
            ls = torch.linspace(data_tuples[1][ep_i], data_tuples[1][ep_i] + len(ep), sub_sampled_traces[ep_i].shape[0])
            for frame in ls:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame.item())
                _, frame = cap.read()
                frame = cv2.resize(frame, (IMG_SZ, IMG_SZ))
                frame = torch.tensor(frame).permute(2, 0, 1) / 255.
                steps[counter] = frame
                counter += 1
            frames.append(steps)
        cap.release()
        return sub_sampled_traces, frames

    def __getitem__(self, idx):
        target, step_through = self._get_frames(self.data_tuples[idx])
        return target, step_through

    def __len__(self):
        return len(self.data_tuples)
