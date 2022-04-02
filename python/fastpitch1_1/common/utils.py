# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

import torch
from scipy.io.wavfile import read


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


def load_wav_to_torch(full_path, force_sampling_rate=None):
    if force_sampling_rate is not None:
        data, sampling_rate = librosa.load(full_path, sr=force_sampling_rate)
    else:
        sampling_rate, data = read(full_path)

    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


# def load_filepaths_and_text(dataset_path, fnames, has_speakers=False, split="|", dm=1):
#     def split_line(root, line):
#         parts = line.strip().split(split)
#         if has_speakers:
#             paths, non_paths = parts[:-2], parts[-2:]
#         else:
#             paths, non_paths = parts[:-1], parts[-1:]
#         return tuple(str(Path(root, p)) for p in paths) + tuple(non_paths)

#     fpaths_and_text = []
#     for fname in fnames:
#         with open(fname, encoding='utf-8') as f:
#             fpaths_and_text += [split_line(dataset_path, line) for line in f]
#     print(fpaths_and_text[:5])
#     print("====")
#     print(fpaths_and_text[0])
#     # dfgfdg()
#     return fpaths_and_text

def load_filepaths_and_text(dataset_path, fnames, has_speakers=False, split="|", dm=1):
    def split_line(root, line):
        parts = line.strip().split(split)
        # print(f'parts, {parts}')

        if has_speakers:
            paths = [str(Path(root, f'wavs/{parts[0]}')), str(Path(root, f'pitch/{parts[0].replace(".wav", ".pt")}')), parts[1], parts[2]]
            # paths, non_paths = parts[:-2], parts[-2:]
        else:
            paths = [str(Path(root, f'wavs/{parts[0]}')), str(Path(root, f'pitch/{parts[0].replace(".wav", ".pt")}')), parts[1]]

            # non_paths = []

        # audiopath.replace("wavs", "durations").replace(".wav", ".pt")
        # if os.path.exists(f'{root}/pitch'):
        # # if os.path.exists(f'{root}/wavs'):
        #     # if not os.path.exists(f'{root}/{paths[0]}'):
        #     if not os.path.exists(paths[0]):
        #         # print(f'{root}/{paths[0]}')
        #         return None
            # # if not os.path.exists(f'{root}/{paths[0]}'.replace("wavs", "mels").replace(".wav", ".pt")):
            # if not os.path.exists(paths[0].replace("wavs", "mels").replace(".wav", ".pt")):
            #     # print(f'{root}/{paths[0]}'.replace("wavs", "mels").replace(".wav", ".pt"))
            #     return None
            # # if not os.path.exists(f'{root}/{paths[0]}'.replace("wavs", "pitch").replace(".wav", ".pt")):
            # if not os.path.exists(paths[0].replace("wavs", "pitch").replace(".wav", ".pt")):
            #     # print(f'{root}/{paths[0]}'.replace("wavs", "pitch").replace(".wav", ".pt"))
            #     return None
            # # if not os.path.exists(f'{root}/{paths[0]}'.replace("wavs", "durations").replace(".wav", ".pt")):
            # #     print(f'{root}/{paths[0]}'.replace("wavs", "durations").replace(".wav", ".pt"))
            # #     return None

            # paths, non_paths = parts[:-1], parts[-1:]
        if not os.path.exists(paths[0]):
            if os.path.exists(paths[0]+".wav"):
                paths[0] = paths[0]+".wav"
                return tuple(str(p) for p in paths)
            # print(f'NO: {paths[0]}')
            return None

        return tuple(str(p) for p in paths) #+ tuple(non_paths)
        # return tuple(str(Path(root, p)) for p in paths) + tuple(non_paths)

    fpaths_and_text = []
    fpaths_and_text_total = []
    nope = 0
    # loops_done = 0
    actual_num_lines = None

    for fname in fnames:
        with open(fname, encoding='utf-8') as f:
            # fpaths_and_text += [split_line(dataset_path, line) for line in f]
            lines = f.read().split("\n")
            lines = [line for line in lines if len(line.strip())]
            actual_num_lines = len(lines)
            # dm *= int(13000/actual_num_lines)



            # Keep looping through the files until the number of lines gathered is similar to LJ
            # while len(fpaths_and_text)<13000:
                # if loops_done>0:
                #     random.shuffle(lines)
            for line in lines:
                if line.startswith("|"):
                    continue
                data = split_line(dataset_path, line)
                if data is not None:
                    fpaths_and_text.append(data)
                else:
                #     if loops_done==0:
                    nope += 1
                # Stop if mid-way through a loop and the target number of lines is reached (to stop v large datasets from adding too much, if just under 13k)
                # if loops_done>0 and len(fpaths_and_text)>=13000:
                #     break
                # if dm==-1:
                #     break
                # loops_done += 1

                # I should do more of the dm repeating, instead of the 13k repeating, as the 13k repeating will imbalance the data more.

            if (actual_num_lines-nope)==0:
                raise "No audio files were found in the dataset folder. Check your metadata.csv file, to see if the file names match what you have"
            dm *= 13000/(actual_num_lines-nope)


    dm = max(1, int(dm))
    # dm = 1
    # print("NOT SUFFLING ----- DEBUG")
    for _ in range(dm):
        random.shuffle(fpaths_and_text)
        fpaths_and_text_total += fpaths_and_text


    summary_str = f'Composed items: {len(fpaths_and_text_total)} | Base number of data lines: {actual_num_lines} | Missing: {nope} | DM: {dm}'
    print(summary_str)

    # print(fpaths_and_text[:5])
    # print("====")
    # print(fpaths_and_text[0])
    # dfgdf()
    return fpaths_and_text_total, actual_num_lines, summary_str


def stats_filename(dataset_path, filelist_path, feature_name):
    stem = Path(filelist_path).stem
    return Path(dataset_path, f'{feature_name}_stats__{stem}.json')


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def to_device_async(tensor, device):
    return tensor.to(device, non_blocking=True)


def to_numpy(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


def prepare_tmp(path):
    if path is None:
        return
    p = Path(path)
    if p.is_dir():
        warnings.warn(f'{p} exists. Removing...')
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=False, exist_ok=False)
