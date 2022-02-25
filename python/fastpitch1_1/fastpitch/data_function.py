# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import sys
import time
import functools
import json
import re
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import betabinom

try:
    sys.path.append(".")
    import resources.app.python.fastpitch1_1.common.layers as layers
    from resources.app.python.fastpitch1_1.common.text.text_processing import TextProcessing
    from resources.app.python.fastpitch1_1.common.utils import load_wav_to_torch, load_filepaths_and_text#, to_gpu
except:
    try:
        import python.fastpitch1_1.common.layers as layers
        from python.fastpitch1_1.common.text.text_processing import TextProcessing
        from python.fastpitch1_1.common.utils import load_wav_to_torch, load_filepaths_and_text#, to_gpu
    except:
        import common.layers as layers
        from common.text.text_processing import TextProcessing
        from common.utils import load_wav_to_torch, load_filepaths_and_text#, to_gpu


class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        # self.bank = functools.lru_cache(beta_binomial_prior_distribution)
        self.bank = lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def estimate_pitch(wav, mel_len, method='pyin', normalize_mean=None, normalize_std=None, n_formants=1):

    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'praat':

        dfgdfg()

        # snd = parselmouth.Sound(wav)
        # pitch_mel = snd.to_pitch(time_step=snd.duration / (mel_len + 3)
        #                          ).selected_array['frequency']
        # assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

        # pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)

        # if n_formants > 1:
        #     formant = snd.to_formant_burg(
        #         time_step=snd.duration / (mel_len + 3))
        #     formant_n_frames = formant.get_number_of_frames()
        #     assert np.abs(mel_len - formant_n_frames) <= 1.0

        #     formants_mel = np.zeros((formant_n_frames + 1, n_formants - 1))
        #     for i in range(1, formant_n_frames + 1):
        #         formants_mel[i] = np.asarray([
        #             formant.get_value_at_time(
        #                 formant_number=f,
        #                 time=formant.get_time_from_frame_number(i))
        #             for f in range(1, n_formants)
        #         ])

        #     pitch_mel = torch.cat(
        #         [pitch_mel, torch.from_numpy(formants_mel).permute(1, 0)],
        #         dim=0)

    elif method == 'pyin':

        snd, sr = librosa.load(wav)
        pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
            snd, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), frame_length=1024)
        assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError

    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel


def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch




class TTSDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self,
                 dataset_path,
                 audiopaths_and_text,
                 text_cleaners,
                 n_mel_channels,
                 symbol_set='english_basic',
                 p_arpabet=1.0,
                 n_speakers=1,
                 load_mel_from_disk=True,
                 load_pitch_from_disk=True,
                 pitch_mean=214.72203,  # LJSpeech defaults
                 pitch_std=65.72038,
                 max_wav_value=None,
                 sampling_rate=None,
                 filter_length=None,
                 hop_length=None,
                 win_length=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 prepend_space_to_text=True,
                 append_space_to_text=True,
                 pitch_online_dir=None,
                 betabinomial_online_dir=None,
                 use_betabinomial_interpolator=True,
                 pitch_online_method='pyin',
                 dm=1,
                 use_file_caching=True,
                 training_stage=1,
                 cmudict=None,
                 **ignored):


        # Expect a list of filenames
        if type(audiopaths_and_text) is str:
            audiopaths_and_text = [audiopaths_and_text]

        self.dm = dm
        self.use_file_caching = use_file_caching
        self.training_stage = training_stage
        self.dataset_path = dataset_path
        self.audiopaths_and_text, self.actual_num_lines = load_filepaths_and_text(
            dataset_path, audiopaths_and_text,
            has_speakers=(n_speakers > 1), dm=self.dm)
        self.load_mel_from_disk = load_mel_from_disk
        # if not load_mel_from_disk:
        if True:
            self.max_wav_value = max_wav_value
            self.sampling_rate = sampling_rate
            self.stft = layers.TacotronSTFT(
                filter_length, hop_length, win_length,
                n_mel_channels, sampling_rate, mel_fmin, mel_fmax)
        self.load_pitch_from_disk = load_pitch_from_disk

        self.prepend_space_to_text = True#prepend_space_to_text
        self.append_space_to_text = True#append_space_to_text

        self.p_arpabet = p_arpabet
        # print(f'self.p_arpabet, {self.p_arpabet}')
        # assert p_arpabet == 0.0 or p_arpabet == 1.0, (
        #     'Only 0.0 and 1.0 p_arpabet is currently supported. '
        #     'Variable probability breaks caching of betabinomial matrices.')

        self.tp = TextProcessing(cmudict, symbol_set, text_cleaners, p_arpabet=p_arpabet)
        self.n_speakers = n_speakers
        self.pitch_tmp_dir = pitch_online_dir
        self.f0_method = pitch_online_method
        use_betabinomial_interpolator = False
        self.betabinomial_tmp_dir = betabinomial_online_dir
        self.use_betabinomial_interpolator = use_betabinomial_interpolator

        if use_betabinomial_interpolator:
            self.betabinomial_interpolator = BetaBinomialInterpolator()

        expected_columns = (2 + int(load_pitch_from_disk) + (n_speakers > 1))

        assert not (load_pitch_from_disk and self.pitch_tmp_dir is not None)

        if len(self.audiopaths_and_text[0]) < expected_columns:
            raise ValueError(f'Expected {expected_columns} columns in audiopaths file. '
                             'The format is <mel_or_wav>|[<pitch>|]<text>[|<speaker_id>]')

        if len(self.audiopaths_and_text[0]) > expected_columns:
            print('WARNING: Audiopaths file has more columns than expected')

        to_tensor = lambda x: torch.Tensor([x]) if type(x) is float else x
        self.pitch_mean = to_tensor(pitch_mean)
        self.pitch_std = to_tensor(pitch_std)



        self.dataset_cache = {}
        if "mel" not in self.dataset_cache.keys():
            self.dataset_cache["mel"] = {}
        if "text_prior" not in self.dataset_cache.keys():
            self.dataset_cache["text_prior"] = {}
        if "phon_prior" not in self.dataset_cache.keys():
            self.dataset_cache["phon_prior"] = {}
        if "text_text" not in self.dataset_cache.keys():
            self.dataset_cache["text_text"] = {}
        if "phon_text" not in self.dataset_cache.keys():
            self.dataset_cache["phon_text"] = {}
        if "pitch" not in self.dataset_cache.keys():
            self.dataset_cache["pitch"] = {}
        if "durs_arpabet" not in self.dataset_cache.keys():
            self.dataset_cache["durs_arpabet"] = {}
        if "durs_text" not in self.dataset_cache.keys():
            self.dataset_cache["durs_text"] = {}
        if "resemblyzer" not in self.dataset_cache.keys():
            self.dataset_cache["resemblyzer"] = {}

        self.total_cached_samples = 0
        self.MAX_CACHE_SAMPLES = 1000
        # self.MAX_CACHE_SAMPLES = 5000
        # self.MAX_CACHE_SAMPLES = 4000
        # self.MAX_CACHE_SAMPLES = 3000




    def __getitem__(self, index):

        # global dataset_cache_ref
        # print(f'__getitem__ dataset_cache_ref, {dataset_cache_ref}')

        # Separate filename and text
        # if self.n_speakers > 1:
        #     audiopath, *extra, text, speaker = self.audiopaths_and_text[index]
        #     speaker = int(speaker)
        # else:
        #     audiopath, *extra, text = self.audiopaths_and_text[index]
        #     speaker = None

        audiopath, *extra, text = self.audiopaths_and_text[index]
        speaker = None
        # if self.training_stage!=-1:
        # speaker = self.get_speaker(audiopath)

        use_arpabet = np.random.uniform() < self.p_arpabet


        mel = self.get_mel(audiopath)
        # text = self.get_text(text, use_arpabet, print_text=index==1)
        text = self.get_text(text, use_arpabet, print_text=False)

        if self.training_stage==1 or self.training_stage==2:
            pitch = [0]
            energy = [0]
        else:
            pitch = self.get_pitch(index, mel.shape[-1])
            energy = np.linalg.norm(mel, ord=2, axis=0)



            try:
                # assert pitch.size(-1) == mel.size(-1)
                assert pitch.shape[-1] == mel.shape[-1]
            except:
                print("pitch.shape, mel.shape", pitch.shape, mel.shape)
                raise

        if self.training_stage==-1 or self.training_stage==1:
            attn_prior = self.get_prior(index, mel.shape[1], text.shape[0], use_arpabet)
            durs = None
        else:
            attn_prior = None
            durs = self.get_durs(index, use_arpabet)



        # No higher formants?
        if (self.training_stage==-1 or self.training_stage==3 or self.training_stage==4) and len(pitch.shape) == 1:
            pitch = pitch[None, :]


        return (text, mel, len(text), pitch, energy, speaker, attn_prior, durs, audiopath)

    def __len__(self):
        return len(self.audiopaths_and_text)


    def get_speaker (self, filename):

        emb_filename = filename.replace("\\wavs\\", "\\resemblyzer\\").replace(".wav", ".npy")

        try:
            emb = self.dataset_cache["resemblyzer"][emb_filename]
        except:
            emb = np.load(emb_filename)
            self.dataset_cache["resemblyzer"][emb_filename] = emb

        return emb


    def get_durs(self, index, use_arpabet):
        audiopath, *fields = self.audiopaths_and_text[index]
        # durs_filename = audiopath.replace("\\wavs\\", f'\\durs_{"arpabet" if use_arpabet else "text"}\\').replace(".wav", ".pt")
        durs_filename = audiopath.replace("\\wavs\\", f'\\durs_{"arpabet" if use_arpabet else "text"}\\').replace(".wav", ".npy")

        try:
            durs = self.dataset_cache["durs_arpabet" if use_arpabet else "durs_text"][durs_filename]
        except:
            durs = np.load(durs_filename.replace(".pt", ".npy"))
            self.dataset_cache["durs_arpabet" if use_arpabet else "durs_text"][durs_filename] = durs

        return durs


    def get_mel(self, filename):

        mel_filename = filename.replace("\\wavs\\", "\\mels\\").replace(".wav", ".npy")
        try:

            try:
                melspec = self.dataset_cache["mel"][mel_filename]
            except:
                try:
                    melspec = np.load(mel_filename)
                except:
                    # Some other dataloader process might have started but not finished writing this to file
                    # so loading it during that will error. Try waiting a couple of seconds, to give the other
                    # process some time to finish writing to file, then try again
                    time.sleep(4)
                    melspec = np.load(mel_filename)

                if self.total_cached_samples < self.MAX_CACHE_SAMPLES:
                    self.total_cached_samples += 1
                    self.dataset_cache["mel"][mel_filename] = melspec
        except:

            if not os.path.exists(mel_filename):
                audio, sampling_rate = load_wav_to_torch(filename)
                if sampling_rate != self.stft.sampling_rate:
                    raise ValueError("{} SR doesn't match target {} SR".format(
                        sampling_rate, self.stft.sampling_rate))
                audio_norm = audio / self.max_wav_value
                audio_norm = audio_norm.unsqueeze(0)
                audio_norm = torch.autograd.Variable(audio_norm,
                                                     requires_grad=False)
                melspec = self.stft.mel_spectrogram(audio_norm)
                melspec = torch.squeeze(melspec, 0)

                os.makedirs("/".join(mel_filename.split("\\")[:-1]), exist_ok=True)
                melspec = melspec.numpy()
                np.save(mel_filename, melspec)

                if self.total_cached_samples < self.MAX_CACHE_SAMPLES:
                    self.total_cached_samples += 1
                    self.dataset_cache["mel"][mel_filename] = melspec
            else:
                raise

        return melspec

    def get_text(self, text, use_arpabet, print_text=False):

        input_text = text

        try:
            return self.dataset_cache["phon_text" if use_arpabet else "text_text"][input_text]
        except:
            text = self.tp.encode_text(text, use_arpabet=use_arpabet, print_text=print_text)
            space = [self.tp.encode_text("A A", use_arpabet=use_arpabet)[1]]

            if self.prepend_space_to_text:
                text = space + text

            if self.append_space_to_text:
                text = text + space

            text = torch.LongTensor(text)

            self.dataset_cache["phon_text" if use_arpabet else "text_text"][input_text] = text
            return text


    def get_prior(self, index, mel_len, text_len, use_arpabet):

        audiopath, *_ = self.audiopaths_and_text[index]
        # prior_path = audiopath.replace("wavs", f'betabinomial_{"arpabet" if use_arpabet else "text"}').replace(".wav", ".pt")
        prior_path = audiopath.replace("wavs", f'betabinomial_{"arpabet" if use_arpabet else "text"}').replace(".wav", ".npy")

        try:
            try:
                return self.dataset_cache["phon_prior" if use_arpabet else "text_prior"][prior_path]
            except:

                # try:
                #     # attn_prior = torch.load(prior_path)
                #     attn_prior = np.load(prior_path)
                # except:
                #     # Some other dataloader process might have started but not finished writing this to file
                #     # so loading it during that will error. Try waiting a couple of seconds, to give the other
                #     # process some time to finish writing to file, then try again
                #     time.sleep(4)
                #     attn_prior = np.load(prior_path)
                #     # attn_prior = torch.load(prior_path)

                try:
                    attn_prior = np.load(prior_path)
                except:
                    # Some other dataloader process might have started but not finished writing this to file
                    # so loading it during that will error. Try waiting a couple of seconds, to give the other
                    # process some time to finish writing to file, then try again
                    time.sleep(1)
                    # attn_prior = np.load(prior_path)
                    try:
                        attn_prior = np.load(prior_path)
                    except:
                        # Some other dataloader process might have started but not finished writing this to file
                        # so loading it during that will error. Try waiting a couple of seconds, to give the other
                        # process some time to finish writing to file, then try again
                        time.sleep(1)
                        # attn_prior = np.load(prior_path)
                        try:
                            attn_prior = np.load(prior_path)
                        except:
                            # Some other dataloader process might have started but not finished writing this to file
                            # so loading it during that will error. Try waiting a couple of seconds, to give the other
                            # process some time to finish writing to file, then try again
                            time.sleep(1)
                            # attn_prior = np.load(prior_path)
                            try:
                                attn_prior = np.load(prior_path)
                            except:
                                # Some other dataloader process might have started but not finished writing this to file
                                # so loading it during that will error. Try waiting a couple of seconds, to give the other
                                # process some time to finish writing to file, then try again
                                time.sleep(1)
                                attn_prior = np.load(prior_path)



                if self.total_cached_samples < self.MAX_CACHE_SAMPLES:
                    self.total_cached_samples += 1
                    self.dataset_cache["phon_prior" if use_arpabet else "text_prior"][prior_path] = attn_prior
                # print(f'attn_prior, {attn_prior.shape}')
                return attn_prior
        except:

            attn_prior = beta_binomial_prior_distribution(text_len, mel_len)
            os.makedirs("/".join(prior_path.split("\\")[:-1]), exist_ok=True)
            np.save(prior_path, attn_prior.numpy())
            if self.total_cached_samples < self.MAX_CACHE_SAMPLES:
                self.total_cached_samples += 1
                self.dataset_cache["phon_prior" if use_arpabet else "text_prior"][prior_path] = attn_prior
            return attn_prior


    def get_pitch(self, index, mel_len=None):
        audiopath, *fields = self.audiopaths_and_text[index]
        filename = audiopath.replace("\\wavs\\", "\\pitch\\").replace(".wav", ".npy")

        try:
            return self.dataset_cache["pitch"][filename]
        except:
            try:
                if self.training_stage==-1:
                    raise # Avoid using any cached (pre-normalized) files
                pitch = np.load(filename)
            except:
                if not os.path.exists(filename):
                    pitch = estimate_pitch(audiopath, mel_len, self.f0_method, None, None)

                    os.makedirs("/".join(filename.split("\\")[:-1]), exist_ok=True)
                    if self.training_stage==-1:
                        return pitch

                    if self.pitch_mean is not None:
                        pitch = normalize_pitch(pitch, self.pitch_mean.numpy(), self.pitch_std.numpy())

                    pitch = pitch.numpy()
                    np.save(filename, pitch)
                else:
                    # Some other dataloader process might have started but not finished writing this to file
                    # so loading it during that will error. Try waiting a couple of seconds, to give the other
                    # process some time to finish writing to file, then try again
                    time.sleep(4)
                    pitch = np.load(filename)

            self.dataset_cache["pitch"][filename] = pitch
            return pitch



class TTSCollate:
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __call__(self, batch):
        """Collate training batch from normalized text and mel-spec"""
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].shape[0]
        max_target_len = max([x[1].shape[1] for x in batch])

        # Include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            # mel_padded[i, :, :mel.size(1)] = mel
            mel_padded[i, :, :mel.shape[1]] += mel
            output_lengths[i] = mel.shape[1]

        # if not self.KL_ONLY:
        if self.training_stage==3 or self.training_stage==4 or self.training_stage==-1:
            n_formants = batch[0][3].shape[0]
            pitch_padded = torch.zeros(mel_padded.size(0), n_formants,
                                       mel_padded.size(2), dtype=batch[0][0].dtype)
            energy_padded = torch.zeros_like(pitch_padded[:, 0, :])

            for i in range(len(ids_sorted_decreasing)):
                pitch = batch[ids_sorted_decreasing[i]][3]
                energy = batch[ids_sorted_decreasing[i]][4]
                try:
                    pitch_padded[i, :, :pitch.shape[1]] += pitch#.long()
                except:
                    pitch_padded[i, :, :pitch.shape[1]] += pitch.long()

                energy_padded[i, :energy.shape[0]] += energy
        else:
            pitch_padded = torch.tensor([0])
            energy_padded = torch.tensor([0])

        if batch[0][5] is not None:
            # speaker = torch.zeros_like(input_lengths)
            speaker = torch.zeros((len(batch), 256))
            for i in range(len(ids_sorted_decreasing)):
                # speaker[i] = batch[ids_sorted_decreasing[i]][5]
                speaker[i, :] += batch[ids_sorted_decreasing[i]][5]
        else:
            speaker = None


        # Either the alignments, or the extracted durations
        if self.training_stage!=1 and self.training_stage!=-1:
            # n_formants = batch[0][3].shape[0]
            max_dur = 0
            for i in range(len(ids_sorted_decreasing)):
                durs = batch[ids_sorted_decreasing[i]][7]
                # max_dur = max(max_dur, len(durs))
                # max_dur = max(max_dur, durs.shape[0])
                try:
                    max_dur = max(max_dur, durs.shape[0])
                except:
                    print("max_dur", max_dur)
                    print("durs", durs)
                    print("durs.shape", durs.shape)
                    raise


            # durs_padded = torch.zeros(mel_padded.size(0), n_formants,
            durs_padded = torch.zeros(len(batch), max_dur,
                                       # mel_padded.size(2), dtype=batch[0][7].dtype)
                                       # dtype=batch[0][7].dtype)
                                       dtype=batch[0][0].dtype)
            for i in range(len(ids_sorted_decreasing)):
                durs = batch[ids_sorted_decreasing[i]][7]
                durs_padded[i, :durs.shape[0]] += durs

            attn_prior_padded = torch.tensor([0])

        else:
            durs_padded = torch.tensor([0])


            # attn_prior_padded = torch.zeros(len(batch), max_target_len, max_input_len)

            # =======
            max_prior_2 = max([x[6].shape[1] for x in batch])
            max_prior_size = 0
            for i in range(len(ids_sorted_decreasing)):
                max_prior_size = max(max_prior_size, batch[ids_sorted_decreasing[i]][6].shape[0])
            attn_prior_padded = torch.zeros(len(batch), max(max_target_len, max_prior_size), max(max_prior_2, max_input_len))
            # =======
            attn_prior_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                prior = batch[ids_sorted_decreasing[i]][6]
                # attn_prior_padded[i, :prior.size(0), :prior.size(1)] = prior
                try:
                    attn_prior_padded[i, :prior.shape[0], :prior.shape[1]] += prior
                except:
                    print("\nmax", max(max_target_len, batch[ids_sorted_decreasing[0]][6].shape[0]))
                    print(f'attn_prior_padded, {attn_prior_padded.shape}')
                    print(f'max_target_len, {max_target_len}')
                    print(f'prior, {prior.shape}')



        # Count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)

        max_inp_lengths = [max_input_len for i in ids_sorted_decreasing]
        max_inp_lengths = torch.tensor(max_inp_lengths)

        max_mel_lengths = [max_target_len for i in ids_sorted_decreasing]
        max_mel_lengths = torch.tensor(max_mel_lengths)

        audiopaths = [batch[i][8] for i in ids_sorted_decreasing]

        return (text_padded, input_lengths, mel_padded, output_lengths, len_x,
                pitch_padded, energy_padded, speaker, attn_prior_padded,
                durs_padded,
                max_inp_lengths,
                max_mel_lengths,
                audiopaths)


def to_gpu(x, device):
    x = x.contiguous()
    if torch.cuda.is_available():
        # x = x.cuda(non_blocking=True)
        x = x.to(torch.device(f'cuda:{device}'), non_blocking=True)
    return torch.autograd.Variable(x)

# def batch_to_gpu(batch, KL_ONLY=False):
def batch_to_gpu(batch, training_stage=1, device=0):
    (text_padded, input_lengths, mel_padded, output_lengths, len_x,
     pitch_padded, energy_padded, speaker, attn_prior, durs_padded, max_inp_lengths, max_mel_lengths, audiopaths) = batch

    text_padded = to_gpu(text_padded, device).long()
    input_lengths = to_gpu(input_lengths, device).long()
    mel_padded = to_gpu(mel_padded, device).float()
    output_lengths = to_gpu(output_lengths, device).long()
    # if not KL_ONLY:
    if not training_stage==1:
        pitch_padded = to_gpu(pitch_padded, device).float()
        energy_padded = to_gpu(energy_padded, device).float()
        durs_padded = to_gpu(durs_padded, device).float()
        attn_prior = None
    else:
        del pitch_padded, energy_padded, durs_padded
        pitch_padded = None
        energy_padded = None
        durs_padded = None
        attn_prior = to_gpu(attn_prior, device).float()

    if speaker is not None:
        # speaker = to_gpu(speaker, device).long()
        speaker = to_gpu(speaker, device).float()

    # Alignments act as both inputs and targets - pass shallow copies

    # max_inp_lengths = text_padded ##### TEMP
    max_inp_lengths = to_gpu(max_inp_lengths, device).float()
    max_mel_lengths = to_gpu(max_mel_lengths, device).float()

    x = [text_padded, input_lengths, mel_padded, output_lengths,
         pitch_padded, energy_padded, speaker, attn_prior, durs_padded, max_inp_lengths, max_mel_lengths, audiopaths]
    y = [mel_padded, input_lengths, output_lengths]
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
