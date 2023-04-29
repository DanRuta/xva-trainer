import math
import os
import random
import traceback

import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

# ====================
# ====================
# ====================
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :].copy())

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        input_data = input_data.view((input_data.shape[0], input_data.shape[-1]))
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples


        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        with torch.no_grad():
            inverse_transform = F.conv_transpose2d(
                recombine_magnitude_phase.unsqueeze(-1),
                self.inverse_basis.unsqueeze(-1),
                stride=self.hop_length,
                padding=0).squeeze(-1)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
stft = TacotronSTFT(1024, 256, 1024, 80, 22050, 0, 8000)

def get_mel (y, n_fft=None, num_mels=None, sampling_rate=None, hop_size=None, win_size=None, fmin=None, fmax=None, center=False, device="cpu"):
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    max_wav_value = 32768.0
    audio_norm = y / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec

# ====================
# ====================
# ====================



def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_group_dataset_filelist(a, voice_dirs):
    training_files = []
    for v_dir in voice_dirs:
        found = 0
        not_found = 0
        with open(f'{v_dir}/metadata.csv', 'r', encoding='utf-8') as fi:
            lines = fi.read().split("\n")
            for i in range(a.dm):
                for line in lines:
                    if len(line):
                        if os.path.exists(f'{v_dir}/wavs/{line.split("|")[0].split(".")[0]}.wav'):
                            training_files.append(f'{v_dir}/wavs/{line.split("|")[0].split(".")[0]}.wav')
                            found += 1

                        else:
                            not_found += 1
        if not_found>0:
            if not_found==not_found+found:
                print(f'NOT FOUND: {v_dir}')
            else:
                print(f'{not_found}/{not_found+found} not found for: {v_dir}')

    print(f'training_files, {len(training_files)}')
    return training_files, None

def get_dataset_filelist(input_training_file, input_wavs_dir, dm=None, use_embs=False):
    with open(input_training_file, 'r', encoding='utf-8') as fi:
        training_files = []
        found = 0
        not_found = 0
        lines = fi.read().split("\n")
        for line in lines:
            if len(line):
                # fname = line.split("|")[0].replace("wavs", "").split("/")[-1]
                fname = line.split("|")[0].split("/")[-1]

                if len(fname.strip())==0:
                    continue

                if ".wav" not in fname:
                    fname = fname+".wav"

                if os.path.exists(f'{input_wavs_dir}/{fname}') and (not use_embs or os.path.exists(f'{input_wavs_dir.replace("/wavs/", "/cond_embs/")}/{fname.replace(".wav", ".npy")}')):
                # if os.path.exists(f'{input_wavs_dir}/{fname}'):
                    training_files.append(f'{input_wavs_dir}/{fname}')
                    found += 1
                else:
                    not_found += 1
        if not_found>0:
            print(f'{not_found}/{not_found+found} not found for {input_wavs_dir}')
        else:
            print(f'OK: {input_wavs_dir}')

    if dm is None:
        dm = 1000/(len(training_files)-int(not_found))
        dm = max(1, round(dm))

    training_files_total = []

    for _ in range(dm):
        random.shuffle(training_files)
        training_files_total += training_files

    # print(f'training_files_total, {len(training_files_total)} ({len(training_files_total) / dm})')
    return training_files_total, int(not_found), dm


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, h=None):
        self.audio_files = training_files
        self.h = h
        self.USE_EMB_CONDITIONING = self.h is not None and self.h.USE_EMB_CONDITIONING
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0

        self.audio_cache = {}
        self.MAX_CACHE_ITEMS = 10000
        self.MAX_CACHE_ITEMS = 5000
        self.audio_cache_count = 0


    def __getitem__(self, index):
        filename = self.audio_files[index]
        filename = filename if filename.endswith(".wav") else filename+".wav"

        if filename in self.audio_cache.keys():
            audio = self.audio_cache[filename]
        else:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95

            audio = torch.FloatTensor(audio)
            audio = audio.unsqueeze(0)

            if self.audio_cache_count<self.MAX_CACHE_ITEMS:
                self.audio_cache[filename] = audio
                self.audio_cache_count += 1

        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start+self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel = mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax, center=False)
        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss, center=False)

        cond_emb = 0
        if self.USE_EMB_CONDITIONING:
            cond_emb = np.load(filename.replace("/wavs/", "/cond_embs/").replace(".wav", ".npy"))

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze(), cond_emb)




    def __len__(self):
        return len(self.audio_files)
