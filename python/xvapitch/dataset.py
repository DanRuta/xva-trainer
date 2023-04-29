import collections
import os
import random
import traceback
from typing import Dict, List

import time
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

try:
    from python.xvapitch.audio import AudioProcessor
    from python.xvapitch.util import text_to_sequence, prepare_data, prepare_stop_target, prepare_tensor, sequence_mask
    # from python.xvapitch.util import prepare_data, prepare_stop_target, prepare_tensor, sequence_mask
    from python.xvapitch.speaker_representation.main import ResNetSpeakerEncoder
    from python.xvapitch.text import get_text_preprocessor, lang_names
    from python.xvapitch.stft import STFT
except:
    from audio import AudioProcessor
    from util import text_to_sequence, prepare_data, prepare_stop_target, prepare_tensor, sequence_mask
    # from util import prepare_data, prepare_stop_target, prepare_tensor, sequence_mask
    from speaker_representation.main import ResNetSpeakerEncoder
    from text import get_text_preprocessor, lang_names
    from stft import STFT

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
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
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)
def dynamic_range_decompression(x, C=1):
    return torch.exp(x) / C

class TTSDataset(Dataset):
    def __init__(
        self,
        args,
        # outputs_per_step: int,
        # text_cleaner: list,
        # compute_linear_spec: bool,
        meta_data=None,
        priors_data_list=None,
        meta_data_finetune=None,
        # compute_f0: bool = False,
        # f0_cache_path: str = None,
        # characters: Dict = None,
        # custom_symbols: List = None,
        # add_blank: bool = False,
        # return_wav: bool = False,
        # batch_group_size: int = 0,
        min_seq_len: int = 0,
        # max_seq_len: int = float("inf"),
        # use_phonemes: bool = False,
        # phoneme_cache_path: str = None,
        # enable_eos_bos: bool = False,
        # speaker_id_mapping: Dict = None,
        # language_id_mapping: Dict = None,
        # use_noise_augment: bool = False,
        # verbose: bool = False,
        lang_override = None,
        is_ft = False
    ):
        super().__init__()
        self.args = args
        self.is_ft = is_ft

        # ====================================== TODO, remove
        _pad = "_"
        _punctuations = "!'(),-.:;? "
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u00af\u00b7\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e6\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f9\u00fa\u00fb\u00fc\u00ff\u0101\u0105\u0107\u0113\u0119\u011b\u012b\u0131\u0142\u0144\u014d\u0151\u0153\u015b\u016b\u0171\u017a\u017c\u01ce\u01d0\u01d2\u01d4\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f\u0451\u0454\u0456\u0457\u0491\u2013!'(),-.:;? "
        symbols = [_pad] + list(_punctuations) + list(_letters)

        chars_config = {
            "pad": _pad,
            "eos": "&",
            "bos": "*",
            "characters": _letters,
            "punctuations": _punctuations,
            "phonemes": None,
            "unique": True
        }

        self.characters = chars_config
        self.custom_symbols = symbols
        # ======================================



        # self.batch_group_size = batch_group_size
        self.items = [line.split("|") for line in priors_data_list] if priors_data_list is not None else meta_data
        # self.filename_to_items_mapping = None
        # self.filenames = {}
        # for item in self.items:
        #     text, wav_file, speaker_name, lang = item
        #     self.filenames[wav_file] = None
        self.filename_to_items_mapping = {}
        for item in self.items:
            text, wav_file, speaker_name, lang = item
            if lang_override is not None:
                item[-1] = lang_override
            self.filename_to_items_mapping[wav_file.split("/")[-1]] = item

        # self.items_finetune = meta_data_finetune
        self.outputs_per_step = 1
        self.sample_rate = 22050
        # self.cleaners = text_cleaner
        # self.compute_linear_spec = compute_linear_spec
        # self.return_wav = return_wav
        # self.compute_f0 = compute_f0
        # self.f0_cache_path = f0_cache_path
        self.min_seq_len = min_seq_len
        # self.max_seq_len = max_seq_len
        # self.characters = characters
        # self.custom_symbols = custom_symbols
        # self.add_blank = add_blank
        # self.use_phonemes = use_phonemes
        # self.phoneme_cache_path = phoneme_cache_path
        # self.enable_eos_bos = enable_eos_bos
        # self.speaker_id_mapping = speaker_id_mapping
        # self.language_id_mapping = language_id_mapping
        # self.use_noise_augment = use_noise_augment

        self.language_id_mapping = {name: i for i, name in enumerate(sorted(list(lang_names.keys())))}
        # self.language_id_mapping = {name: i for i, name in enumerate(["de", "en", "it"])}

        # self.verbose = verbose
        # self.input_seq_computed = False
        # self.rescue_item_idx = 1
        # self.pitch_computed = False
        self.spec_segment_size = 32

        # setup audio processor
        self.ap = AudioProcessor(fft_size= 1024, win_length= 1024, hop_length= 256, frame_shift_ms= None, frame_length_ms= None, stft_pad_mode= "reflect", sample_rate= 22050, resample= False, preemphasis= 0.0, ref_level_db= 20, do_sound_norm= False, log_func= "np.log", do_trim_silence= True, trim_db= 45, do_rms_norm= False, db_level= None, power= 1.5, griffin_lim_iters= 60, num_mels= 80, mel_fmin= 0.0, mel_fmax= 8000.0, spec_gain= 1, do_amp_to_db_linear= False, do_amp_to_db_mel= True, signal_norm= False, min_level_db= -100, symmetric_norm= True, max_norm= 4.0, clip_norm= True, stats_path= None)

        # print("Loading text processors...")
        self.tp = {}
        for lang_code in lang_names.keys():
            base_dir = "/".join(os.path.abspath(__file__).split("\\")[:-1])+f'/text'
            self.tp[lang_code] = get_text_preprocessor(lang_code, base_dir, override_useAnyG2P=False)



        self.prepend_space_to_text = False
        self.append_space_to_text = False


        self.dataset_cache = {}
        self.dataset_cache["mel"] = {}
        self.dataset_cache["wav"] = {}
        self.dataset_cache["text"] = {}
        self.dataset_cache["pitch"] = {}
        self.dataset_cache["text_too_short"] = {}
        self.dataset_cache["text_too_long"] = {}
        self.dataset_cache["speaker_embedding"] = {}
        self.total_cached_samples = 0
        # self.MAX_CACHE_SAMPLES = 3000
        self.MAX_CACHE_SAMPLES = 30000
        self.MAX_CACHE_SAMPLES = 10000
        # self.MAX_CACHE_SAMPLES = 50000
        # self.MAX_CACHE_SAMPLES = 100000
        # self.MAX_CACHE_SAMPLES = 0

        # TODO, eventually, get the stats for the entire xVASpeech dataset
        # self.pitch_mean = torch.Tensor([214.72203]) # LJSpeech
        # self.pitch_std = torch.Tensor([65.72038]) # LJSpeech
        self.pitch_mean = torch.Tensor([104.606]) # xVASpeech
        self.pitch_std = torch.Tensor([123.4384]) # xVASpeech


        if self.args.fp_emels:
            self.stft = TacotronSTFT(filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000)



    # def load_wav(self, filename):
    #     audio = self.ap.load_wav(filename)
    #     return audio

    def calibrate_loss_sampling(self, loss_sampling_dict):
        # print("calibrate_loss_sampling")

        # if self.filename_to_items_mapping is None:
        #     self.filename_to_items_mapping = {}
        #     for item in self.items:
        #         text, wav_file, speaker_name, lang = item
        #         self.filename_to_items_mapping[wav_file.split("/")[-1]] = item
            # print(f'self.filename_to_items_mapping, {list(self.filename_to_items_mapping.keys())}')
            # print(f'self.filename_to_items_mapping, {len(self.filename_to_items_mapping)}')
            # print(f'self.items, {len(self.items)}')
            # print("self.filename_to_items_mapping", list(self.filename_to_items_mapping.keys())[:5])

        # Given a dict of losses for each filename in the dataset files list, use gaussian sampling over the centre of a list
        # of these files sorted by the loss, to get mainly the ones in the middle, leaving out samples that are already learnt
        # and also lines where the loss is high (probably badly labelled). Focus instead on the samples which have most
        # potential to actually help train the model, which will be in the middle



        # loss_sampling_dict - a dict like:  {stages: [{},{},{},..]}   where each sub-item is a dict of filename:loss, eg {file1.wav: 0.45, file2.wav: 0.21, ..}
        # loss_sampling_dict - a dict like:  {[{},{},{},..]}   where each sub-item is a dict of filename:loss, eg {file1.wav: 0.45, file2.wav: 0.21, ..}

        # files_losses = [[fname, loss_sampling_dict[stage][fname]] for fname in loss_sampling_dict[stage]]
        files_losses = [[fname, loss_sampling_dict[fname]] for fname in loss_sampling_dict.keys()]
        files_losses = sorted(files_losses, key=sort_file_loss_list)


        mu = 100
        sigma = 50
        percent_subsample = 0.5

        sample_inds_collected = []
        target_num_samples = int(len(files_losses)*percent_subsample)

        MAX_ITER = 1000000 # Prevent infinite loops if something weird happens with the data.
        iter_counter = 0
        while len(sample_inds_collected)<target_num_samples and iter_counter<MAX_ITER:
            sample_val = random.gauss(mu, sigma)
            iter_counter += 1
            if iter_counter==MAX_ITER:
                print(f'Data loss sorting calibration exceeded dataset quantity. sample_inds_collected: {len(sample_inds_collected)}/{target_num_samples}')

            if sample_val>=0 and sample_val<200:

                sample_val = sample_val/200 # Bring into 0-1 range
                sample_val = sample_val * len(files_losses) # Bring into 0-<size of data list> range

                data_list_index = int(sample_val) # Pick as an index

                if data_list_index in sample_inds_collected and iter_counter<MAX_ITER:
                    continue # Skip if this sample has already been picked


                # sample_inds_collected.append(data_list_index)
                # if files_losses[data_list_index][0] in self.filename_to_items_mapping.keys():
                #     sample_inds_collected.append(data_list_index)
                sample_inds_collected.append(data_list_index)


        # Double up, to form the sameish total dataset length
        sample_inds_collected = sample_inds_collected + sample_inds_collected


        # sampled_data = [self.audiopaths_and_text[ind] for ind in sample_inds_collected]
        sampled_data = [files_losses[ind][0] for ind in sample_inds_collected]


        items = []
        for i in range(self.args.data_mult_ft):
            items += sampled_data

        self.items = [self.filename_to_items_mapping[fname.split("/")[-1]] for fname in items]


    def load_data(self, idx):

        item = None

        # If fine-tuning a voice, randomly select new data or old data, for prior preservation against xVASpeech
        # if self.items_finetune is not None:
        #     prior_weight = 0.5
        #     if random.random() < prior_weight:
        #         data_index = random.randint(0, len(self.items_finetune)-1)
        #         try:
        #             item = self.items_finetune[data_index]
        #         except:
        #             print(f'data_index: {data_index} | {len(self.items_finetune)}')
        #             raise

        # If data is not selected from the finetuning dataset, pick one from the priors dataset
        if item is None:
            item = self.items[idx%len(self.items)]



        # if self.args.hifi_only:
        #     print(f'hifi_debug data, 1')

        text, wav_file, speaker_name, lang = item
        raw_text = text

        wav_file = wav_file.replace("\\","/")

        text = self.get_text(text, lang, wav_file)
        if len(text)==9 and text=="TOO SHORT":
            # return self.load_data(self.rescue_item_idx)
            return self.load_data(random.randint(0, len(self.items)))


        # wav = np.asarray(self.load_wav(wav_file), dtype=np.float32)
        wav = self.get_wav(wav_file)
        if wav is None:
            print("wav is None")
            return self.load_data(random.randint(0, len(self.items)))


        try:
            mel = self.ap.melspectrogram(wav).astype("float32")
        except:
            print(f'wav_file, {wav_file}')
            raise
        linear = self.ap.spectrogram(wav).astype("float32")
        # [mel, linear] = self.get_mel(wav_file, wav)

        if mel.shape[1]<self.spec_segment_size:
            # print(f'Mel shorter than self.spec_segment_size', mel.shape)
            self.dataset_cache["text_too_short"][raw_text] = "x"
            return self.load_data(random.randint(0, len(self.items)))

        # emb_path = wav_file.replace("/wavs/", "/se_embs/").replace(".wav", ".npy")
        # embedding = np.load(emb_path)
        embedding = self.get_embedding(wav_file)



        pitch = [0]
        energy = [0]
        if self.args.pitch:
            pitch = self.get_pitch(wav_file, mel.shape[-1])
            pitch = pitch[None, :]
            pitch = torch.tensor(pitch)
            # print(f'pitch, {pitch.shape}')
        if self.args.energy:
            # print(f'mel, {mel.shape}')
            # energy = np.linalg.norm(mel, ord=2, axis=0)
            energy = self.get_energy(wav_file, mel)
            energy = energy[None, :]
            energy = torch.tensor(energy)
            # print(f'energy, {energy.shape}')

        # if self.args.ow_flow:
        #     print(f'mel, {mel}')
        #     print(f'energy, {energy}')
        #     fgf()


        sample = {
            # "raw_text": raw_text,
            "text": text,
            "wav": wav,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "linear": linear,
            "d_vectors": embedding,
            "item_idx": self.items[idx%len(self.items)][1],
            "language_name": lang,
            "dataset_name": wav_file.split("/")[-3],
            "wav_file_name": os.path.basename(wav_file),
        }
        return sample

    def get_energy(self, wav_file, mel):
        if self.args.fp_emels:
            # audio, sampling_rate = load_wav_to_torch(filename)
            _, audio = read(wav_file)
            audio = torch.FloatTensor(audio.astype(np.float32))
            audio_norm = audio / 32768.0 #self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
            melspec = melspec.numpy()


            return np.linalg.norm(melspec, ord=2, axis=0)
        else:
            # if self.args.energy_magnitude:
            #     print("mel", mel, mel.shape)
            #     print("norm", np.linalg.norm(mel, ord=2, axis=0))
            #     print("sum", np.sum(mel, axis=0), np.sum(mel, axis=0).shape)
            #     print("mean", np.mean(mel, axis=0), np.mean(mel, axis=0).shape)
            #     print("max", np.max(mel, axis=0), np.max(mel, axis=0).shape)
            #     print("std", np.std(mel, axis=0), np.std(mel, axis=0).shape)
            #     fgd()
            #     return np.sum(mel, axis=0)
            # else:
            return np.linalg.norm(mel, ord=2, axis=0)

    def get_text(self, raw_text, lang, wav_file):
        if raw_text in self.dataset_cache["text_too_short"].keys():
            return "TOO SHORT"

        try:
            text = self.dataset_cache["text"][raw_text]
            return text
        except:
            # text_orig = np.asarray(
            #     text_to_sequence(
            #         raw_text,
            #         # [self.cleaners],
            #         ["multilingual_cleaners"],
            #         custom_symbols=self.custom_symbols,
            #         tp=self.characters,
            #         # add_blank=self.add_blank,
            #         add_blank=True,
            #     ),
            #     dtype=np.int32,
            # )
            # text = text_orig
            # print(f'text ORIG, {text}')




            try:
                text, _ = self.tp[lang].text_to_sequence(raw_text)
            except:
                print(f'File: {wav_file}')
                raise

            space = [self.tp[lang].ALL_SYMBOLS.index("_")]

            if self.prepend_space_to_text:
                text = space + text

            if self.append_space_to_text:
                text = text + space

            text = np.asarray(text)




        if self.total_cached_samples < self.MAX_CACHE_SAMPLES:
            self.total_cached_samples += 1
            self.dataset_cache["text"][raw_text] = text

        return text

    # def get_mel(self, filename, wav):
    #     mel_filename = filename.replace("/wavs/", "/mels/").replace(".wav", ".npy")
    #     try:
    #         # try:
    #         [mel, linear] = self.dataset_cache["mel"][mel_filename]
    #         return [mel, linear]
    #         # except:
    #         #     [mel, linear] = np.load(mel_filename, allow_pickle=True)
    #     except:
    #         # if not os.path.exists(mel_filename):
    #         mel = self.ap.melspectrogram(wav).astype("float32")

    #         linear = self.ap.spectrogram(wav).astype("float32")

    #         os.makedirs("/".join(mel_filename.split("/")[:-1]), exist_ok=True)
    #         # np.save(mel_filename, [mel, linear])
    #         # else:
    #         #     raise

    #     if self.total_cached_samples < self.MAX_CACHE_SAMPLES:
    #         self.total_cached_samples += 1
    #         self.dataset_cache["mel"][mel_filename] = [mel, linear]

    #     return [mel, linear]

    def get_wav(self, filename, retryCount=0):
        # try:
        if filename in self.dataset_cache["wav"].keys():
            wav = self.dataset_cache["wav"][filename]
            return wav
        # except:
        else:
            if not os.path.exists(filename):
                return None
            wav = self.ap.load_wav(filename)
            # if wav=="==DEL_BAD_FILE==":
            if wav is None:
                print("==DEL_BAD_FILE==")
                os.remove(filename)
                return None
            # if os.path.exists(filename):
            # else:
            #     raise

        if self.total_cached_samples < self.MAX_CACHE_SAMPLES:
            self.total_cached_samples += 1
            self.dataset_cache["wav"][filename] = wav

        return np.asarray(wav, dtype=np.float32)

    def get_embedding(self, wav_file):
        # emb_path = wav_file.replace("/wavs_postprocessed/" if self.is_ft else "/wavs/", "/se_embs/").replace(".wav", ".npy")
        emb_path = wav_file.replace("/wavs_postprocessed/", "/wavs/")
        emb_path = emb_path.replace("/wavs/", "/se_embs/").replace(".wav", ".npy")
        try:
            emb = self.dataset_cache["speaker_embedding"][emb_path]
            return emb
        except:
            emb = np.load(emb_path)

        if self.total_cached_samples < self.MAX_CACHE_SAMPLES:
            self.total_cached_samples += 1
            self.dataset_cache["speaker_embedding"][emb_path] = emb

        return emb


    # def get_pitch (self, wav_file, mel_len):
    #     pitch_path = wav_file.replace("/wavs/", "/pitch/").replace(".wav", ".npy")

    #     try:
    #         return self.dataset_cache["pitch"][pitch_path]
    #     except:
    #         try:
    #             pitch = np.load(pitch_path)
    #         except:

    #             if "allow_pickle=False" in traceback.format_exc():
    #                 os.remove(pitch_path)

    #             if not os.path.exists(pitch_path):
    #                 pitch = estimate_pitch(wav_file, mel_len, None, None)
    #                 os.makedirs("/".join(pitch_path.split("/")[:-1]), exist_ok=True)

    #                 # if self.pitch_mean is not None:
    #                 #     pitch = normalize_pitch(pitch, self.pitch_mean.numpy(), self.pitch_std.numpy())

    #                 pitch = pitch.numpy()
    #                 np.save(pitch_path, pitch)
    #             else:
    #                 # Some other dataloader process might have started but not finished writing this to file
    #                 # so loading it during that will error. Try waiting a couple of seconds, to give the other
    #                 # process some time to finish writing to file, then try again
    #                 time.sleep(4)
    #                 pitch = np.load(pitch_path)

    #         if self.pitch_mean is not None:
    #             pitch = normalize_pitch(pitch, self.pitch_mean.numpy(), self.pitch_std.numpy())
    #         if self.total_cached_samples < self.MAX_CACHE_SAMPLES:
    #             self.total_cached_samples += 1
    #             self.dataset_cache["pitch"][pitch_path] = pitch
    #         return pitch



    def sort_and_filter_items(self, by_audio_len=False):
        r"""Sort `items` based on text length or audio length in ascending order. Filter out samples out or the length
        range.

        Args:
            by_audio_len (bool): if True, sort by audio length else by text length.
        """
        lengths = np.array([len(ins[0]) for ins in self.items])

        idxs = np.argsort(lengths)
        new_items = []
        ignored_short = []
        # ignored_long = []
        for i, idx in enumerate(idxs):
            length = lengths[idx]
            # if length < self.min_seq_len or length > self.max_seq_len:
            if length < self.min_seq_len:
                ignored_short.append(idx)
            # elif  length > self.max_seq_len:
            # # if length > self.max_seq_len:
            #     ignored_long.append(idx)
            # else:
            #     new_items.append(self.items[idx])
            else:
                new_items.append(self.items[idx])
        # shuffle batch groups
        # if self.batch_group_size > 0:
        #     for i in range(len(new_items) // self.batch_group_size):
        #         offset = i * self.batch_group_size
        #         end_offset = offset + self.batch_group_size
        #         temp_items = new_items[offset:end_offset]
        #         random.shuffle(temp_items)
        #         new_items[offset:end_offset] = temp_items
        # print(f'Number of dataset samples ignored for being too short = {len(ignored_short)}, or too long = {len(ignored_long)}')
        print(f'Number of dataset samples ignored for being too short = {len(ignored_short)}')
        print(f'Final number of dataset lines: {len(new_items)}')
        self.items = new_items


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.load_data(idx)

    @staticmethod
    def _sort_batch(batch, text_lengths):
        """Sort the batch by the input text length for RNN efficiency.

        Args:
            batch (Dict): Batch returned by `__getitem__`.
            text_lengths (List[int]): Lengths of the input character sequences.
        """
        text_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor(text_lengths), dim=0, descending=True)
        batch = [batch[idx] for idx in ids_sorted_decreasing]
        return batch, text_lengths, ids_sorted_decreasing

    def collate_fn(self, batch):
        r"""
        Perform preprocessing and create a final data batch:
        1. Sort batch instances by text-length
        2. Convert Audio signal to features.
        3. PAD sequences wrt r.
        4. Load to Tofrch.
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.abc.Mapping):

            text_lengths = np.array([len(d["text"]) for d in batch])

            # sort items with text input length for RNN efficiency
            batch, text_lengths, ids_sorted_decreasing = self._sort_batch(batch, text_lengths)

            # convert list of dicts to dict of lists
            batch = {k: [dic[k] for dic in batch] for k in batch[0]}

            # get language ids from language names
            if self.language_id_mapping is not None:
                language_ids = [self.language_id_mapping[ln] for ln in batch["language_name"]]
            else:
                language_ids = None
            # get pre-computed d-vectors
            # d_vectors = []
            wav_files_names = list(batch["item_idx"])

            d_vectors = list(batch["d_vectors"])


            # get numerical speaker ids from speaker names
            # if self.speaker_id_mapping:
            #     speaker_ids = [self.speaker_id_mapping[sn] for sn in batch["speaker_name"]]
            # else:
            #     speaker_ids = None
            mel = list(batch["mel"])

            mel_lengths = [m.shape[1] for m in mel]

            # lengths adjusted by the reduction factor
            mel_lengths_adjusted = [
                m.shape[1] + (self.outputs_per_step - (m.shape[1] % self.outputs_per_step))
                if m.shape[1] % self.outputs_per_step
                else m.shape[1]
                for m in mel
            ]

            # compute 'stop token' targets
            # stop_targets = [np.array([0.0] * (mel_len - 1) + [1.0]) for mel_len in mel_lengths]

            # # PAD stop targets
            # stop_targets = prepare_stop_target(stop_targets, self.outputs_per_step)

            # PAD sequences with longest instance in the batch
            text = prepare_data(batch["text"]).astype(np.int32)

            # PAD features with longest instance
            mel = prepare_tensor(mel, self.outputs_per_step)

            # B x D x T --> B x T x D
            mel = mel.transpose(0, 2, 1)

            # convert things to pytorch
            text_lengths = torch.LongTensor(text_lengths)
            text = torch.LongTensor(text)
            # mel = torch.FloatTensor(mel).contiguous()
            mel_lengths = torch.LongTensor(mel_lengths)
            # stop_targets = torch.FloatTensor(stop_targets)

            if d_vectors is not None:
                d_vectors = torch.FloatTensor(np.array(d_vectors))


            if language_ids is not None:
                language_ids = torch.LongTensor(language_ids)


            # print(f'linear 1, {linear.shape}')
            linear = prepare_tensor(batch["linear"], self.outputs_per_step)
            # if batch["linear"][0] is not None:
            #     linear = prepare_tensor(batch["linear"], self.outputs_per_step)
            # else:
            #     linear = [self.ap.spectrogram(w).astype("float32") for w in batch["wav"]]
            #     linear = prepare_tensor(linear, self.outputs_per_step)

            # print(f'linear 2, {linear.shape}')
            linear = linear.transpose(0, 2, 1)
            linear = torch.FloatTensor(linear).contiguous()
            # else:
            #     linear = None

            # format waveforms
            wav_padded = None
            # if self.return_wav:
            wav_lengths = [w.shape[0] for w in batch["wav"]]
            max_wav_len = max(mel_lengths_adjusted) * self.ap.hop_length
            wav_lengths = torch.LongTensor(wav_lengths)
            wav_padded = torch.zeros(len(batch["wav"]), 1, max_wav_len)
            for i, w in enumerate(batch["wav"]):
                mel_length = mel_lengths_adjusted[i]
                w = np.pad(w, (0, self.ap.hop_length * self.outputs_per_step), mode="edge")
                w = w[: mel_length * self.ap.hop_length]
                wav_padded[i, :, : w.shape[0]] = torch.from_numpy(w)
            wav_padded.transpose_(1, 2)

            mel_mask = sequence_mask(mel_lengths)


            pitch_padded = torch.tensor([0])
            energy_padded = torch.tensor([0])

            if self.args.pitch:
                # Right zero-pad mel-spec

                n_formants = batch["pitch"][0].shape[0]
                # pitch_padded = torch.zeros(mel.size(0), n_formants, mel.size(2), dtype=mel.dtype)
                try:
                    max_target_len = max([x.shape[1] for x in list(batch["mel"])])
                    # pitch_padded = torch.zeros(len(wav_files_names), n_formants, batch["mel"][0].size[2], dtype=mel.dtype)
                    # pitch_padded = torch.zeros(len(wav_files_names), n_formants, max_target_len, dtype=mel.dtype)
                    pitch_padded = torch.zeros(len(wav_files_names), n_formants, max_target_len)#, dtype=mel.dtype)
                except:
                    print("mel", batch["mel"][0].size, list(batch["mel"]), wav_files_names[0])
                    raise
                # energy_padded = torch.zeros_like(pitch_padded[:, 0, :])

                # for i in range(len(ids_sorted_decreasing)):
                for i in range(len(batch["pitch"])):
                    # pitch = batch[ids_sorted_decreasing[i]]["pitch"]
                    # pitch = batch["pitch"][ids_sorted_decreasing[i]]
                    pitch = batch["pitch"][i]

                    # energy = batch[ids_sorted_decreasing[i]][4]
                    try:
                        # pitch_padded[i, :, :pitch.shape[1]] += torch.squeeze(pitch, 0)#.long()
                        pitch_padded[i, :, :pitch.shape[-1]] += torch.squeeze(pitch, 0)
                    except:
                        # print("pitch_padded", pitch_padded.shape, pitch.shape, torch.squeeze(pitch, 0).shape)
                        pitch_padded[i, :, :min(pitch.shape[-1], pitch_padded.shape[2])] += torch.squeeze(pitch, 0)[:, :pitch_padded.shape[2]]
                        # raise
                        # pitch_padded[i, :, :pitch.shape[1]] += pitch.long()

                    # energy_padded[i, :energy.shape[0]] += energy


            if self.args.energy:
                n_formants = batch["energy"][0].shape[0]
                max_target_len = max([x.shape[1] for x in list(batch["mel"])])
                energy_padded = torch.zeros(len(wav_files_names), n_formants, max_target_len)#, dtype=mel.dtype)

                for i in range(len(batch["energy"])):
                    energy = batch["energy"][i].unsqueeze(dim=0)
                    # print(f'energy_padded, {energy_padded.shape}')
                    # print(f'energy, {energy.shape}')
                    try:
                        energy_padded[i, :, :min(energy.shape[-1], energy_padded.shape[-1])] += torch.squeeze(energy, 0)
                    except:
                        energy_padded[i, :, :min(energy.shape[-1], energy_padded.shape[2])] += torch.squeeze(energy, 0)[:, :energy_padded.shape[2]]





            return {
                "text": text,
                "text_lengths": text_lengths,
                "linear": linear,
                "pitch_padded": pitch_padded,
                "energy_padded": energy_padded,
                "mel_lengths": mel_lengths,
                "mel_mask": mel_mask,
                # "stop_targets": stop_targets,
                # "item_idxs": batch["item_idx"],
                "d_vectors": d_vectors,
                "waveform": wav_padded,
                "language_ids": language_ids,
                "wav_files_names": wav_files_names,
            }

        raise TypeError(
            (
                "batch must contain tensors, numbers, dicts or lists;\
                         found {}".format(
                    type(batch[0])
                )
            )
        )
def sort_file_loss_list(x):
    return x[1]

import librosa
import torch.nn.functional as F
def estimate_pitch(wav, mel_len, normalize_mean=None, normalize_std=None, n_formants=1):

    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)


    snd, sr = librosa.load(wav)
    pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
        snd, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'), frame_length=1024)
    try:
        # assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0
        assert np.abs(mel_len - pitch_mel.shape[0]) <= 2.0
        # print("OK!")
    except:
        print("mel_len", mel_len)
        print("pitch_mel.shape", pitch_mel.shape)
        print("np.abs(mel_len - pitch_mel.shape[0])", np.abs(mel_len - pitch_mel.shape[0]))
        # raise

    pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
    pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
    pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

    if n_formants > 1:
        raise NotImplementedError


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


def read_datasets (languages, dataset_roots, extract_embs, device, data_mult=1, trainer=None, cmd_training=True, is_ft=False):

    # if extract_embs:
    if device is None:
        device = torch.device("cpu")
    model = ResNetSpeakerEncoder()
    model = model.to(device)

    metadata = []

    # is_single_dataset

    # Gather all datasets in all root directories, corresponding to the selected languages
    all_datasets = []
    for root in dataset_roots:
        # if isinstance(root, str):
        # else:
        sub_files = os.listdir(root)
        if "metadata.csv" in sub_files:
            # if root.split("_")[0] in languages:
            all_datasets.append(f'{root}')
        for fname in sub_files:
            if "." not in fname and "_" in fname and fname.split("_")[0] in languages and os.path.exists(f'{root}/{fname}/metadata.csv'):
                all_datasets.append(f'{root}/{fname}')
    # print(f'all_datasets, {len(all_datasets)}')

    # Go through each dataset's metadata.csv file, and read in the lines. Optionally extract embeddings for the wav files
    for di,dataset_path in enumerate(all_datasets):

        speaker_name = dataset_path.split("/")[-1]
        language = speaker_name.split("_")[0]

        embs_folder_exists = os.path.exists(f'{dataset_path}/se_embs')

        with open(f'{dataset_path}/metadata.csv', encoding="utf8") as f:
            lines = f.read().split("\n")[:-2]

        if not cmd_training and trainer:
            print_line = f'Reading datasets | Dataset {di+1}/{len(all_datasets)} | Items {len(lines)}     '
            trainer.training_log_live_line = print_line
            trainer.print_and_log(save_to_file=trainer.dataset_output)

        for li,line in enumerate(lines):

            if not len(line) or "|" not in line:
                continue

            if cmd_training and li%(10 if extract_embs else 1000)==0:
                print_line = f'Reading datasets | Dataset {di+1}/{len(all_datasets)} | Item {li+1}/{len(lines)}     '
                print(f'\r{print_line}', end="", flush=True)


            try:
                text = line.split("|")[1]
            except:
                print(dataset_path)
                raise
            wav_file = line.split("|")[0]
            if not wav_file.endswith(".wav"):
                wav_file = wav_file + ".wav"
            wav_path = f'{dataset_path}/{"wavs_postprocessed" if is_ft else "wavs"}/{wav_file}'

            if os.path.exists(wav_path):

                emb_path = wav_path.replace("/wavs_postprocessed/" if is_ft else "/wavs/", "/se_embs/").replace(".wav", ".npy")

                if os.path.exists(emb_path):
                    metadata.append([text, wav_path, speaker_name, language])
                else:
                    if extract_embs or not embs_folder_exists:
                        os.makedirs(f'{dataset_path}/se_embs/', exist_ok=True)
                        try:
                            embedding = model.compute_embedding(wav_path)
                            embedding = embedding.squeeze().cpu().detach().numpy()
                            np.save(emb_path, embedding)
                            metadata.append([text, wav_path, speaker_name, language])
                        except KeyboardInterrupt:
                            raise
                        except:
                            print(f'BAD: {wav_path}')

    # if extract_embs:
    del model

    if not cmd_training and trainer:
        trainer.training_log_live_line = ""

    all_metadata = []
    for _ in range(data_mult):
        all_metadata += metadata

    with open(f'{dataset_roots[0]}/.has_extracted_embs', "w+") as f: # TODO, detect dataset changes, to invalidate this? md5?
        f.write("")
    return all_metadata, len(all_datasets), data_mult


def pre_cache_g2p (dataset_roots, lang=None):

    print("Loading text loaders for g2p pre-caching...")

    languages = list(lang_names.keys())

    # Gather all datasets in all root directories, corresponding to the selected languages
    all_datasets = []
    for root in dataset_roots:
        sub_files = os.listdir(root)
        if "metadata.csv" in sub_files:
            # if root.split("_")[0] in languages:
            all_datasets.append(f'{root}/metadata.csv')
        for fname in sub_files:
            if "." not in fname and "_" in fname and fname.split("_")[0] in languages and os.path.exists(f'{root}/{fname}/metadata.csv'):
                all_datasets.append(f'{root}/{fname}/metadata.csv')

    # tp = get_text_preprocessor(lang_code, base_dir)
    tp = {}
    base_dir = "/".join(os.path.abspath(__file__).split("\\")[:-1])+f'/text'
    for lang_code in languages:
        tp[lang_code] = get_text_preprocessor(lang_code, base_dir, override_useAnyG2P=True)


    for mfi,metadata_file in enumerate(all_datasets):
        lang = metadata_file.split("/")[-2].split("_")[0] if lang is None else lang
        with open(metadata_file, "r", encoding="utf8") as f:
            lines = f.read().split("\n")
            for li,line in enumerate(lines):

                print(f'\rPre-extracting g2p | Dataset: {mfi}/{len(all_datasets)} | Line {li+1}/{len(lines)}   ', end="", flush=True)

                if "|" in line:
                    text = line.split("|")[1]
                    text, _ = tp[lang].text_to_sequence(text)