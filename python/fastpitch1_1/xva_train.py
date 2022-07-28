import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import json
import argparse
import traceback

import gc
import math
import time
import warnings
import asyncio
import glob
import re
import sys
import datetime

import torch
import numpy as np
import wave
import contextlib

# Still allow command-line use, from within this directory
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from scipy.io.wavfile import write


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    from lamb import Lamb
else:
    from python.fastpitch1_1.lamb import Lamb

try:
    sys.path.append(".")
    from resources.app.python.fastpitch1_1.fastpitch.model import FastPitch
    from resources.app.python.fastpitch1_1.common.text import text_to_sequence
except:
    from python.fastpitch1_1.fastpitch.model import FastPitch
    from python.fastpitch1_1.common.text import text_to_sequence

import torch.nn as nn
# Continue allowing access to model attributes, when using DataParallel
# https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
# class DataParallel(DistributedDataParallel):
class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)



async def handleTrainer (models_manager, data, websocket, gpus, resume=False):

    gc.collect()
    torch.cuda.empty_cache()

    if not resume:
        models_manager.sync_init_model("fastpitch1_1", websocket=websocket, gpus=[0])

        trainer = models_manager.models_bank["fastpitch1_1"]

        dataset_id = data["dataset_path"].split("/")[-1]
        dataset_output = data["output_path"] + f'/{dataset_id}'

        # trainer.init_logs(dataset_output=data["output_path"])
        trainer.init_logs(dataset_output=dataset_output)

        ckpt_fname = data["checkpoint"]

        # ======== Get the checkpoint START
        final_ckpt_fname = None
        if ckpt_fname is not None:
            if os.path.exists(f'{dataset_output}'):
                ckpts = os.listdir(f'{dataset_output}')
                ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("FastPitch_checkpoint_")]
                if len(ckpts):
                    ckpts = sorted(ckpts, key=sort_fp)
                    final_ckpt_fname = f'{dataset_output}/{ckpts[-1]}'

            if ckpt_fname=="[male]":
                final_ckpt_fname = trainer.pretrained_ckpt_male
            elif ckpt_fname=="[female]":
                final_ckpt_fname = trainer.pretrained_ckpt_female
            else:
                ckpt_is_dir = os.path.isdir(ckpt_fname)
                if final_ckpt_fname is None and ckpt_is_dir:
                    ckpts = os.listdir(f'{ckpt_fname}')
                    ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("FastPitch_checkpoint_")]

                    if len(ckpts):
                        ckpts = sorted(ckpts, key=sort_fp)
                        final_ckpt_fname = f'{ckpt_fname}/{ckpts[-1]}'

                if final_ckpt_fname is None:
                    final_ckpt_fname = ckpt_fname

        data["checkpoint"] = final_ckpt_fname
        # ======== Get the checkpoint END
    else:
        trainer = models_manager.models_bank["fastpitch1_1"]

    try:
        await trainer.start(data, gpus=gpus, resume=resume)
    except KeyboardInterrupt:
        trainer.running = False
        raise
    except RuntimeError as e:
        running = trainer.running
        trainer.running = False
        stageFinished = trainer.force_stage or trainer.model.training_stage - 1
        try:
            del trainer.train_loader
        except:
            pass
        try:
            del trainer.dataloader_iterator
            del trainer.model
            del trainer.criterion
            del trainer.trainset
            del trainer.optimizer
        except:
            pass

        gc.collect()
        torch.cuda.empty_cache()
        if "CUDA out of memory" in str(e) or "PYTORCH_CUDA_ALLOC_CONF" in str(e):
            torch.cuda.empty_cache()
            trainer.print_and_log(f'Out of VRAM')
            if running:
                trainer.print_and_log(f'============= Reducing base batch size from {trainer.batch_size} to {trainer.batch_size-3}', save_to_file=trainer.dataset_output)
                data["batch_size"] = data["batch_size"] - 3
            del trainer
            try:
                del models_manager.models_bank["fastpitch1_1"]
            except:
                pass
            if running:
                gc.collect()
                torch.cuda.empty_cache()
                return await handleTrainer(models_manager, data, websocket, gpus)

        elif trainer.JUST_FINISHED_STAGE:
            if trainer.force_stage:
                trainer.print_and_log(f'Moving to HiFi-GAN...\n', save_to_file=trainer.dataset_output)
            else:
                trainer.print_and_log(f'Finished training stage {stageFinished}...\n', save_to_file=trainer.dataset_output)
            trainer.JUST_FINISHED_STAGE = False
            trainer.is_init = False
            del trainer
            try:
                del models_manager.models_bank["fastpitch1_1"]
            except:
                pass
            gc.collect()
            if stageFinished==4 or stageFinished==5:
                models_manager.models_bank["fastpitch1_1"] = "move to hifi"
                return "move to hifi"
            else:
                # # ======================
                # # data.force_stage = 1 if "force_stage" not in data.keys() else data.force_stage
                # data["force_stage"] += 1
                # # ======================
                return await handleTrainer(models_manager, data, websocket, gpus)
        else:
            try:
                trainer.logger.info(str(e))
                del trainer
                del models_manager.models_bank["fastpitch1_1"]
            except:
                pass
            raise








class FastPitchTrainer(object):
    def __init__(self, logger, PROD, gpus, models_manager, websocket=None):
        super(FastPitchTrainer, self).__init__()

        self.logger = logger
        self.logger.info("New FastPitchTrainer")
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = torch.device(f'cuda:{gpus[0]}')
        self.ckpt_path = None
        self.websocket = websocket
        self.training_log = []

        self.model = None
        self.isReady = True

        # self.data = []
        self.epoch = None
        self.running = False
        self.is_init = False
        self.logs_are_init = False

        self.dataset_id = None
        self.dataset_input = None
        self.dataset_output = None
        self.batch_size = None
        self.force_stage = None
        self.workers = None

        self.local_rank = os.getenv('LOCAL_RANK', 0)
        self.cmudict_path = f'{"./resources/app" if self.PROD else "."}/python/fastpitch1_1/cmudict/cmudict-0.7b.txt'
        self.gpus = gpus

        self.pretrained_ckpt_male = f'{"./resources/app" if self.PROD else "."}/python/fastpitch1_1/pretrained_models/f4_nate_FastPitch_checkpoint_5760_67000.pt'
        self.pretrained_ckpt_female = f'{"./resources/app" if self.PROD else "."}/python/fastpitch1_1/pretrained_models/f4_nora_FastPitch_checkpoint_4520_65550.pt'


        self.JUST_FINISHED_STAGE = False
        self.END_OF_TRAINING = False


    def print_and_log (self, line=None, end="\n", flush=False, save_to_file=False):
        if line is None:
            print(f'\r{self.training_log_live_line}', end="", flush=True)
        else:
            time_str = str(datetime.datetime.now().time())
            time_str = time_str.split(":")[0]+":"+time_str.split(":")[1]+":"+time_str.split(":")[2].split(".")[0]
            self.training_log.append(f'{time_str} | {line}')
            print(("\r" if flush else "")+line, end=end, flush=flush)


        if save_to_file:
            with open(f'{save_to_file}/training.log', "w+") as f:
                f.write("\n".join(self.training_log+[self.training_log_live_line]))


    def load_state_dict (self, ckpt_path, sd):
        pass

    def set_device (self, device):
        pass


    async def init (self):

        if __name__ == '__main__': # Do this here, instead of top-level because __main__ can be false for both xVATrainer import, and multi-worker cmd use training
            from fastpitch.attn_loss_function import AttentionBinarizationLoss
            from fastpitch.model import FastPitch
            from fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
            from fastpitch.loss_function import FastPitchLoss
            from common.text.cmudict import CMUDict

        else:
            try:
                sys.path.append(".")
                from resources.app.python.fastpitch1_1.fastpitch.attn_loss_function import AttentionBinarizationLoss
                from resources.app.python.fastpitch1_1.fastpitch.model import FastPitch
                from resources.app.python.fastpitch1_1.fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
                from resources.app.python.fastpitch1_1.fastpitch.loss_function import FastPitchLoss
                from resources.app.python.fastpitch1_1.common.text.cmudict import CMUDict
            except:
                try:
                    from python.fastpitch1_1.fastpitch.attn_loss_function import AttentionBinarizationLoss
                    from python.fastpitch1_1.fastpitch.model import FastPitch
                    from python.fastpitch1_1.fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
                    from python.fastpitch1_1.fastpitch.loss_function import FastPitchLoss
                    from python.fastpitch1_1.common.text.cmudict import CMUDict
                except:
                    self.logger.info(traceback.format_exc())


        self.batch_to_gpu = batch_to_gpu
        self.TTSDataset = TTSDataset
        self.TTSCollate = TTSCollate


        if not os.path.exists(self.dataset_output):
            os.makedirs(self.dataset_output)
        self.init_logs(dataset_output=self.dataset_output)
        self.print_and_log(f'Dataset: {self.dataset_input}', save_to_file=self.dataset_output)
        ckpt_path = last_checkpoint(self.dataset_output)
        if ckpt_path is None:
            ckpt_path = self.checkpoint
            self.print_and_log(f'Checkpoint: {ckpt_path}', save_to_file=self.dataset_output)

        self.print_and_log(f'FP16: {"Enabled" if self.amp else "Disabled"}', save_to_file=self.dataset_output)


        # Set-up start
        np.random.seed(1234 + self.local_rank)
        torch.manual_seed(1234 + self.local_rank)
        torch.backends.cudnn.benchmark = False
        self.writer = SummaryWriter(log_dir=self.dataset_output, flush_secs=120)

        self.cmudict = CMUDict(file_or_path=self.cmudict_path)
        self.cmudict.initialize(self.cmudict_path, keep_ambiguous=True)
        # Set-up end


        self.start_iterations = 50000
        # Dataset start
        self.p_arpabet = 0.3
        num_data_lines = 0
        text_cleaners = ['english_cleaners_v2']
        dataset_file_lengths = []
        with open(self.dataset_input+"/metadata.csv") as f:
            lines = f.read().split("\n")
            for line in lines:
                if len(line.strip()):
                    fname = line.split("|")[0]
                    fname = f'{self.dataset_input}/wavs/{fname+(".wav" if not fname.endswith(".wav") else "")}'
                    if os.path.exists(fname):
                        try:
                            with contextlib.closing(wave.open(fname,'r')) as f:
                                frames = f.getnframes()
                                rate = f.getframerate()
                                duration = frames / float(rate)
                                dataset_file_lengths.append(duration)
                        except:
                            self.logger.info(f'Error opening file: {fname}')
                            raise
        num_data_lines = len(dataset_file_lengths)

        if num_data_lines>1000:
            self.start_iterations = 47500
        if num_data_lines>2000:
            self.start_iterations = 45000
        if num_data_lines>4000:
            self.start_iterations = 42500
        if num_data_lines>8000:
            self.start_iterations = 40000



        pitch_mean, pitch_std = await self.get_or_calculate_pitch_stats(self.training_log, self.dataset_input, self.cmudict, self.p_arpabet)

        self.model = FastPitch(logger=self.logger).to(self.device)
        self.attention_kl_loss = AttentionBinarizationLoss()

        # Store pitch mean/std as params to translate from Hz during inference
        self.model.pitch_mean[0] = torch.tensor(pitch_mean)
        self.model.pitch_std[0] = torch.tensor(pitch_std)

        kw = dict(lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=self.weight_decay)
        self.optimizer = Lamb(self.model.parameters(), **kw)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        # Load checkpoint
        self.model.training_stage, start_epoch, start_iter, avg_loss_per_epoch = self.load_checkpoint(ckpt_path)
        # if (self.model.training_stage==5 or self.force_stage==5) and (self.force_stage is not None or self.force_stage==5):
        if self.model.training_stage==5 and (self.force_stage is None or self.force_stage==5):
            self.END_OF_TRAINING = True
            self.JUST_FINISHED_STAGE = True
            raise
        self.avg_loss_per_epoch = avg_loss_per_epoch
        if self.force_stage:
            self.print_and_log(f'Forcing stage: {self.force_stage}', save_to_file=self.dataset_output)
            if self.model.training_stage<self.force_stage and not (self.model.training_stage==3):
                start_iter = self.start_iterations
            self.avg_loss_per_epoch = []
            self.model.training_stage = self.force_stage

        if self.websocket:
            await self.websocket.send(f'Set stage to: {self.model.training_stage} ')

            # ===================== DEVELOPING START
            # self.JUST_FINISHED_STAGE = True
            # raise
            # ===================== DEVELOPING END


        start_epoch = start_epoch
        self.total_iter = start_iter

        IS_NEW = self.dataset_id not in ckpt_path
        if IS_NEW:
            self.print_and_log("New voice", save_to_file=self.dataset_output)
            self.model.training_stage = 1
            self.total_iter = self.start_iterations
            self.avg_loss_per_epoch = []

        training_stage_bs_mult = 1

        if self.model.training_stage==1:
            self.gpus = [self.gpus[0]] # Not faster to multi-gpu; slower if anything, as it's cpu-bound
            training_stage_bs_mult = 1.5
        elif self.model.training_stage==2:
            self.epochs_per_checkpoint = self.epochs_per_checkpoint * 3
            training_stage_bs_mult = 12
        elif self.model.training_stage==3:
            training_stage_bs_mult = 3.5
        elif self.model.training_stage==4:
            training_stage_bs_mult = 4


        file_lengths_bs_mult = 10 / np.max(dataset_file_lengths)
        base_batch_size = self.batch_size
        self.batch_size = self.batch_size * training_stage_bs_mult * len(self.gpus) * file_lengths_bs_mult
        self.batch_size = max(1, int(self.batch_size))


        self.gam = max(1, round(256/(self.batch_size)))
        self.print_and_log(f'CUDA device IDs: {",".join([str(v) for v in self.gpus])}', save_to_file=self.dataset_output)
        torch.cuda.set_device(int(self.gpus[0]))

        if self.model.training_stage==1:
            self.print_and_log(f'Stage 1: Pre-training only the alignment.', save_to_file=self.dataset_output)
        elif self.model.training_stage==2:
            self.print_and_log(f'Stage 2: Pre-training durations predictor', save_to_file=self.dataset_output)
        elif self.model.training_stage==3:
            self.print_and_log(f'Stage 3: Fine-tuning pitch/energy/mel', save_to_file=self.dataset_output)
        else:
            self.print_and_log(f'Stage 4: Fine-tuning mel', save_to_file=self.dataset_output)

        self.print_and_log(f'Batch size: {self.batch_size} (Base: {base_batch_size}, Stage mult: {training_stage_bs_mult}, File lengths mult: {int(file_lengths_bs_mult*100)/100}, GPUs mult: {len(self.gpus)}) | GAM: {self.gam} -> ({self.batch_size*self.gam})', save_to_file=self.dataset_output)
        # if self.batch_size > num_data_lines*3:
        #     self.print_and_log(f'Capping batch size to {num_data_lines*3} (3 times dataset size)')
        #     self.batch_size = num_data_lines*3

        self.print_and_log(f'ARPAbet: {self.p_arpabet}', save_to_file=self.dataset_output)
        bs_multiplier = int(self.batch_size / base_batch_size)
        DATA_MULT = bs_multiplier
        DATA_MULT = max(1, min(4, DATA_MULT))
        self.EPOCH_AVG_SPAN = 20
        self.EPOCH_AVG_SPAN = int(self.EPOCH_AVG_SPAN / DATA_MULT)
        self.print_and_log(f'Data multiplier: {DATA_MULT}', save_to_file=self.dataset_output)

        self.criterion = FastPitchLoss(dur_predictor_loss_scale=self.dur_predictor_loss_scale, pitch_predictor_loss_scale=self.pitch_predictor_loss_scale, attn_loss_scale=self.attn_loss_scale, gpus=self.gpus)



        collate_fn = TTSCollate()
        collate_fn.training_stage = self.model.training_stage


        self.trainset = self.TTSDataset(dataset_path=self.dataset_input, audiopaths_and_text=self.dataset_input+"/metadata.csv", text_cleaners=text_cleaners, n_mel_channels=80, dm=DATA_MULT, pitch_mean=pitch_mean, pitch_std=pitch_std, training_stage=self.model.training_stage, p_arpabet=self.p_arpabet, max_wav_value=32768.0, sampling_rate=22050, filter_length=1024, hop_length=256, win_length=1024, mel_fmin=0, mel_fmax=8000, betabinomial_online_dir=None, pitch_online_dir=None, cmudict=self.cmudict, pitch_online_method="pyin")
        self.print_and_log(self.trainset.summary_print_str, save_to_file=self.dataset_output)

        num_data_lines = self.trainset.actual_num_lines
        self.print_and_log(f'Workers: {self.workers}', save_to_file=self.dataset_output)
        distributed_run = False
        if distributed_run:
            # train_sampler, shuffle = DistributedSampler(trainset), False
            pass # TODO
        else:
            self.train_sampler, shuffle = None, True
        self.train_loader = DataLoader(self.trainset, num_workers=self.workers, shuffle=shuffle, sampler=self.train_sampler, batch_size=self.batch_size, pin_memory=True, persistent_workers=self.workers>0, drop_last=True, collate_fn=collate_fn)
        self.model.train()


        self.epoch_frames_per_sec = []
        self.last_loss = None
        self.avg_frames_s = []

        self.target_delta = self.get_target_delta(num_data_lines)
        self.target_patience = 3
        self.target_patience_count = 0

        training_stage = self.model.training_stage
        if len(self.gpus)>1:
            self.model = DataParallel(self.model, device_ids=self.gpus)
        self.model.training_stage = training_stage

        self.graphs_json["stages"][str(self.model.training_stage)]["target_delta"] = self.target_delta

        if not os.path.exists(f'{self.dataset_input}/mean_emb.txt'):
            self.get_dataset_emb()
        if self.model.training_stage>=2 and not os.path.exists(f'{self.dataset_input}/durs_arpabet'):
            self.extract_durations(text_cleaners, pitch_mean, pitch_std)

        torch.cuda.synchronize()

        self.epoch = start_epoch
        self.num_iters = len(self.train_loader) // self.gam

        gc.collect()
        # https://github.com/pytorch/pytorch/issues/1917#issuecomment-433698337
        self.print_and_log(f'Starting training.')
        self.dataloader_iterator = iter(self.train_loader)
        self.start_new_epoch()

        if self.websocket:
            await self.websocket.send(f'Set stage to: {self.model.training_stage} ')

        self.is_init = True


    async def get_or_calculate_pitch_stats (self, training_log, dataset_dir, cmudict, p_arpabet):
        try:
            if os.path.exists(f'{dataset_dir}/pitch_stats.json'):
                with open(f'{dataset_dir}/pitch_stats.json') as f:
                    json_data = f.read()
                    data = json.loads(json_data)
                    pitch_mean = data["mean"]
                    pitch_std = data["std"]
                    self.print_and_log(f'pitch_mean: {pitch_mean} | pitch_std: {pitch_std}', save_to_file=self.dataset_output)
            else:
                self.print_and_log(f'No existing pitch mean/std stats.', save_to_file=self.dataset_output)

                pitchCalcDataset = self.TTSDataset(dataset_dir, audiopaths_and_text=self.dataset_input+"/metadata.csv", text_cleaners=[], n_mel_channels=80, dm=-1, pitch_mean=None, pitch_std=None, training_stage=-1, p_arpabet=p_arpabet, max_wav_value=32768.0, sampling_rate=22050, filter_length=1024, hop_length=256, win_length=1024, mel_fmin=0, mel_fmax=8000, betabinomial_online_dir=None, pitch_online_dir=None, cmudict=cmudict, pitch_online_method="pyin")

                collate_fn = self.TTSCollate()
                collate_fn.training_stage = -1
                pitchCalcDataloader = DataLoader(pitchCalcDataset, num_workers=1, shuffle=False, sampler=None, batch_size=1, pin_memory=True, persistent_workers=False, drop_last=True, collate_fn=collate_fn)

                pitch_vals = []

                self.print_and_log(f'Extracting baseline pitch data', save_to_file=self.dataset_output)
                for bi, batch in enumerate(pitchCalcDataloader):
                    print(f'\rExtracting baseline pitch data | {bi+1}/{len(pitchCalcDataloader)} ', end="", flush=True)
                    self.training_log_live_line = f'\rExtracting baseline pitch data | {bi+1}/{len(pitchCalcDataloader)} '
                    self.print_and_log(save_to_file=self.dataset_output)

                    (text_padded, input_lengths, mel_padded, output_lengths, len_x, pitch_padded, energy_padded, speaker, attn_prior, durs_padded, max_inp_lengths, max_mel_lengths, audiopaths) = batch
                    pitch_vals += list(pitch_padded.numpy()[0])
                self.training_log_live_line = ""

                pitch_mean = np.mean([np.mean(vals) for vals in pitch_vals])
                pitch_std = np.std([np.std(vals) for vals in pitch_vals])

                with open(f'{dataset_dir}/pitch_stats.json', "w+") as f:
                    f.write(json.dumps({"mean": float(pitch_mean), "std": float(pitch_std)}))

                self.print_and_log(f'\npitch_mean: {pitch_mean} | pitch_std: {pitch_std}', save_to_file=self.dataset_output)

                del pitchCalcDataset, pitchCalcDataloader
        except:
            self.logger.info(traceback.format_exc())
            raise

        return pitch_mean, pitch_std


    def init_logs (self, dataset_output):
        if self.logs_are_init:
            return
        if not os.path.exists(dataset_output):
            os.makedirs(dataset_output)
        self.training_log = []
        self.training_log_live_line = ""
        self.graphs_json = {
            "stages": {
                "1": {
                    "loss": [],
                    "loss_delta": []
                },
                "2": {
                    "loss": [],
                    "loss_delta": []
                },
                "3": {
                    "loss": [],
                    "loss_delta": []
                },
                "4": {
                    "loss": [],
                    "loss_delta": []
                },
                "5": {
                    "loss": [],
                    "loss_delta": []
                },
            }
        }
        if os.path.exists(f'{dataset_output}/training.log'):
            with open(f'{dataset_output}/training.log') as f:
                self.training_log = f.read().split("\n")
            self.training_log.append("\nNew Session")
        else:
            self.training_log.append(f'No {dataset_output}/training.log file found. Starting anew.')
            print(self.training_log[0])

        if os.path.exists(f'{dataset_output}/graphs.json'):
            with open(f'{dataset_output}/graphs.json') as f:
                self.graphs_json = f.read()
                self.graphs_json = json.loads(self.graphs_json)
        else:
            self.print_and_log("No graphs.json file found. Starting anew.", save_to_file=dataset_output)

        self.logs_are_init = True

    def get_target_delta(self, num_data_lines):
        target_delta = 0

        if self.model.training_stage==1:
            if num_data_lines>4000:
                target_delta = 2e-5
            elif num_data_lines>4000:
                target_delta = 5e-5
            elif num_data_lines>2000:
                target_delta = 15e-5
            elif num_data_lines>500:
                target_delta = 4e-4

            if num_data_lines<500:
                target_delta = 4e-4

            target_delta = target_delta #* 0.75

            for module in [self.model.duration_predictor, self.model.decoder, self.model.pitch_predictor, self.model.pitch_emb, self.model.energy_predictor, self.model.proj]:
                for param in module.parameters():
                    param.requires_grad = False


        elif self.model.training_stage==2:
            target_delta = 5e-4  # 0.0003

            if num_data_lines>4000:
                target_delta = 5e-5
            elif num_data_lines>2000:
                target_delta = 1e-4

            if num_data_lines<500:
                target_delta = 4e-3

            target_delta = target_delta * 1.5

            for module in [self.model.attention, self.model.decoder, self.model.pitch_predictor, self.model.pitch_emb, self.model.energy_predictor, self.model.proj]:
                for param in module.parameters():
                    param.requires_grad = False


        elif self.model.training_stage==3:
            target_delta = 6e-4

            if num_data_lines>4000:
                target_delta = 5e-5
            elif num_data_lines>2000:
                target_delta = 1e-4

            if num_data_lines<500:
                if num_data_lines<250:
                    target_delta = 2e-3
                else:
                    target_delta = 1e-3


            target_delta = target_delta * 2.5

            for module in [self.model.attention, self.model.duration_predictor]:
                for param in module.parameters():
                    param.requires_grad = False

        elif self.model.training_stage==4:
            target_delta = 25e-5 # 2e-4
            if num_data_lines>4000:
                target_delta = 35e-6 # 3e-5
            elif num_data_lines>2000:
                target_delta = 1e-4 # 8e-5

            if num_data_lines<500:
                if num_data_lines<250:
                    target_delta = 15e-4 # 1e-3
                else:
                    target_delta = 45e-5 # 4e-4

            target_delta = target_delta * 1.5
            target_delta = target_delta * 2

            for module in [self.model.attention, self.model.duration_predictor, self.model.pitch_predictor, self.model.pitch_emb, self.model.energy_predictor]:
                for param in module.parameters():
                    param.requires_grad = False


        return target_delta


    async def start (self, data, gpus=None, resume=False):
        if self.running:
            return
        self.running = True

        if not resume:
            if gpus is not None:
                self.gpus = gpus

            self.force_stage = int(data["force_stage"]) if "force_stage" in data.keys() else None

            self.dataset_input = data["dataset_path"]
            self.dataset_id = self.dataset_input.split("/")[-1]
            self.dataset_output = data["output_path"] + f'/{self.dataset_id}'
            os.makedirs(self.dataset_output, exist_ok=True)
            self.checkpoint = data["checkpoint"]

            self.workers = data["num_workers"]
            self.batch_size = data["batch_size"]
            self.epochs_per_checkpoint = data["epochs_per_checkpoint"]

            # Maybe TODO, make these configurable
            self.learning_rate = 0.1
            self.amp = data["use_amp"]=="true" if "use_amp" in data.keys() else True

            # Fixed
            self.weight_decay = 1e-6
            self.dur_predictor_loss_scale = 0.1
            self.pitch_predictor_loss_scale = 0.1
            self.attn_loss_scale = 1.0
            self.warmup_steps = 1000
            self.kl_loss_start_epoch = 0
            self.kl_loss_warmup_epochs = 100
            self.kl_loss_weight = 1
            self.grad_clip_thresh = 1000


        torch.cuda.empty_cache()
        while self.running and not self.JUST_FINISHED_STAGE and not self.END_OF_TRAINING:
            try:
                await self.iteration()
            except KeyboardInterrupt:
                raise
            except:
                err_msg = traceback.format_exc()
                if "recursion depth" in err_msg:
                    continue
                else:
                    print(err_msg)
                    raise


    def pause (self, websocket=None):
        self.logger.info("pause")
        self.running = False
        torch.cuda.empty_cache()

    # def resume (self, websocket=None):
    #     self.logger.info("xvatrain resume")
    #     torch.cuda.empty_cache()
    #     if self.running:
    #         return
    #     self.running = True
    #     self.iteration()



    def start_new_epoch (self):
        self.avg_loss_per_epoch += [0.0]
        self.epoch_frames_per_sec += [0.0]

        self.accumulated_steps = 0
        self.iter_loss = 0
        self.iter_num_frames = 0
        self.iter_meta = {}
        self.iter_start_time = None
        self.iter_losses = []

        self.epoch_iter = 0
        self.num_iters = len(self.train_loader) // self.gam


    async def iteration(self):

        if not self.is_init:
            await self.init()

        try:
            batch = next(self.dataloader_iterator)
        except StopIteration:
            # Finished epoch
            self.finish_epoch()

            self.start_new_epoch()
            self.dataloader_iterator = iter(self.train_loader)
            batch = next(self.dataloader_iterator)



        if self.accumulated_steps == 0:
            self.total_iter += 1
            self.epoch_iter += 1
            if self.iter_start_time is None:
                self.iter_start_time = time.perf_counter()

            adjust_learning_rate(self.total_iter, self.optimizer, self.learning_rate, self.warmup_steps)

            self.model.zero_grad(set_to_none=True)

        x, y, num_frames = self.batch_to_gpu(batch, training_stage=self.model.training_stage, device=self.gpus[0])
        y.append(x[-3]) # Add max_inp_lengths

        with torch.cuda.amp.autocast(enabled=self.amp):
            y_pred = self.model(x)

            loss, meta, [mel_loss, dur_loss, pitch_loss, energy_loss] = self.criterion(y_pred, y, training_stage=self.model.training_stage)

            if self.model.training_stage==1 and (self.kl_loss_start_epoch is not None and self.epoch >= self.kl_loss_start_epoch):

                _, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _, _ = y_pred
                binarization_loss = self.attention_kl_loss(attn_hard, attn_soft)
                kl_weight = min((self.epoch - self.kl_loss_start_epoch) / self.kl_loss_warmup_epochs, 1.0) * self.kl_loss_weight
                meta['kl_loss'] = binarization_loss.clone().detach() * kl_weight
                loss += kl_weight * binarization_loss
                del _, attn_soft, attn_hard

            else:
                meta['kl_loss'] = torch.zeros_like(loss)
                kl_weight = 0
                binarization_loss = 0

            loss /= self.gam

        meta = {k: v / self.gam for k, v in meta.items()}

        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.model.training_stage==3:
            reduced_loss = (pitch_loss * self.pitch_predictor_loss_scale)
        elif self.model.training_stage==4:
            reduced_loss = mel_loss
        else:
            reduced_loss = loss.item()
        reduced_num_frames = num_frames.item()

        del loss, binarization_loss, mel_loss, y_pred, x, y, batch, num_frames

        if np.isnan(reduced_loss):
            self.print_and_log(f'loss is NaN', save_to_file=self.dataset_output)

            self.model.zero_grad(set_to_none=True)
            if self.running:
                return await self.iteration()
            else:
                return

        self.iter_loss += reduced_loss
        self.iter_num_frames += reduced_num_frames
        self.iter_meta = {k: self.iter_meta.get(k, 0) + meta.get(k, 0) for k in meta}
        del meta, reduced_loss
        if self.accumulated_steps==0:
            if not self.model.training_stage==1:
                if dur_loss:
                    self.writer.add_scalar(f'loss/dur', dur_loss, self.total_iter)
                iter_mel_loss = 0 if self.model.training_stage==1 else self.iter_meta['mel_loss'].item()
                if iter_mel_loss:
                    self.writer.add_scalar(f'loss/mel', iter_mel_loss, self.total_iter)
                if pitch_loss:
                    self.writer.add_scalar(f'loss/pitch', pitch_loss, self.total_iter)
                if energy_loss:
                    self.writer.add_scalar(f'loss/energy', energy_loss, self.total_iter)

        self.accumulated_steps += 1
        del dur_loss, pitch_loss, energy_loss

        if self.accumulated_steps % self.gam == 0:

            if self.amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_thresh)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_thresh)
                self.optimizer.step()

            self.iter_time = time.perf_counter() - self.iter_start_time
            self.iter_mel_loss = 0 if self.model.training_stage==1 else self.iter_meta['mel_loss'].item()
            self.iter_kl_loss = self.iter_meta['kl_loss'].item()
            self.epoch_frames_per_sec[-1] += self.iter_num_frames / self.iter_time
            self.avg_loss_per_epoch[-1] += self.iter_loss

            acc_epoch_deltas = []
            avg_loss_per_epoch_for_printing = [val for val in self.avg_loss_per_epoch]
            avg_loss_per_epoch_for_printing[-1] /= self.epoch_iter
            for vi, val in enumerate(avg_loss_per_epoch_for_printing):
                if vi:
                    acc_epoch_deltas.append((avg_loss_per_epoch_for_printing[vi-1]-avg_loss_per_epoch_for_printing[vi])/avg_loss_per_epoch_for_printing[vi-1])

            if len(acc_epoch_deltas)>=2:
                acc_epoch_deltas_avg20 = None
                acc_epoch_deltas_avg20 = np.mean(acc_epoch_deltas if len(acc_epoch_deltas)<self.EPOCH_AVG_SPAN else acc_epoch_deltas[-self.EPOCH_AVG_SPAN:])
                avg_losses_print = int(acc_epoch_deltas_avg20*100000)/100000

                avg_losses_print = f' | Avg loss % delta: {avg_losses_print}'
            else:
                avg_losses_print = ""

            print_line = f'Stage: {self.model.training_stage} | Epoch: {self.epoch} | iter: {(self.total_iter+1)%self.num_iters}/{self.num_iters} -> {self.total_iter} | loss: {int(self.iter_loss*10000)/10000} | frames/s {int(self.iter_num_frames / self.iter_time)}{avg_losses_print} | Target: {str(self.target_delta).split("00000")[0]}    '
            self.training_log_live_line = print_line
            self.print_and_log(save_to_file=self.dataset_output)

            self.writer.add_scalar(f'loss/loss', self.iter_loss, self.total_iter)
            self.iter_losses.append(self.iter_loss)
            if self.iter_kl_loss>0:
                self.writer.add_scalar(f'loss/kl', self.iter_kl_loss, self.total_iter)

            self.writer.add_scalar(f'meta/frames/s', self.iter_num_frames / self.iter_time, self.total_iter)
            self.avg_frames_s.append(self.iter_num_frames / self.iter_time)
            self.writer.add_scalar(f'meta/lrate', self.optimizer.param_groups[0]['lr'], self.total_iter)

            self.accumulated_steps = 0
            self.iter_loss = 0
            self.iter_num_frames = 0
            self.iter_meta = {}
            self.iter_start_time = time.perf_counter()


        if self.running:
            await self.iteration()
        else:
            return



    def finish_epoch (self):
        self.epoch += 1
        self.avg_loss_per_epoch[-1] /= self.epoch_iter
        self.iter_start_time = None

        acc_epoch_deltas = []
        for vi, val in enumerate(self.avg_loss_per_epoch):
            if vi:
                acc_epoch_deltas.append((self.avg_loss_per_epoch[vi-1]-self.avg_loss_per_epoch[vi])/self.avg_loss_per_epoch[vi-1])

        acc_epoch_deltas_avg20 = None
        avg_loss = np.mean(self.iter_losses)
        if len(acc_epoch_deltas)>=2:
            acc_epoch_deltas_avg20 = np.mean(acc_epoch_deltas if len(acc_epoch_deltas)<self.EPOCH_AVG_SPAN else acc_epoch_deltas[-self.EPOCH_AVG_SPAN:])

        fpath = os.path.join(self.dataset_output, f"FastPitch_checkpoint_{self.epoch}_{self.total_iter}.pt")
        self.save_checkpoint(force_save=False, frames_s=np.mean(self.avg_frames_s), total_iter=self.total_iter, avg_loss=avg_loss, loss_delta=acc_epoch_deltas_avg20, avg_loss_per_epoch=self.avg_loss_per_epoch, fpath=fpath)

        MIN_EPOCHS = 5
        MIN_EPOCHS = 1

        self.graphs_json["stages"][str(self.model.training_stage)]["loss"].append([self.total_iter, self.avg_loss_per_epoch[-1]])
        with open(f'{self.dataset_output}/graphs.json', "w+") as f:
            f.write(json.dumps(self.graphs_json))

        if len(self.iter_losses):
            avg_iter_losses = np.mean(self.iter_losses)

            if self.last_loss:
                self.writer.add_scalar(f'meta/acc_epoch_delta', acc_epoch_deltas[-1], self.total_iter)

                if len(acc_epoch_deltas)>=2:
                    acc_epoch_deltas_avg20 = np.mean(acc_epoch_deltas if len(acc_epoch_deltas)<self.EPOCH_AVG_SPAN else acc_epoch_deltas[-self.EPOCH_AVG_SPAN:])
                    self.writer.add_scalar(f'meta/stage_{self.model.training_stage}_acc_epoch_deltas_avg20', acc_epoch_deltas_avg20, self.total_iter)

                    self.graphs_json["stages"][str(self.model.training_stage)]["loss_delta"].append([self.total_iter, acc_epoch_deltas_avg20])
                    with open(f'{self.dataset_output}/graphs.json', "w+") as f:
                        f.write(json.dumps(self.graphs_json))

                    if len(acc_epoch_deltas)>=max(MIN_EPOCHS, (20 if self.model.training_stage==2 else MIN_EPOCHS)) and acc_epoch_deltas_avg20<=self.target_delta:
                        self.target_patience_count += 1
                        if self.target_patience_count>=self.target_patience:
                            fpath_stage = os.path.join(self.dataset_output, f"Stage_{self.model.training_stage}_DONE_FastPitch_checkpoint_{self.epoch}_{self.total_iter}.pt")

                            if self.model.training_stage==4:
                                self.END_OF_TRAINING = True

                            self.JUST_FINISHED_STAGE = True
                            self.logger.info("[Trainer] JUST_FINISHED_STAGE...")
                            self.model.training_stage += 1

                            self.avg_loss_per_epoch = []
                            self.save_checkpoint(force_save=True, frames_s=np.mean(self.avg_frames_s), total_iter=(self.total_iter if self.model.training_stage==4 else self.start_iterations), avg_loss=avg_loss, loss_delta=acc_epoch_deltas_avg20, avg_loss_per_epoch=self.avg_loss_per_epoch, fpath=fpath)
                            self.save_checkpoint(force_save=True, frames_s=np.mean(self.avg_frames_s), total_iter=(self.total_iter if self.model.training_stage==4 else self.start_iterations), avg_loss=avg_loss, loss_delta=acc_epoch_deltas_avg20, avg_loss_per_epoch=self.avg_loss_per_epoch, fpath=fpath_stage, doPrintLog=False)

                            raise
                    else:
                        self.target_patience_count = 0

            self.last_loss = avg_iter_losses

        self.avg_frames_s = []


    def save_checkpoint (self, force_save=False, frames_s=0, total_iter=0, avg_loss=None, loss_delta=None, avg_loss_per_epoch=[], fpath="out.pt", doPrintLog=True):

        intermediate = (self.epochs_per_checkpoint > 0 and self.epoch % self.epochs_per_checkpoint == 0)
        if not intermediate and self.epoch < 100000:
            if not force_save:
                return

        # Remove old checkpoints to avoid using too much disk space
        old_ckpts = sorted([fname for fname in os.listdir(self.dataset_output) if fname.startswith("FastPitch_checkpoint_")], key=sort_fp)
        if len(old_ckpts)>2:
            for ckpt in old_ckpts[:-2]:
                os.remove(f'{self.dataset_output}/{ckpt}')

        # Log the epoch summary
        print_line = f'Stage: {self.model.training_stage} | Epoch: {self.epoch} | {self.dataset_output}~{self.epoch}_{self.total_iter}.pt | frames/s: {int(frames_s)}'

        if avg_loss is not None:
            print_line += f' | Loss: {int(avg_loss*100000)/100000}'
        if loss_delta is not None:
            print_line += f' | Delta: {int(loss_delta*100000)/100000}'
        print_line += f' | Target: {int(self.target_delta*100000)/100000}'

        checkpoint = {
            'epoch': self.epoch,
            'iteration': total_iter,
            'avg_loss_per_epoch': avg_loss_per_epoch,
            "training_stage": self.model.training_stage,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.amp:
            checkpoint['scaler'] = self.scaler.state_dict()
        torch.save(checkpoint, fpath)

        model_half = self.model.half()
        if len(self.gpus)>1:
            model_half = model_half.module
        torch.save(model_half.state_dict(), f'{self.dataset_output}/{self.dataset_id}.pt')
        del model_half

        self.model.float()
        self.model.load_state_dict(checkpoint["state_dict"])


        resemblyzer_emb = []
        try:
            with open(f'{self.dataset_input}/mean_emb.txt', "r") as f:
                resemblyzer_emb = f.read().split(",")
        except:
            pass

        with open(f'{self.dataset_output}/{self.dataset_id}.json', "w+") as f:
            json_data = {
                "version": "2.0",
                "modelVersion": "2.0",
                "modelType": "FastPitch1.1",
                "author": "",
                "lang": "en",
                "games": [
                    {
                        "gameId": "other",
                        "voiceId": self.dataset_id,
                        "voiceName": self.dataset_output.split("/")[-1],
                        "resemblyzer": resemblyzer_emb,
                        "gender": "male"
                    }
                ]
            }
            json.dump(json_data, f, indent=4)

        del checkpoint
        self.training_log_live_line = ""
        if doPrintLog:
            self.print_and_log(print_line+"      ", end="", flush=True, save_to_file=self.dataset_output)

    def load_checkpoint (self, filepath):
        if self.local_rank == 0:
            self.print_and_log(f'Loading model and optimizer state from {filepath}', save_to_file=self.dataset_output)
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
        except:
            self.print_and_log(f'Failed to load the checkpoint! Maybe try the second-last checkpoint (delete the last one). Full error message: {traceback.format_exc()}', save_to_file=self.dataset_output)
            raise
        try:
            epoch = checkpoint['epoch'] + 1
        except:
            epoch = 1
        try:
            total_iter = checkpoint['iteration']
        except:
            total_iter = 0

        sd = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        try:
            getattr(self.model, 'module', self.model).load_state_dict(sd)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            self.print_and_log(f'========== OPTIM NOT LOADED ==========', save_to_file=self.dataset_output)
            pass

        avg_loss_per_epoch = checkpoint["avg_loss_per_epoch"] if "avg_loss_per_epoch" in checkpoint.keys() else []

        return checkpoint["training_stage"] if "training_stage" in checkpoint.keys() else 1, epoch, total_iter, avg_loss_per_epoch


    def get_dataset_emb (self):
        from pathlib import Path
        from resemblyzer import VoiceEncoder, preprocess_wav

        files = [f'{self.dataset_input}/wavs/{file}' for file in list(os.listdir(self.dataset_input+"/wavs")) if ".wav" in file]
        self.print_and_log(f'{len(files)} files')
        embeddings = []
        files_done = []

        encoder = VoiceEncoder()

        for fi, file in enumerate(files):
            # print(f'\r Indexing data... {fi+1}/{len(files)} ', end="", flush=True)
            if fi%5==0 or fi==len(files)-1:
                self.training_log_live_line = f'Indexing data... {fi+1}/{len(files)} '
                self.print_and_log(save_to_file=self.dataset_output)

            try:
                fpath = Path(file)
                wav = preprocess_wav(fpath)
                embedding = encoder.embed_utterance(wav)
                embeddings.append(embedding)
                files_done.append(file)
            except KeyboardInterrupt:
                raise
            except:
                self.print_and_log(traceback.format_exc())

        self.training_log_live_line = ""
        mean_embedding = np.mean(embeddings, axis=0)
        std_embedding = np.std(embeddings, axis=0)
        with open(f'{self.dataset_input}/mean_emb.txt', "w+") as f:
            f.write(",".join([str(val) for val in list(mean_embedding)]))
        with open(f'{self.dataset_input}/std_emb.txt', "w+") as f:
            f.write(",".join([str(val) for val in list(std_embedding)]))

    def extract_durations(self, text_cleaners, pitch_mean, pitch_std):
        acc_model_stage = self.model.training_stage
        if len(self.gpus)>1:
            acc_model_stage = self.model.module.training_stage
            self.model.module.training_stage = 1
        else:
            acc_model_stage = self.model.training_stage
            self.model.training_stage = 1

        for p_arpabet in [1.0, 0.0]:
            self.print_and_log(f'Extracting durations from alignments ({"ARPAbet" if p_arpabet==1.0 else "text"})...', save_to_file=self.dataset_output)

            trainset = self.TTSDataset(dataset_path=self.dataset_input, audiopaths_and_text=self.dataset_input+"/metadata.csv", text_cleaners=text_cleaners, n_mel_channels=80, dm=-1, use_file_caching=True, pitch_mean=pitch_mean, pitch_std=pitch_std, training_stage=1, p_arpabet=p_arpabet, max_wav_value=32768.0, sampling_rate=22050, filter_length=1024, hop_length=256, win_length=1024, mel_fmin=0, mel_fmax=8000, betabinomial_online_dir=None, pitch_online_dir=None, cmudict=self.cmudict, pitch_online_method="pyin")
            collate_fn = self.TTSCollate()
            collate_fn.training_stage = 1
            dataloader = DataLoader(trainset, num_workers=1, shuffle=False, sampler=None, batch_size=1, pin_memory=True, persistent_workers=False, drop_last=True, collate_fn=collate_fn)

            os.makedirs(f'{self.dataset_input}/durs_arpabet', exist_ok=True)
            os.makedirs(f'{self.dataset_input}/durs_text', exist_ok=True)

            for bi, batch in enumerate(dataloader):
                if bi%5==0 or bi==len(dataloader)-1:
                    self.training_log_live_line = f'\r{bi+1}/{len(dataloader)} '
                    self.print_and_log(save_to_file=self.dataset_output)

                with torch.no_grad():
                    x, y, num_frames = self.batch_to_gpu(batch, training_stage=self.model.training_stage, device=self.gpus[0])

                    try:
                        output = self.model(x)
                        (_None, _None, _None, _None, _None, _None, _None, _None, _, _, attn_hard_dur, _, _) = output

                        for ai, audiopath in enumerate(x[-1]):
                            out_path = audiopath.split("/")[-1].replace(".wav", ".pt").replace("\\", "/").replace("/wavs/", "/durs_arpabet/" if p_arpabet==1.0 else "/durs_text/")
                            durs = attn_hard_dur[ai].squeeze().cpu().detach().numpy()
                            np.save(out_path.replace(".pt", ".npy"), durs)
                    except:
                        self.logger.info(f'Failed for: {x[-1]}')
                        self.print_and_log(f'Failed for: {x[-1]}', save_to_file=self.dataset_output)
                        self.print_and_log(traceback.format_exc(), save_to_file=self.dataset_output)
                        raise
            self.training_log_live_line = ""


        if len(self.gpus)>1:
            self.model.module.training_stage = acc_model_stage
        else:
            self.model.training_stage = acc_model_stage
        torch.cuda.empty_cache()



class FastPitch1_1(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(FastPitch1_1, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.arpabet_dict = {}

        torch.backends.cudnn.benchmark = True

        self.init_model("english_basic")
        self.isReady = True

    def init_model (self, symbols_alphabet):
        self.symbols_alphabet = symbols_alphabet
        self.model = FastPitch(logger=self.logger)
        self.model.eval()
        self.model.device = self.device

    def load_state_dict (self, ckpt_path, ckpt, n_speakers=1):

        self.ckpt_path = ckpt_path

        with open(ckpt_path.replace(".pt", ".json"), "r") as f:
            data = json.load(f)
            if "symbols_alphabet" in data.keys() and data["symbols_alphabet"]!=self.symbols_alphabet:
                self.logger.info(f'Changing symbols_alphabet from {self.symbols_alphabet} to {data["symbols_alphabet"]}')
                self.init_model(data["symbols_alphabet"])

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        self.model.load_state_dict(ckpt, strict=False)
        self.model = self.model.float()
        self.model.eval()

    def infer(self, plugin_manager, text, output, vocoder, speaker_i, pace=1.0, pitch_data=None, old_sequence=None, globalAmplitudeModifier=None):

        sampling_rate = 22050
        text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s\(\)\[\]0-9\?\.\,\!\'\{\}]+', '', text)
        text = text.replace("(", "").replace(")", "")
        sequence = text_to_sequence(text, "english_basic", ['english_cleaners'])
        text = torch.LongTensor(sequence)
        text = pad_sequence([text], batch_first=True).to(self.models_manager.device)

        with torch.no_grad():
            mel, mel_lens, _, _, _ = self.model.infer(text, pace=pace)

            y_g_hat = self.models_manager.models("infer_hifigan").model(mel)
            audio = y_g_hat.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype('int16')
            write(output, sampling_rate, audio)
            del audio

            del mel, mel_lens, _

        return ""


def sort_fp (x):
    return int(x.split("FastPitch_checkpoint_")[-1].split(".")[0].split("_")[0])

def last_checkpoint(dataset_output):
    final_ckpt_fname = None
    # If the output directory already exists
    if os.path.exists(f'{dataset_output}'):
        ckpts = os.listdir(f'{dataset_output}')
        ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("FastPitch_checkpoint_")]
        # Pick the latest checkpoint from the output directory
        if len(ckpts):
            ckpts = sorted(ckpts, key=sort_fp)
            final_ckpt_fname = f'{dataset_output}/{ckpts[-1]}'
    # Return nothing, if one is not found
    return final_ckpt_fname

def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate * scale



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Training', allow_abbrev=False)
    parser.add_argument('-gpus', type=str, default=f'0', help='CUDA devices')
    args, _ = parser.parse_known_args()

    print(f'args.gpus, {args.gpus}')
    gpus = [int(val) for val in args.gpus.split(",")]




    async def do_next_dataset_or_stage ():

        dataset_pairs = []

        dataset_pairs.append(["gtavc_victor", "f4_nate"])
        dataset_pairs.append(["sk_frea", "f4_nora"])


        TRAINER = None
        err_counter = 0

        while len(dataset_pairs):
            torch.cuda.empty_cache()

            TRAINER = FastPitchTrainer(None, PROD=False, device=gpus, models_manager=None)
            TRAINER.cmudict_path = f'./cmudict/cmudict-0.7b.txt'

            bs = 32 #args.batch_size



            data_folder = dataset_pairs[0][0]
            print(f'')
            ckpt_fname = dataset_pairs[0][1]
            force_stage = dataset_pairs[0][2] if len(dataset_pairs[0])==3 else None
            if force_stage: # Remove the stage forcing
                dataset_pairs[0] = [dataset_pairs[0][0], dataset_pairs[0][1]]

            if "/" not in ckpt_fname:
                ckpt_fname = f'D:/FP_OUTPUT/{ckpt_fname}'

            output_path = f'D:/FP_OUTPUT/{data_folder}'
            dataset_path = f'D:/FP_INPUT/{data_folder}'

            # ======== Get the checkpoint START
            final_ckpt_fname = None
            if os.path.exists(f'{output_path}'):
                ckpts = os.listdir(f'{output_path}')
                ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("FastPitch_checkpoint_")]
                if len(ckpts):
                    ckpts = sorted(ckpts, key=sort_fp)
                    final_ckpt_fname = f'{output_path}/{ckpts[-1]}'
            ckpt_is_dir = os.path.isdir(ckpt_fname)
            if final_ckpt_fname is None and ckpt_is_dir:
                ckpts = os.listdir(f'{ckpt_fname}')
                ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("FastPitch_checkpoint_")]

                if len(ckpts):
                    ckpts = sorted(ckpts, key=sort_fp)
                    final_ckpt_fname = f'{ckpt_fname}/{ckpts[-1]}'

            if final_ckpt_fname is None:
                final_ckpt_fname = ckpt_fname
            # ======== Get the checkpoint END


            try:
                init_data = {}
                init_data["force_stage"] = force_stage
                init_data["dataset_path"] = dataset_path
                init_data["output_path"] = output_path
                init_data["checkpoint"] = final_ckpt_fname
                # init_data["num_workers"] = 24
                init_data["num_workers"] = 16
                init_data["batch_size"] = bs
                init_data["epochs_per_checkpoint"] = 5

                print("start training")
                await TRAINER.start(init_data, gpus=gpus)
                print("end training")

            except KeyboardInterrupt:
                raise
            except RuntimeError as e:
                TRAINER.running = False
                if "CUDA out of memory" in str(e):
                    TRAINER.print_and_log(TRAINER.training_log, f'============= Reducing batch size from {bs} to {bs-10}')
                    bs -= 10
                elif TRAINER.JUST_FINISHED_STAGE:
                    TRAINER.print_and_log(TRAINER.training_log, "===== Finished pre-training...")
                    TRAINER.JUST_FINISHED_STAGE = False

                elif TRAINER.END_OF_TRAINING:
                    TRAINER.print_and_log(TRAINER.training_log, "=====Moving on...")
                    del dataset_pairs[0]
                    bs = 32 #args.batch_size
                    TRAINER.END_OF_TRAINING = False
                    err_counter = 0
                else:
                    TRAINER.print_and_log(TRAINER.training_log, traceback.format_exc())
                    err_counter += 1

                if err_counter>10:
                    TRAINER.print_and_log(TRAINER.training_log, "=====Moving on...")
                    del dataset_pairs[0]
                    bs = 32 #args.batch_size
                    TRAINER.END_OF_TRAINING = False
                    err_counter = 0


    import asyncio
    try:
        asyncio.run(do_next_dataset_or_stage())
    except:
        import traceback
        print("last traceback " + traceback.format_exc())
        print("exit")
        quit()