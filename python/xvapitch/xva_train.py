import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import json
import argparse
import traceback

# import gc
import math
import time
import codecs
# import warnings
# import asyncio
# import glob
import re
import sys
import datetime

import torch
import scipy
import numpy as np
import pickle as pkl
# import wave
# import contextlib

# Still allow command-line use, from within this directory
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from scipy.io.wavfile import write


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

try:
    sys.path.append(".")
    from resources.app.python.xvapitch.model import xVAPitch
    from resources.app.python.xvapitch.util import get_language_weighted_sampler
    from resources.app.python.xvapitch.losses import VitsDiscriminatorLoss, VitsGeneratorLoss
    from resources.app.python.xvapitch.dataset import TTSDataset, read_datasets, pre_cache_g2p
    from resources.app.python.xvapitch.get_dataset_emb import get_emb
    from resources.app.python.xvapitch.training_util import make_optim, get_scheduler, format_time
    from resources.app.python.xvapitch.text import get_text_preprocessor, lang_names
except:
    try:
        from python.xvapitch.model import xVAPitch
        from python.xvapitch.util import get_language_weighted_sampler
        from python.xvapitch.losses import VitsDiscriminatorLoss, VitsGeneratorLoss
        from python.xvapitch.dataset import TTSDataset, read_datasets, pre_cache_g2p
        from python.xvapitch.get_dataset_emb import get_emb
        from python.xvapitch.training_util import make_optim, get_scheduler, format_time
        from python.xvapitch.text import get_text_preprocessor, lang_names
    except:
        from model import xVAPitch
        from util import get_language_weighted_sampler
        from losses import VitsDiscriminatorLoss, VitsGeneratorLoss
        from dataset import TTSDataset, read_datasets, pre_cache_g2p
        from get_dataset_emb import get_emb
        from training_util import make_optim, get_scheduler, format_time
        from text import get_text_preprocessor, lang_names

    # from python.xvapitch.fastpitch.model import FastPitch
    # from python.xvapitch.common.text import text_to_sequence

def is_apex_available():
    import importlib
    return importlib.util.find_spec("apex") is not None

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

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    if not resume:
        models_manager.sync_init_model("xvapitch", websocket=websocket, gpus=[0])

        trainer = models_manager.models_bank["xvapitch"]

        dataset_id = data["dataset_path"].split("/")[-1]
        dataset_output = data["output_path"] + f'/{dataset_id}'

        trainer.init_logs(dataset_output=dataset_output)

        ckpt_fname = data["checkpoint"]

        # ======== Get the checkpoint START
        final_ckpt_fname = None
        if ckpt_fname is not None:
            if os.path.exists(f'{dataset_output}'):
                ckpts = os.listdir(f'{dataset_output}')
                ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("xVAPitch_")]
                if len(ckpts):
                    ckpts = sorted(ckpts, key=sort_xvap)
                    final_ckpt_fname = f'{dataset_output}/{ckpts[-1]}'

            if final_ckpt_fname is None:
                if ckpt_fname=="[base]":
                    final_ckpt_fname = trainer.pretrained_ckpt
                else:
                    ckpt_is_dir = os.path.isdir(ckpt_fname)
                    if final_ckpt_fname is None and ckpt_is_dir:
                        ckpts = os.listdir(f'{ckpt_fname}')
                        ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("xVAPitch_")]

                        if len(ckpts):
                            ckpts = sorted(ckpts, key=sort_xvap)
                            final_ckpt_fname = f'{ckpt_fname}/{ckpts[-1]}'

                    if final_ckpt_fname is None:
                        final_ckpt_fname = ckpt_fname


        data["checkpoint"] = final_ckpt_fname
        # ======== Get the checkpoint END
    else:
        trainer = models_manager.models_bank["xvapitch"]

    try:
        await trainer.start(data, gpus=gpus, resume=resume)
    except KeyboardInterrupt:
        trainer.running = False
        raise
    except RuntimeError as e:
        running = trainer.running
        trainer.running = False

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

        # gc.collect()
        torch.cuda.empty_cache()
        if "CUDA out of memory" in str(e) or "PYTORCH_CUDA_ALLOC_CONF" in str(e) or "CUDA error: out of memory" in str(e) and trainer.batch_size>3:
            torch.cuda.empty_cache()
            trainer.print_and_log(f'Out of VRAM', save_to_file=trainer.dataset_output)

            DO_LOWER_BATCHSIZE_REATTEMPT = False # This doesn't seem to work. I guess the cache isn't being cleared properly
            if DO_LOWER_BATCHSIZE_REATTEMPT:
                if running:
                    trainer.print_and_log(f'============= Reducing base batch size from {trainer.batch_size} to {trainer.batch_size-3}', save_to_file=trainer.dataset_output)
                    data["batch_size"] = data["batch_size"] - 3
                del trainer
                try:
                    del models_manager.models_bank["xvapitch"]
                except:
                    pass
                if running:
                    # gc.collect()
                    torch.cuda.empty_cache()
                    return await handleTrainer(models_manager, data, websocket, gpus)
            else:
                raise


        elif trainer.JUST_FINISHED_STAGE:
            # if trainer.force_stage:
            #     trainer.print_and_log(f'Moving to HiFi-GAN...\n', save_to_file=trainer.dataset_output)
            # else:
            stageFinished = trainer.force_stage or trainer.model.training_stage - 1
            trainer.print_and_log(f'Finished training stage {stageFinished}...\n', save_to_file=trainer.dataset_output)
            trainer.JUST_FINISHED_STAGE = False
            trainer.is_init = False
            del trainer
            try:
                del models_manager.models_bank["xvapitch"]
            except:
                pass
            # gc.collect()
            # if stageFinished==4 or stageFinished==5:
            #     models_manager.models_bank["xvapitch"] = "move to hifi"
            #     return "move to hifi"
            # else:
            return #await handleTrainer(models_manager, data, websocket, gpus)
        else:
            try:
                trainer.logger.info(str(e))
                del trainer
                del models_manager.models_bank["xvapitch"]
            except:
                pass
            raise








class xVAPitchTrainer(object):
    def __init__(self, logger, PROD, gpus, models_manager, websocket=None, amp=None, cmd_training=False):
        super(xVAPitchTrainer, self).__init__()

        self.logger = logger
        if self.logger is not None:
            self.logger.info("New xVAPitchTrainer")
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = torch.device(f'cuda:{gpus[0]}')
        self.ckpt_path = None
        self.websocket = websocket
        self.training_log = []

        self.model = None
        self.isReady = True

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
        self.amp = amp

        self.local_rank = os.getenv('LOCAL_RANK', 0)
        self.gpus = gpus
        self.pretrained_ckpt = f'{"./resources/app" if self.PROD else "."}/python/xvapitch/pretrained_models/xVAPitch_5820651.pt'
        self.cmd_training = cmd_training
        if self.cmd_training:
            self.pretrained_ckpt = f'./pretrained_models/xVAPitch_5820651.pt'

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
            with open(f'{save_to_file}/training.log', "w+", encoding="utf") as f:
                f.write("\n".join(self.training_log+[self.training_log_live_line]))


    def load_state_dict (self, ckpt_path, sd):
        pass

    def set_device (self, device):
        pass


    async def init (self):

        # if __name__ == '__main__': # Do this here, instead of top-level because __main__ can be false for both xVATrainer import, and multi-worker cmd use training
        #     from fastpitch.attn_loss_function import AttentionBinarizationLoss
        #     from fastpitch.model import FastPitch
        #     from fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
        #     from fastpitch.loss_function import FastPitchLoss
        #     from common.text.cmudict import CMUDict

        # else:
        #     try:
        #         sys.path.append(".")
        #         from resources.app.python.xvapitch.fastpitch.attn_loss_function import AttentionBinarizationLoss
        #         from resources.app.python.xvapitch.fastpitch.model import FastPitch
        #         from resources.app.python.xvapitch.fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
        #         from resources.app.python.xvapitch.fastpitch.loss_function import FastPitchLoss
        #         from resources.app.python.xvapitch.common.text.cmudict import CMUDict
        #     except:
        #         try:
        #             from python.xvapitch.fastpitch.attn_loss_function import AttentionBinarizationLoss
        #             from python.xvapitch.fastpitch.model import FastPitch
        #             from python.xvapitch.fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
        #             from python.xvapitch.fastpitch.loss_function import FastPitchLoss
        #             from python.xvapitch.common.text.cmudict import CMUDict
        #         except:
        #             self.logger.info(traceback.format_exc())
        # self.batch_to_gpu = batch_to_gpu
        # self.TTSDataset = TTSDataset
        # self.TTSCollate = TTSCollate

        self.save_step = 50 #100 # TODO, make a UI option for this
        self.do_loss_sorting = True # maybe TODO, add UI toggle for it?
        self.FINETUNE_WEIGHT = 20


        np.random.seed(1234 + self.local_rank)
        torch.manual_seed(1234 + self.local_rank)
        torch.cuda.set_device(int(self.gpus[0]))

        if not os.path.exists(self.dataset_output):
            os.makedirs(self.dataset_output)
        self.init_logs(dataset_output=self.dataset_output)
        self.print_and_log(f'Dataset: {self.dataset_input}', save_to_file=self.dataset_output)
        self.print_and_log(f'Language: {lang_names[self.lang]}', save_to_file=self.dataset_output)

        ckpt_path = last_checkpoint(self.dataset_output)
        if ckpt_path is None:
            ckpt_path = self.pretrained_ckpt
            self.print_and_log(f'Checkpoint: {ckpt_path}', save_to_file=self.dataset_output)


        # Set up training metadata
        args, batch_size, gam, target_bs = self.get_training_metadata(self.gpus)
        self.args = args
        self.gam = gam
        self.target_bs = target_bs

        base_batch_size = self.batch_size
        file_lengths_bs_mult = 1 # TODO, get max filelength from datasets, 10 / np.max(dataset_file_lengths)
        self.batch_size = self.batch_size * len(self.gpus) * file_lengths_bs_mult
        self.batch_size = max(1, int(self.batch_size))

        self.print_and_log(f'CUDA device IDs: {",".join([str(v) for v in self.gpus])}', save_to_file=self.dataset_output)
        self.print_and_log(f'FP16: {"Enabled" if self.amp else "Disabled"}', save_to_file=self.dataset_output)
        self.print_and_log(f'Batch size: {self.batch_size} (Base: {base_batch_size}, GPUs mult: {len(self.gpus)}) | GAM: {self.gam} -> ({self.batch_size*self.gam}) | Target: {self.target_bs}', save_to_file=self.dataset_output)
        self.print_and_log(f'Outputting model backups every {self.backup_model_every_x_ckpt} checkpoint{"s" if self.backup_model_every_x_ckpt>1 else ""}  ', save_to_file=self.dataset_output)


        # Init the model
        self.model = self.init_model(self.args, self.device, self.gpus)
        self.model.trainer = self

        # Set up the training modules
        self.writer, criterion, self.model, self.scaler, self.optimizer = self.setup_training_modules(self.args, self.model, self.dataset_output, self.device)
        self.model.criterion = criterion

        # Load the model checkpoints
        epoch, total_steps_done, avg_disc_loss_per_epoch, avg_disc_loss_per_epoch_deltas = self.load_checkpoint(ckpt_path)
        IS_NEW = self.dataset_id not in ckpt_path
        if IS_NEW:
            self.print_and_log("New voice", save_to_file=self.dataset_output)
            self.model.training_stage = 1
        if self.force_stage:
            self.model.training_stage = self.force_stage
            self.print_and_log(f'Forcing stage: {self.force_stage} ', save_to_file=self.dataset_output)
        self.epoch = epoch
        self.total_steps_done = total_steps_done
        self.avg_disc_loss_per_epoch = avg_disc_loss_per_epoch
        self.avg_disc_loss_per_epoch_deltas = avg_disc_loss_per_epoch_deltas

        # Pre-process the audio, if it hasn't been done already
        await self.preprocess_audio()

        # Set up the dataloaders
        priors_datasets_root = "./PRIORS" if self.cmd_training else f'{"./resources/app" if self.PROD else "."}/python/xvapitch/PRIORS'
        self.print_and_log(f'Workers: {self.workers}', save_to_file=self.dataset_output)
        self.train_loader, self.finetune_loader, batch_num_steps, ft_dataset_num_files = self.setup_dataloaders(self.args, priors_datasets_root, self.device)
        self.target_deltas = self.get_target_delta(ft_dataset_num_files)


        self.loss_analysis_dict = {}
        if self.args.analyze_loss and os.path.exists(f'{self.dataset_output}/loss_analysis.pkl'):
            with open(f'{self.dataset_output}/loss_analysis.pkl', "rb") as f:
                self.loss_analysis_dict = pkl.load(f)


        # Get the dataset embeddings
        ft_dataset_emb, other_centroids = get_emb(f'{self.dataset_input}/se_embs', f'{self.dataset_output}/emb.txt', f'{self.dataset_output}/other_embs.txt')
        self.ft_dataset_emb = ft_dataset_emb
        self.other_centroids = other_centroids

        self.scheduler = get_scheduler(self.args, self.model, self.optimizer, self.total_steps_done)

        if self.websocket:
            await self.websocket.send(f'Set stage to: {self.model.training_stage} ')

        torch.cuda.set_device(int(self.gpus[0]))
        self.model.text_encoder.emb = self.model.text_encoder.emb.to(self.device)

        if self.do_loss_sorting:
            self.loss_sampling_dict = self.init_data_losses(f'{self.dataset_output}/loss_sampling_dict.pkl', self.device)


        if self.model.training_stage==1:
            self.print_and_log(f'Stage 1: Warming up the training via text processing training.', save_to_file=self.dataset_output)
        elif self.model.training_stage==2:
            self.print_and_log(f'Stage 2: Full training', save_to_file=self.dataset_output)
        elif self.model.training_stage==3:
            self.print_and_log(f'Stage 3: [Training finished] Extra training time with no auto-stop', save_to_file=self.dataset_output)




        # TODO, explore this - if batch size is larger than dataset, is that just wasted compute due to averaged gradients?
        # if self.batch_size > num_data_lines:
        #     self.batch_size = num_data_lines



        self.target_patience = 3
        self.target_patience_count = 0

        training_stage = self.model.training_stage
        if len(self.gpus)>1:
            self.model = DataParallel(self.model, device_ids=self.gpus)
        self.model.training_stage = training_stage
        self.model.emb_l = self.model.emb_l.to(self.device)

        self.graphs_json["stages"]["1"]["target_delta"] = round(self.target_deltas[0]*100, 3) # Make the number bigger, as it's easier to read. The actual value doesn't matter as long as it maintains relative comparison to the target
        self.graphs_json["stages"]["2"]["target_delta"] = round(self.target_deltas[1]*100, 3) # Make the number bigger, as it's easier to read. The actual value doesn't matter as long as it maintains relative comparison to the target


        torch.cuda.synchronize()

        self.print_and_log(f'Starting training.')
        self.priors_iterator = iter(self.train_loader)
        self.finetune_iterator = iter(self.finetune_loader)
        self.ckpt_start_time = None
        self.step_start_time = None
        self.accumulated_steps = 0
        self.gam_num_frames = 0
        self.finetune_counter = 0
        self.training_iters = 0
        self.start_new_epoch()

        if self.websocket:
            await self.websocket.send(f'Set stage to: {self.model.training_stage} ')

        self.is_init = True



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
            }
        }
        if os.path.exists(f'{dataset_output}/training.log'):
            with open(f'{dataset_output}/training.log', encoding="utf8") as f:
                self.training_log = f.read().split("\n")
            time_str = str(datetime.datetime.now().time())
            time_str = time_str.split(":")[0]+":"+time_str.split(":")[1]+":"+time_str.split(":")[2].split(".")[0]
            self.training_log.append(f'\n{time_str} | New Session')
        else:
            self.training_log.append(f'No {dataset_output}/training.log file found. Starting anew.')
            print(self.training_log[0])

        if os.path.exists(f'{dataset_output}/graphs.json'):
            with open(f'{dataset_output}/graphs.json', encoding="utf8") as f:
                self.graphs_json = f.read()
                self.graphs_json = json.loads(self.graphs_json)
        else:
            self.print_and_log("No graphs.json file found. Starting anew.", save_to_file=dataset_output)

        self.logs_are_init = True

    def get_target_delta(self, num_data_lines):
        target_deltas = []

        # Stage 1
        target_deltas.append(0.04)

        # Stage 2
        # A really rough initial formula based off manually derived best stopping point for F4:Nate, and comparisons with other datasets
        # Will refine with more voices trained and evaluated
        NATE_DELTA = 0.0002
        NATE_NUMFILES = 8000
        mult = NATE_NUMFILES/num_data_lines
        if (mult-1) < 1:
            target_delta = NATE_DELTA * math.sqrt(mult)/1.5
        else:
            target_delta = NATE_DELTA * math.sqrt((mult-1))/1.5
        target_delta *= 0.5
        target_deltas.append(target_delta)

        return target_deltas


    async def start (self, data, gpus=None, resume=False):
        if self.running:
            return
        self.running = True

        if not resume:
            if gpus is not None:
                self.gpus = gpus

            if self.logger:
                self.logger.info(f'self.gpus: {self.gpus}')

            self.force_stage = int(data["force_stage"]) if "force_stage" in data.keys() else None

            self.dataset_input = data["dataset_path"]
            self.dataset_id = self.dataset_input.split("/")[-1]
            self.dataset_output = data["output_path"] + f'/{self.dataset_id}'
            os.makedirs(self.dataset_output, exist_ok=True)
            self.checkpoint = data["checkpoint"]

            self.workers = data["num_workers"]
            self.batch_size = data["batch_size"]
            self.lang = data["lang"]
            self.backup_model_every_x_ckpt = data["bkp_every_x"] # 2 # TODO, config
            self.backup_model_counter = 0

            # self.epochs_per_checkpoint = data["epochs_per_checkpoint"]

            # Maybe TODO, make these configurable
            self.learning_rate = 0.000175
            self.amp = data["use_amp"]=="true" if "use_amp" in data.keys() else True


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
        if self.logger:
            self.logger.info("pause")
        self.running = False
        torch.cuda.empty_cache()


    def start_new_epoch (self):
        self.keep_avg_train = {
            "step_time": [],
            "loss": [],
            "loss_gen": [],
            "loss_kl": [],
            "loss_feat": [],
            "loss_mel": [],
            "loss_mel_pred": [],
            "loss_duration": [],
            "loss_disc": [],
            "frames_per_second": [],
        }

        self.steps_since_log = 0
        self.epoch_steps = 0
        self.finetune_it = True # If the next iteration should run finetuning, or priors reinforcement








    async def iteration(self):

        if not self.is_init:
            await self.init()


        # Sample the next data point, either from the finetune dataset, or the priors dataset
        # If either of the dataloaders have reached the end, re-init them
        try:
            batch = next(self.finetune_iterator if self.finetune_it else self.priors_iterator)
        except KeyboardInterrupt:
            raise
        except StopIteration:
            # Finished epoch
            if len(self.keep_avg_train["step_time"])>0:
                self.finish_epoch()
                if self.do_loss_sorting:
                    self.finetune_loader.dataset.calibrate_loss_sampling(self.loss_sampling_dict)
            self.start_new_epoch()

            self.epoch += 1

            if self.finetune_it:
                self.finetune_iterator = iter(self.finetune_loader)
            else:
                self.priors_iterator = iter(self.train_loader)
            batch = next(self.finetune_iterator if self.finetune_it else self.priors_iterator)
        self.epoch_steps += 1


        if self.ckpt_start_time is None:
            self.ckpt_start_time = time.time()
        if self.step_start_time is None:
            self.step_start_time = time.time()


        # Format data
        loss_dict = {}
        batch = self.model.format_batch(batch)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.contiguous().to(self.device, non_blocking=True)

        num_frames = torch.sum(batch["mel_lengths"])
        self.gam_num_frames += num_frames

        y_disc_cache = None
        wav_seg_disc_cache = None
        # last_loss = None

        for idx in range(2):

            optimizer = self.optimizer[idx]
            optimizer.zero_grad()

            # Forward pass and loss computation
            with torch.cuda.amp.autocast(enabled=self.amp):
                outputs, loss_dict_it = self.model(batch, idx, y_disc_cache, wav_seg_disc_cache)

                # compute losses
                if idx==0:
                    y_disc_cache = outputs["model_outputs"].detach()
                    wav_seg_disc_cache = outputs["waveform_seg"]
                else:
                    del y_disc_cache, wav_seg_disc_cache
                    y_disc_cache, wav_seg_disc_cache = None, None
                del outputs

            if idx==0 and self.do_loss_sorting and self.finetune_it:
                for di in range(len(batch["wav_file_name"])):
                    fname_item = batch["wav_file_name"][di]#.split("\\")[-1].split("/")[-1]
                    if fname_item not in self.loss_sampling_dict.keys():
                        self.loss_sampling_dict[fname_item] = loss_dict_it["per_sample_kl_loss"][di].item() + loss_dict_it["per_sample_pitch_loss"][di].item() + loss_dict_it["per_sample_mel_loss"][di].item()

            # optimizer step
            if self.amp:
                last_loss = loss_dict_it["loss"].mean()
                self.scaler.scale(last_loss).backward()
                last_loss = last_loss.item()
            else:
                last_loss = loss_dict_it["loss"].mean()
                last_loss.backward()
                # last_loss = last_loss.item()
                del last_loss

            loss_dict_it["loss"] = loss_dict_it["loss"].mean().item()

            # detach losses
            loss_dict_detached = {}
            for key, value in loss_dict_it.items():
                if isinstance(value, (int, float)):
                    loss_dict_detached[key] = value
                elif "per_sample" in key:
                    loss_dict_detached[key] = value.detach()
                else:
                    loss_dict_detached[key] = value.detach().mean().item()
            loss_dict_it = loss_dict_detached

            if loss_dict_it is not None:
                for k, v in loss_dict_it.items():
                    if k in loss_dict:
                        loss_dict[f"{k}-{idx}"] = v
                    else:
                        loss_dict[k] = v

        del loss_dict_it, loss_dict_detached

        if self.args.analyze_loss and len(self.gpus)==1:
            for sample_i in range(len(batch["wav_file_name"])):
                wav_file_name = batch["wav_file_name"][sample_i]
                dataset_name = wav_file_name.split("/")[-3]
                if dataset_name not in self.loss_analysis_dict.keys():
                    self.loss_analysis_dict[dataset_name] = {}

                per_sample_kl_loss = loss_dict["per_sample_kl_loss"][sample_i].item()
                per_sample_mel_loss = loss_dict["per_sample_mel_loss"][sample_i].item()

                self.loss_analysis_dict[dataset_name][wav_file_name] = [per_sample_kl_loss, per_sample_mel_loss]


        self.accumulated_steps += 1
        if self.accumulated_steps%self.gam==0:

            self.accumulated_steps = 0
            if self.model.training_stage=="1" or not self.finetune_it: # Don't train the vocoder and posterior when doing priors enforcement
                self.model.posterior_encoder.zero_grad()
                self.model.waveform_decoder.zero_grad()

            for idx in range(2):
                optimizer = self.optimizer[idx]

                if self.amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

            step_time = time.time() - self.step_start_time
            self.step_start_time = time.time()

            self.keep_avg_train["step_time"].append(step_time)
            self.keep_avg_train["loss"].append(loss_dict["loss"])
            self.keep_avg_train["loss_gen"].append(loss_dict["loss_gen"])
            self.keep_avg_train["loss_kl"].append(loss_dict["loss_kl"])
            self.keep_avg_train["loss_feat"].append(loss_dict["loss_feat"])
            self.keep_avg_train["loss_mel"].append(loss_dict["loss_mel"])
            self.keep_avg_train["loss_duration"].append(loss_dict["loss_duration"])
            self.keep_avg_train["loss_disc"].append(loss_dict["loss_disc"])
            self.keep_avg_train["current_lr"] = self.optimizer[0].param_groups[0]["lr"]

            frames_per_second = int(self.gam_num_frames / step_time)
            self.gam_num_frames = 0
            self.keep_avg_train["frames_per_second"].append(frames_per_second)

            self.training_iters += 1
            self.steps_since_log += 1
            if self.steps_since_log >= 21:
                self.steps_since_log = 0
                avg_fps = round(np.mean(self.keep_avg_train["frames_per_second"]),4)
                avg_loss = round(np.mean(self.keep_avg_train["loss"]),4)
                avg_loss_kl = round(np.mean(self.keep_avg_train["loss_kl"][-10:]),4)
                avg_loss_duration = round(np.mean(self.keep_avg_train["loss_duration"][-10:]),4)
                avg_loss_mel = round(np.mean(self.keep_avg_train["loss_mel"][-10:]),4)

                self.writer.add_scalar(f'loss/loss', avg_loss, self.total_steps_done)
                self.writer.add_scalar(f'loss/kl', avg_loss_kl, self.total_steps_done)
                self.writer.add_scalar(f'loss/duration', avg_loss_duration, self.total_steps_done)
                self.writer.add_scalar(f'loss/mel', avg_loss_mel, self.total_steps_done)

                self.writer.add_scalar(f'meta/frames/s', avg_fps, self.total_steps_done)
                self.writer.add_scalar(f'meta/lrate', self.optimizer[0].param_groups[0]['lr'], self.total_steps_done)

            loss_delta = 0
            avg_loss = round(np.mean(self.keep_avg_train["loss"][-10:]),4)
            frames_per_second = int(np.mean(self.keep_avg_train["frames_per_second"]))

            self.graphs_json["stages"][str(self.model.training_stage)]["loss"].append([self.total_steps_done, avg_loss])
            if (self.training_iters%self.save_step) % 10 == 0:
                with open(f'{self.dataset_output}/graphs.json', "w+", encoding="utf8") as f:
                    f.write(json.dumps(self.graphs_json))

            if self.model.training_stage<=2 and len(self.avg_disc_loss_per_epoch[self.model.training_stage-1])>1:
                adlpe = self.avg_disc_loss_per_epoch[self.model.training_stage-1] # Shorter variable name
                self.avg_disc_loss_per_epoch_deltas[self.model.training_stage-1].append((adlpe[-2]-adlpe[-1])/adlpe[-2])
                adlped = self.avg_disc_loss_per_epoch_deltas[self.model.training_stage-1] # Shorter variable name

                ckpt_AVG_SPAN = 10
                loss_delta = np.mean(adlped if len(adlped)<ckpt_AVG_SPAN else adlped[-ckpt_AVG_SPAN:])

            if self.training_iters % self.save_step == 0 and self.training_iters != 0:

                ckpt_time = time.time() - self.ckpt_start_time
                ckpt_avg_loss_disc = np.mean(self.keep_avg_train["loss_disc"]) # Can't explain it - seems the best way to gauge an automatic stopping time
                if self.model.training_stage<=2:
                    self.avg_disc_loss_per_epoch[self.model.training_stage-1].append(ckpt_avg_loss_disc)

                has_saved = False
                if loss_delta:

                    self.graphs_json["stages"][str(self.model.training_stage)]["loss_delta"].append([self.total_steps_done, round(loss_delta*100, 3)])
                    with open(f'{self.dataset_output}/graphs.json', "w+", encoding="utf8") as f:
                        f.write(json.dumps(self.graphs_json))


                    # Early stopping
                    if loss_delta < self.target_deltas[self.model.training_stage-1]:

                        self.target_patience_count += 1
                        if self.model.training_stage<3 and self.target_patience_count>=self.target_patience:

                            output_path = os.path.join(self.dataset_output, f"xVAPitch_{self.total_steps_done}.pt")

                            if self.model.training_stage==1:
                                has_saved = True
                                self.save_checkpoint(frames_s=frames_per_second, avg_loss=avg_loss, loss_delta=loss_delta, fpath=output_path, ckpt_time=ckpt_time, doPrintLog=True)
                                self.print_and_log(f'Finished Stage 1. Moving on.. \n\n', save_to_file=self.dataset_output)
                                self.print_and_log(f'\nStage 2: Full training', save_to_file=self.dataset_output)
                                self.model.training_stage = 2
                                self.target_patience_count = 0
                                loss_delta = 0
                                if self.websocket:
                                    await self.websocket.send(f'Set stage to: {self.model.training_stage} ')

                            elif self.model.training_stage==2:
                                self.END_OF_TRAINING = True

                                self.JUST_FINISHED_STAGE = True
                                if self.logger:
                                    self.logger.info("[Trainer] JUST_FINISHED_STAGE...")
                                self.model.training_stage += 1

                                has_saved = True
                                self.save_checkpoint(frames_s=frames_per_second, avg_loss=avg_loss, loss_delta=loss_delta, fpath=output_path, ckpt_time=ckpt_time, doPrintLog=True)
                                self.print_and_log(f'Finished Stage 2. Stopping training. \n\n', save_to_file=self.dataset_output)
                                raise
                    else:
                        self.target_patience_count = 0
                    round(loss_delta*100, 3)


                else:
                    self.target_patience_count = 0


                output_path = f'{self.dataset_output}/xVAPitch_{self.total_steps_done}.pt'
                self.output_samples(f'{self.dataset_output}/viz/{self.total_steps_done}')
                if not has_saved:
                    self.save_checkpoint(frames_s=frames_per_second, avg_loss=avg_loss, loss_delta=loss_delta, fpath=output_path, ckpt_time=ckpt_time, doPrintLog=True)

                if self.args.analyze_loss:
                    with open(f'{self.dataset_output}/loss_analysis.pkl', "wb+") as f:
                        pkl.dump(self.loss_analysis_dict, f)

                self.writer.flush()



            if loss_delta:
                loss_delta = round(loss_delta*100, 3)
                avg_losses_print = f' | Avg loss % delta: {loss_delta} '
                if self.model.training_stage<=2:
                    # target_delta = round(self.target_deltas[self.model.training_stage-1]*100, 3) # Make the number bigger, as it's easier to read. The actual value doesn't matter as long as it maintains relative comparison to the delta
                    target_delta = round(self.target_deltas[self.model.training_stage-1]*100, 3)
                    avg_losses_print += f'| Target: {target_delta} '
                if self.target_patience_count>0:
                    avg_losses_print += f'| Hit: {self.target_patience_count}/{self.target_patience} '

            else:
                avg_losses_print = "                                                                   "
            iter_loss = round(np.mean(self.keep_avg_train["loss_disc"][-10:]), 4) # Average over the last 10 steps' losses (use the disc loss as that's the one that the deltas operate over)
            # print_line = f'Stage: {self.model.training_stage} | Epoch: {self.epoch} | Steps: {(self.total_steps_done)} | Ckpt: {self.training_iters%self.save_step}/{self.save_step} | Loss: {iter_loss} | frames/s {frames_per_second}{avg_losses_print}   '
            print_line = f'Stage: {self.model.training_stage} | Steps: {(self.total_steps_done)} | Ckpt: {self.training_iters%self.save_step}/{self.save_step} | Loss: {iter_loss} | frames/s {frames_per_second}{avg_losses_print}   '


            self.training_log_live_line = print_line
            self.print_and_log(save_to_file=self.dataset_output)



            self.finetune_counter += 1
            self.finetune_it = True
            if self.finetune_counter>=self.FINETUNE_WEIGHT:
                self.finetune_it = False
                self.finetune_counter = 0
            self.total_steps_done += self.gam
            self.training_log_live_line = print_line
            # self.iter_start_time = time.perf_counter()

        del batch, loss_dict


        if self.running:
            await self.iteration()
        else:
            return






    def finish_epoch (self):
        avg_loss = round(np.mean(self.keep_avg_train["loss"]),4)
        avg_loss_kl = round(np.mean(self.keep_avg_train["loss_kl"]),4)
        avg_loss_duration = round(np.mean(self.keep_avg_train["loss_duration"]),4)
        avg_loss_mel = round(np.mean(self.keep_avg_train["loss_mel"]),4)

        self.writer.add_scalar(f'epoch_loss/loss', avg_loss, self.total_steps_done)
        self.writer.add_scalar(f'epoch_loss/kl', avg_loss_kl, self.total_steps_done)
        self.writer.add_scalar(f'epoch_loss/duration', avg_loss_duration, self.total_steps_done)
        self.writer.add_scalar(f'epoch_loss/mel', avg_loss_mel, self.total_steps_done)

        self.writer.flush()
        self.scheduler[0].step()
        self.scheduler[1].step()



    def save_checkpoint (self, frames_s=0, avg_loss=None, loss_delta=None, fpath="out.pt", ckpt_time=None, doPrintLog=True):

        # Clear out the oldest checkpoint(s), to only keep a rolling window of the latest few checkpoints
        old_ckpts = sorted([fname for fname in os.listdir(self.dataset_output) if fname.startswith("xVAPitch_") and " - " not in fname], key=sort_xvap)
        if len(old_ckpts)>2:
            for ckpt in old_ckpts[:-2]:
                os.remove(f'{self.dataset_output}/{ckpt}')

        # Log the epoch summary
        # print_line = f'Stage: {self.model.training_stage} | Epoch: {self.epoch} | {self.dataset_output.split("/")[-1]}~{self.total_steps_done}.pt | Time: {format_time(ckpt_time)} | frames/s: {int(frames_s)}'
        print("\r                                                                                           ", end="", flush=True)
        print("\r", end="", flush=True)
        print_line = f'Stage: {self.model.training_stage} | {self.dataset_output.split("/")[-1]}~{self.total_steps_done}.pt | Time: {format_time(ckpt_time)} | frames/s: {int(frames_s)}'

        if hasattr(self.model, "module"):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        optimizer_state = [optim.state_dict() for optim in self.optimizer]
        scaler_state = self.scaler.state_dict()


        sd = {k.replace('module.', ''): v for k, v in model_state.items()}
        sd["avg_disc_loss_per_epoch"] = self.avg_disc_loss_per_epoch
        sd["avg_disc_loss_per_epoch_deltas"] = self.avg_disc_loss_per_epoch_deltas

        checkpoint = {
            "model": sd,
            "optimizer": optimizer_state,
            "scaler": scaler_state,
            "step": self.total_steps_done,
            "epoch": self.epoch,
            "lr": self.optimizer[0].param_groups[0]["lr"],
            "date": datetime.date.today().strftime("%B %d, %Y"),
            "avg_disc_loss_per_epoch": self.avg_disc_loss_per_epoch,
            "avg_disc_loss_per_epoch_deltas": self.avg_disc_loss_per_epoch_deltas,
            "training_stage": self.model.training_stage,
        }



        if avg_loss is not None:
            print_line += f' | Loss: {(int(avg_loss*100000)/100000):.5f}'
        if loss_delta is not None and loss_delta>0:
            # print_line += f' | Delta: {(int(loss_delta*100000)/100000):.5f}'
            print_line += f' | Delta: {round(loss_delta*100, 3)}'
        if self.model.training_stage<=2 and loss_delta is not None and loss_delta>0:
            target_delta = round(self.target_deltas[self.model.training_stage-1]*100, 3) # Make the number bigger, as it's easier to read. The actual value doesn't matter as long as it maintains relative comparison to the delta
            print_line += f' | Target: {target_delta}'
            if self.target_patience_count>0:
                print_line += f' | Hit: {self.target_patience_count}/{self.target_patience} '
        else:
            print_line += "                   "

        if self.amp:
            checkpoint['scaler'] = self.scaler.state_dict()
        torch.save(checkpoint, fpath)

        model_half = self.model.half()
        if len(self.gpus)>1:
            model_half = model_half.module
        torch.save(model_half.state_dict(), f'{self.dataset_output}/{self.dataset_id}.pt')

        self.backup_model_counter += 1
        if self.backup_model_counter >= self.backup_model_every_x_ckpt:
            os.makedirs(f'{self.dataset_output}/viz/{self.total_steps_done}', exist_ok=True)
            torch.save(model_half.state_dict(), f'{self.dataset_output}/viz/{self.total_steps_done}/{self.dataset_id}.pt')
            self.backup_model_counter = 0


        torch.cuda.empty_cache()
        del model_half

        self.model.float()
        checkpoint["model"]["step"] = self.total_steps_done
        self.model.load_state_dict(checkpoint["model"], strict=False)


        with open(f'{self.dataset_output}/{self.dataset_id}.json', "w+", encoding="utf8") as f:
            json_data = {
                "version": "3.0",
                "modelVersion": "3.0",
                "modelType": "xVAPitch",
                "author": "",
                "lang": "en", # TODO, add UI setting for this
                "games": [
                    {
                        "gameId": "other",
                        "voiceId": self.dataset_id,
                        "voiceName": self.dataset_output.split("/")[-1],
                        "base_speaker_emb": list(self.ft_dataset_emb),
                        "gender": "male"
                    }
                ]
            }
            json.dump(json_data, f, indent=4)

        del checkpoint
        self.training_log_live_line = ""
        if doPrintLog:
            # self.print_and_log(print_line+"      ", end="", flush=True, save_to_file=self.dataset_output)
            self.print_and_log(print_line+"      ", end="", flush=True, save_to_file=self.dataset_output)






    def load_checkpoint (self, filepath):

        if self.local_rank == 0:
            self.print_and_log(f'Loading model and optimizer state from {filepath}', save_to_file=self.dataset_output)
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
        except:
            self.print_and_log(f'Failed to load the checkpoint! Maybe try the second-last checkpoint (delete the last one). Full error message: {traceback.format_exc()}', save_to_file=self.dataset_output)
            raise

        epoch = 0
        total_steps_done = 0
        if "step" in checkpoint.keys():
            total_steps_done = checkpoint["step"] if filepath.split("/")[-1]!=self.pretrained_ckpt.split("/")[-1] else 0
        if 'model' in checkpoint.keys():
            sd = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        else:
            sd = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.model.load_state_dict(sd, strict=False)

        if "optimizer" in checkpoint.keys():
            for idx, state in enumerate(checkpoint["optimizer"]):
                try:
                    self.optimizer[idx].load_state_dict(state)
                except:
                    print("=== OPTIM NOT LOADED ===")
        else:
            print("Loading optim from the base checkpoint")
            base_checkpoint = torch.load(self.pretrained_ckpt, map_location='cpu')
            try:
                self.optimizer[idx].load_state_dict(base_checkpoint)
            except:
                print("=== OPTIM NOT LOADED ===")

        lr = self.args.lr
        if "lr" in checkpoint.keys():
            lr = checkpoint["lr"]
        for group in self.optimizer[0].param_groups:
            group["lr"] = lr
        for group in self.optimizer[1].param_groups:
            group["lr"] = lr

        avg_disc_loss_per_epoch = checkpoint["avg_disc_loss_per_epoch"] if "avg_disc_loss_per_epoch" in checkpoint.keys() else [[],[]]
        avg_disc_loss_per_epoch_deltas = checkpoint["avg_disc_loss_per_epoch_deltas"] if "avg_disc_loss_per_epoch_deltas" in checkpoint.keys() else [[],[]]
        torch.cuda.empty_cache()

        training_stage = checkpoint["training_stage"] if "training_stage" in checkpoint.keys() else 1
        self.model.training_stage = training_stage
        self.model.emb_l = self.model.emb_l.to(self.device)

        return epoch, total_steps_done, avg_disc_loss_per_epoch, avg_disc_loss_per_epoch_deltas

    def get_argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-gpus', default="1")
        parser.add_argument("-bs", '--batch_size', type=int, default=25)
        parser.add_argument("--lr", type=float, default=0.000175)
        parser.add_argument("-dm", '--data_mult', type=int, default=1)
        parser.add_argument("-dmft", '--data_mult_ft', type=int, default=10)
        parser.add_argument("-w", '--workers', type=int, default=2)
        parser.add_argument('--continue_path', default="")
        parser.add_argument('--output_root', default="E:/XVAP_OUTPUT/")
        parser.add_argument('--out', default="debug")
        parser.add_argument('--rank', default=0)
        parser.add_argument('--pitch', type=int, default=0) # TEMP - toggle the use of pitch conditioning
        parser.add_argument('--energy', type=int, default=0) # TEMP - toggle the use of energy conditioning
        parser.add_argument('--hifi_only', type=int, default=0) # Train only the hifi decoder and posterior encoder
        parser.add_argument('--pe_scaling', type=float, default=0.2)
        parser.add_argument('--target_bs', type=float, default=400)

        parser.add_argument('--analyze_loss', type=int, default=1)
        parser.add_argument('--lion', type=int, default=0)
        parser.add_argument('--bnb', type=int, default=0)

        parser.add_argument('--mltts_rc', type=int, default=0)
        parser.add_argument('--mltts_rc_rev', type=int, default=0)
        parser.add_argument('--lang_w', type=int, default=1)
        parser.add_argument('--big', type=int, default=1)
        parser.add_argument('--flc', type=int, default=0) # Full Lang conditioning (on all modules)
        parser.add_argument('--frozen_vocoder_langs', type=int, default=0)

        parser.add_argument('--single', type=int, default=0)
        parser.add_argument('--langs_config', type=int, default=0)

        parser.add_argument('--fp_emels', type=int, default=0)
        parser.add_argument('--ow_flow', type=int, default=0)
        parser.add_argument('--expanded_flow', type=int, default=0)
        parser.add_argument('--expanded_flow_dim', type=int, default=32)
        parser.add_argument('--debug', type=int, default=0)
        parser.add_argument('--extract', type=int, default=0)
        parser.add_argument('--group_id', default="")
        parser.add_argument('--use_ddp', type=bool, default=False)
        # parser.add_argument("-fh", '--freeze_hifi', type=int, default=0)

        parser.add_argument('--vocoder', type=int, default=0)
        parser.add_argument('--ft_weight', type=int, default=20)
        parser.add_argument('--do_loss_sorting', type=int, default=1)
        parser.add_argument('--data', default="")
        args = parser.parse_args()
        return args

    def get_training_metadata(self, gpus):
        args = self.get_argparse()
        # gpus = [int(g) for g in gpus.split(",")]
        # device = torch.device(f'cuda:{gpus[0]}')
        # batch_size = args.batch_size * len(gpus)
        batch_size = self.batch_size * len(gpus)
        target_bs = args.target_bs
        gam = max(1, math.ceil(target_bs/(batch_size)))
        # print(f'Batch size | Base: {args.batch_size} | Num GPUs: {len(gpus)} | GAM: {gam} | Final: {batch_size*gam}')

        return args, batch_size, gam, target_bs

    def init_model(self, args, device, gpus):
        model = xVAPitch(args)
        model = model.to(device)
        try:
            print("No torch.compile()") # Waiting for someone to verify this works in linux first
            # model = torch.compile(model)
            pass
        except:
            print("No torch.compile()") # Waiting for windows support
            pass
        model.train()
        if len(gpus)>1:
            model = DataParallel(model, device_ids=gpus)
        return model

    def setup_dataloaders(self, args, priors_datasets_root, device):
        languages = ["de", "en", "it", "fr", "ro", "jp", "es", "ru", "ar", "da", "el", "fi", "ha", "hi", "hu", "ko", "la", "nl", "pl", "pt", "sw", "sv", "tr", "uk", "vi", "wo", "yo", "zh"]

        # Finetune dataset
        if not os.path.exists(f'{self.dataset_input}/.has_precached_g2p'):
            pre_cache_g2p([self.dataset_input], lang=self.lang)
            with open(f'{self.dataset_input}/.has_precached_g2p', "w+") as f: # TODO, detect dataset changes, to invalidate this? md5?
                f.write("")
        do_preExtract_embs = os.path.exists(f'{self.dataset_input}/.has_extracted_embs') # TODO, detect dataset changes, to invalidate this? md5?
        train_samples_finetune, _, _ = read_datasets(languages, [self.dataset_input], extract_embs=do_preExtract_embs, device=device, data_mult=args.data_mult_ft, trainer=self, cmd_training=self.cmd_training, is_ft=True)
        base_num_ft_samples = int(len(train_samples_finetune)/args.data_mult_ft)
        self.print_and_log(f'Fine-tune dataset files: {base_num_ft_samples}', save_to_file=self.dataset_output)

        # Priors datasets
        priors_datasets = [f'D:/xVASpeech/DATASETS', f'D:/xVASpeech/GAME_DATA']
        priors_datasets = [f'D:/xVASpeech/DATASETS_GOOD']
        priors_datasets = [f'D:/DATA_DEBUG']
        priors_datasets = [priors_datasets_root]

        if not os.path.exists(priors_datasets_root):
            self.print_and_log(f'Priors dataset now found at location: {priors_datasets_root}\nDid you remember to download and install the dataset?', save_to_file=self.dataset_output)
            raise

        if not os.path.exists(f'{priors_datasets_root}/.has_precached_g2p'):
            pre_cache_g2p(priors_datasets)
            # pre_cache_g2p([priors_datasets_root])
            with open(f'{priors_datasets_root}/.has_precached_g2p', "w+") as f: # TODO, detect dataset changes, to invalidate this? md5?
                f.write("")
        do_preExtract_embs = os.path.exists(f'{priors_datasets_root}/.has_extracted_embs') # TODO, detect dataset changes, to invalidate this? md5?
        train_samples, total_num_speakers, _ = read_datasets(languages, priors_datasets, extract_embs=do_preExtract_embs, device=device, data_mult=args.data_mult, trainer=self, cmd_training=self.cmd_training, is_ft=False)
        self.print_and_log(f'Priors datasets files: {len(train_samples)} | Number of datasets: {total_num_speakers}', save_to_file=self.dataset_output)


        finetune_dataset = TTSDataset(
            args,
            meta_data=train_samples_finetune,
            min_seq_len=15,
            lang_override=self.lang,
            is_ft=True
        )
        train_dataset = TTSDataset(
            args,
            meta_data=train_samples,
            min_seq_len=15,
        )
        train_dataset.sort_and_filter_items()
        finetune_dataset.sort_and_filter_items()

        sampler = get_language_weighted_sampler(train_dataset.items)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            # shuffle=False,
            shuffle=args.hifi_only,
            collate_fn=train_dataset.collate_fn,
            drop_last=True,
            sampler=sampler,
            persistent_workers=args.workers>0,
            num_workers=args.workers,
            pin_memory=args.workers>0,
        )
        finetune_loader = DataLoader(
            finetune_dataset,
            batch_size=self.batch_size,
            # shuffle=False,
            shuffle=args.hifi_only,
            collate_fn=train_dataset.collate_fn,
            # drop_last=True,
            drop_last=False,
            # sampler=sampler,
            persistent_workers=args.workers>0,
            num_workers=args.workers,
            pin_memory=args.workers>0,
        )

        batch_num_steps = int(len(train_loader.dataset) / (self.batch_size))
        return train_loader, finetune_loader, batch_num_steps, base_num_ft_samples




    def setup_training_modules(self, args, model, output_path, device):
        # TODO, add wandb support if API key is given
        writer = SummaryWriter(os.path.join(output_path, 'logs'), flush_secs=120)
        criterion = [VitsGeneratorLoss(args).to(device), VitsDiscriminatorLoss().to(device)]
        model.criterion = criterion
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        optimizers = make_optim(args, model)


        return writer, criterion, model, scaler, optimizers


    def init_data_losses(self, fname, device):
        languages = ["de", "en", "it", "fr", "ro", "jp", "es", "ru", "ar", "da", "el", "fi", "ha", "hi", "hu", "ko", "la", "nl", "pl", "pt", "sw", "sv", "tr", "uk", "vi", "wo", "yo", "zh"]
        loss_sorting_init_train_samples_finetune, _, _ = read_datasets(languages, [self.dataset_input], extract_embs=False, device=device, data_mult=1, trainer=self, cmd_training=self.cmd_training, is_ft=True)
        loss_sorting_init_finetune_dataset = TTSDataset(
            self.args,
            meta_data=loss_sorting_init_train_samples_finetune,
            min_seq_len=0,
            lang_override=self.lang
        )
        loss_sorting_init_loader = DataLoader(
            loss_sorting_init_finetune_dataset,
            batch_size=self.batch_size*4,
            shuffle=False,
            collate_fn=loss_sorting_init_finetune_dataset.collate_fn,
            drop_last=False,
            persistent_workers=False,
            num_workers=self.workers,
        )

        loss_sampling_dict = {}


        self.print_and_log(f'Initializing data losses... ({fname})', save_to_file=self.dataset_output)

        fnames_actually_in_dataset = set() # Used later to make sure there's no extra data entries not actually in the current dataset

        # Iterate through the data, to ensure all items are included
        with torch.no_grad():
            it_count = 0
            for batch in loss_sorting_init_loader:
                # print(, end="", flush=True)
                # print(f'\rInitializing data losses... {it_count}/{len(finetune_loader.dataset)} ', end="", flush=True)
                self.training_log_live_line = f'\rInitializing data losses... {it_count}/{len(loss_sorting_init_loader.dataset)} '
                self.print_and_log(save_to_file=self.dataset_output)


                batch = self.model.format_batch(batch)
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.contiguous().to(device, non_blocking=True)

                # Do the model forward
                y_disc_cache = None
                wav_seg_disc_cache = None

                # forward pass and loss computation
                with torch.cuda.amp.autocast(enabled=self.amp):
                    outputs, loss_dict_it = self.model(batch, 0, y_disc_cache, wav_seg_disc_cache)


                for di in range(len(batch["wav_file_name"])):
                    fname_item = batch["wav_file_name"][di]
                    if fname_item not in loss_sampling_dict.keys() and fname_item.split("/")[-1] in self.finetune_loader.dataset.filename_to_items_mapping.keys():
                        loss_sampling_dict[fname_item] = loss_dict_it["per_sample_kl_loss"][di].item() + loss_dict_it["per_sample_mel_loss"][di].item()

                        fnames_actually_in_dataset.add(fname_item)

                it_count += len(batch["wav_file_name"])

        # Remove any extra entries that are not actually in the dataset
        fnames_in_dict = list(loss_sampling_dict.keys())
        for dict_fname in fnames_in_dict:
            if dict_fname not in fnames_actually_in_dataset:
                del loss_sampling_dict[dict_fname]

        torch.cuda.empty_cache()
        self.save_data_losses(fname, loss_sampling_dict)
        self.training_log_live_line = ""
        self.print_and_log("Data losses initialized", save_to_file=self.dataset_output)
        print("")
        return loss_sampling_dict
    def save_data_losses (self, fname, loss_sampling_dict):
        with open(fname, "wb+") as f:
            pkl.dump(loss_sampling_dict, f)



    def output_samples(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        torch.cuda.empty_cache()

        with codecs.open(f'{"./resources/app" if self.PROD else "."}/python/xvapitch/viz_sentences.json', "r", encoding="utf8") as f:
            texts = json.load(f)

        language_id_mapping = {name: i for i, name in enumerate(sorted(list(lang_names.keys())))}

        embeddings = []
        embeddings.append(torch.tensor(self.ft_dataset_emb).squeeze().to(self.device).float()) # Add the main voice embedding style
        # Add a few more so there's 5 in total
        for emb in self.other_centroids[:4]:
            embeddings.append(torch.tensor(emb).squeeze().to(self.device).float())


        langs = list(texts.keys())
        for li,lang in enumerate(langs):

            if lang in language_id_mapping.keys():
                tp = self.train_loader.dataset.tp[lang]
                language_id = language_id_mapping[lang]

                # tp = get_text_preprocessor(lang, base_dir)
                text_inputs, _ = tp.text_to_sequence(texts[lang])

                text_inputs = torch.tensor(text_inputs).to(self.device).unsqueeze(dim=0)
                language_id_tensor = torch.tensor(language_id).to(self.device)

                for ei,emb in enumerate(embeddings):

                    print(f'\rOutputting visualization samples. Language: {li+1}/{len(langs)} | Style: {ei+1}/{len(embeddings)}  ', end="", flush=True)

                    output = self.model.infer(text_inputs, language_id_tensor, emb, pacing=1)
                    wav = output.squeeze().cpu().detach().numpy()
                    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

                    out_path = f'{output_folder}/{lang}_{ei}_tts.wav'
                    scipy.io.wavfile.write(out_path, 22050, wav_norm.astype(np.int16))

        torch.cuda.empty_cache()


    async def preprocess_audio(self):
        self.print_and_log(f'Pre-processing audio ', save_to_file=self.dataset_output)

        if os.path.exists(self.dataset_input+"/wavs_postprocessed"):
            files = os.listdir(self.dataset_input+"/wavs_postprocessed")
            for file in files:
                os.remove(f'{self.dataset_input}/wavs_postprocessed/{file}')

        data = {}
        data["inPath"] = self.dataset_input+"/wavs"
        data["outputDirectory"] = self.dataset_input+"/wavs_postprocessed"
        data["toolSettings"] = {}
        data["toolSettings"]["useMP"] = True
        os.makedirs(self.dataset_input+"/wavs_postprocessed", exist_ok=True)

        await self.models_manager.init_model("normalize", None)
        await self.models_manager.models_bank["normalize"].normalize(data, None)





class xVAPitchModel(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(xVAPitchModel, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.arpabet_dict = {}

        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False

        self.base_dir = f'{"./resources/app" if self.PROD else "."}/python/xvapitch/text'
        self.lang_tp = {}
        self.lang_tp["en"] = get_text_preprocessor("en", self.base_dir)

        self.language_id_mapping = {name: i for i, name in enumerate(sorted(list(lang_names.keys())))}

        # parser = argparse.ArgumentParser()
        # args = parser.parse_args()
        args = self.get_argparse()

        # Params from training
        args.pitch = 1
        args.pe_scaling = 0.1
        args.expanded_flow = 0
        args.ow_flow = 0
        args.energy = 0


        self.model = xVAPitch(args).to(self.device)
        self.model.eval()
        self.model.device = self.device

        self.isReady = True

    def load_state_dict (self, ckpt_path, ckpt, n_speakers=1):

        self.ckpt_path = ckpt_path

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        self.model.load_state_dict(ckpt, strict=False)
        self.model = self.model.float()
        self.model.eval()

    def infer(self, text, output, embedding):

        embedding = torch.tensor(embedding).to(self.device)

        language_id = self.language_id_mapping["en"]

        text_inputs, _ = self.lang_tp["en"].text_to_sequence(text)
        text_inputs = torch.tensor(text_inputs).to(self.device).unsqueeze(dim=0)
        language_id_tensor = torch.tensor(language_id).to(self.device)


        model_output = self.model.infer(text_inputs, language_id_tensor, embedding, pacing=1)
        wav = model_output.squeeze().cpu().detach().numpy()
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(output, 22050, wav_norm.astype(np.int16))

        torch.cuda.empty_cache()
        return ""

    def get_argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-gpus', default="1")
        parser.add_argument("-bs", '--batch_size', type=int, default=25)
        parser.add_argument("--lr", type=float, default=0.000175)
        parser.add_argument("-dm", '--data_mult', type=int, default=1)
        parser.add_argument("-dmft", '--data_mult_ft', type=int, default=10)
        parser.add_argument("-w", '--workers', type=int, default=2)
        parser.add_argument('--continue_path', default="")
        parser.add_argument('--output_root', default="E:/XVAP_OUTPUT/")
        parser.add_argument('--out', default="debug")
        parser.add_argument('--rank', default=0)
        parser.add_argument('--pitch', type=int, default=0) # TEMP - toggle the use of pitch conditioning
        parser.add_argument('--energy', type=int, default=0) # TEMP - toggle the use of energy conditioning
        parser.add_argument('--hifi_only', type=int, default=0) # Train only the hifi decoder and posterior encoder
        parser.add_argument('--pe_scaling', type=float, default=0.2)
        parser.add_argument('--target_bs', type=float, default=400)

        parser.add_argument('--analyze_loss', type=int, default=1)
        parser.add_argument('--lion', type=int, default=0)
        parser.add_argument('--bnb', type=int, default=0)

        parser.add_argument('--mltts_rc', type=int, default=0)
        parser.add_argument('--mltts_rc_rev', type=int, default=0)
        parser.add_argument('--lang_w', type=int, default=1)
        parser.add_argument('--big', type=int, default=1)
        parser.add_argument('--flc', type=int, default=0) # Full Lang conditioning (on all modules)
        parser.add_argument('--frozen_vocoder_langs', type=int, default=0)

        parser.add_argument('--single', type=int, default=0)
        parser.add_argument('--langs_config', type=int, default=0)

        parser.add_argument('--fp_emels', type=int, default=0)
        parser.add_argument('--ow_flow', type=int, default=0)
        parser.add_argument('--expanded_flow', type=int, default=0)
        parser.add_argument('--expanded_flow_dim', type=int, default=32)
        parser.add_argument('--debug', type=int, default=0)
        parser.add_argument('--extract', type=int, default=0)
        parser.add_argument('--group_id', default="")
        parser.add_argument('--use_ddp', type=bool, default=False)
        # parser.add_argument("-fh", '--freeze_hifi', type=int, default=0)

        parser.add_argument('--vocoder', type=int, default=0)
        parser.add_argument('--ft_weight', type=int, default=20)
        parser.add_argument('--do_loss_sorting', type=int, default=1)
        parser.add_argument('--data', default="")
        args = parser.parse_args()
        return args



def sort_xvap (x):
    return int(x.split("xVAPitch_")[-1].split(".")[0].split("_")[0])
def last_checkpoint(dataset_output):
    final_ckpt_fname = None
    # If the output directory already exists
    if os.path.exists(f'{dataset_output}'):
        ckpts = os.listdir(f'{dataset_output}')
        ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("xVAPitch_")]
        # Pick the latest checkpoint from the output directory
        if len(ckpts):
            ckpts = sorted(ckpts, key=sort_xvap)
            final_ckpt_fname = f'{dataset_output}/{ckpts[-1]}'
    # Return nothing, if one is not found
    return final_ckpt_fname



