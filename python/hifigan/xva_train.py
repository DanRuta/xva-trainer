import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# os.environ['NCCL_P2P_DISABLE'] = '1'
import json
import argparse
import traceback

import gc
import itertools
import math
import time
import warnings
import asyncio
import glob
import re
import datetime

import torch
import numpy as np
import torch.nn.functional as F
import wave
import contextlib

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

# Still allow command-line use, from within this directory
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader





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

def sort_ckpt (ckpt):
    return int(ckpt.split("_")[1])

async def handleTrainer (models_manager, data, websocket, gpus, resume=False):

    gc.collect()
    torch.cuda.empty_cache()

    if not resume:
        models_manager.sync_init_model("hifigan", websocket=websocket, gpus=[0])
        trainer = models_manager.models_bank["hifigan"]

        dataset_id = data["dataset_path"].split("/")[-1]
        dataset_output = data["output_path"] + f'/{dataset_id}'

        trainer.logger.info(f'[debug HiFi] {str(data)}')
        trainer.logger.info(f'[debug] traceback: {traceback.format_exc()}')

        # trainer.init_logs(dataset_output=data["output_path"])
        trainer.init_logs(dataset_output=dataset_output)
    else:
        trainer = models_manager.models_bank["hifigan"]


    try:
        await trainer.start(data, gpus=gpus, resume=resume)
    except KeyboardInterrupt:
        trainer.running = False
        raise
    except RuntimeError as e:
        trainer.running = False
        try:
            del trainer.train_loader
            del trainer.dataloader_iterator
            del trainer.model
            del trainer.generator
            del trainer.mpd
            del trainer.msd
            del trainer.trainset
        except:
            pass

        gc.collect()
        torch.cuda.empty_cache()
        if "CUDA out of memory" in str(e) or "PYTORCH_CUDA_ALLOC_CONF" in str(e):
            trainer.logger.info("CUDA out of memory")
            trainer.print_and_log(f'============= Reducing batch size from {trainer.batch_size} to {trainer.batch_size-5}', save_to_file=trainer.dataset_output)
            data["batch_size"] = data["batch_size"] - 5
            return await handleTrainer(models_manager, data, websocket, gpus)
        elif trainer.END_OF_TRAINING:
            trainer.logger.info("Finished training HiFi-GAN")
            trainer.print_and_log(f'Finished training HiFi-GAN\n', save_to_file=trainer.dataset_output)
            await trainer.websocket.send(f'Finished training HiFi-GAN\n')

            del trainer
            try:
                del models_manager.models_bank["hifigan"]
            except:
                pass
            gc.collect()
            return "done"
        else:
            trainer.logger.info("HiFi-GAN error")
            trainer.logger.info(str(e))
            raise





class HiFiTrainer(object):
    def __init__(self, logger, PROD, gpus, models_manager, websocket=None):
        super(HiFiTrainer, self).__init__()

        self.logger = logger
        if self.logger is not None:
            self.logger.info("New HiFiTrainer")
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = torch.device(f'cuda:{gpus[0]}')
        self.ckpt_path = None
        self.websocket = websocket
        self.training_log = []
        self.training_log_live_line = ""

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
        # self.cmudict_path = f'{"./resources/app" if self.PROD else "."}/python/hifigan/cmudict/cmudict-0.7b'
        self.config_path = f'{"./resources/app" if self.PROD else "."}/python/hifigan/config_v1.json'
        self.gpus = gpus

        self.pretrained_ckpt_male = f'{"./resources/app" if self.PROD else "."}/python/hifigan/pretrained_models/male'
        self.pretrained_ckpt_female = f'{"./resources/app" if self.PROD else "."}/python/hifigan/pretrained_models/female'


        self.JUST_FINISHED_STAGE = False
        self.END_OF_TRAINING = False


    def print_and_log (self, line=None, end="\n", flush=False, save_to_file=False):
        if line is None:
            print(f'\r{self.training_log_live_line}', end="", flush=True)
        else:
            time_str = str(datetime.datetime.now().time())
            time_str = time_str.split(":")[0]+":"+time_str.split(":")[1]+":"+time_str.split(":")[2]
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
            from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss, feature_loss, generator_loss
            from env import AttrDict
            from utils import scan_checkpoint, load_checkpoint, save_checkpoint
            from meldataset import MelDataset, get_dataset_filelist, mel_spectrogram
            with open(f'./config_v1.json') as f:
                data = f.read()
        else:
            from python.hifigan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss, feature_loss, generator_loss
            from python.hifigan.env import AttrDict
            from python.hifigan.utils import scan_checkpoint, load_checkpoint, save_checkpoint
            from python.hifigan.meldataset import MelDataset, get_dataset_filelist, mel_spectrogram
            with open(self.config_path) as f:
                data = f.read()
        self.mel_spectrogram = mel_spectrogram
        self.discriminator_loss = discriminator_loss
        self.feature_loss = feature_loss
        self.generator_loss = generator_loss
        self.save_checkpoint = save_checkpoint
        self.load_checkpoint = load_checkpoint

        self.EPOCH_AVG_SPAN = 25
        self.avg_loss_per_epoch = []


        self.init_logs(dataset_output=self.dataset_output)

        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.h.batch_size = int(self.batch_size * 1.5)
        self.h.num_workers = self.workers
        self.h.USE_EMB_CONDITIONING = False

        torch.cuda.manual_seed(self.h.seed)

        self.generator = Generator(self.h).to(self.device)
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)

        checkpoint_path = self.dataset_output+"/hifi"
        self.print_and_log(f'Output checkpoints directory: {checkpoint_path}', save_to_file=self.dataset_output)
        self.print_and_log(f'Stage 5: HiFi-GAN fine-tuning', save_to_file=self.dataset_output)
        self.print_and_log(f'Batch size: {self.h.batch_size} (Base: {self.batch_size}, Stage mult: 1.5)', save_to_file=self.dataset_output)
        # self.print_and_log(f'Workers: {self.h.num_workers}', save_to_file=self.dataset_output)
        os.makedirs(checkpoint_path, exist_ok=True)
        if self.websocket:
            await self.websocket.send(f'Set stage to: 5 ')

            # # ===================== DEVELOPING START
            # self.END_OF_TRAINING = True
            # raise
            # # ===================== DEVELOPING END

        if os.path.isdir(checkpoint_path):
            cp_g = scan_checkpoint(checkpoint_path, 'g_')
            cp_do = scan_checkpoint(checkpoint_path, 'do_')

        if cp_g is None:
            self.print_and_log(f'No existing HiFi-GAN checkpoints for this voice.', save_to_file=self.dataset_output)

            if self.hifigan_checkpoint=="[male]":
                cp_g = scan_checkpoint(self.pretrained_ckpt_male, 'g_')
                cp_do = scan_checkpoint(self.pretrained_ckpt_male, 'do_')
            elif self.hifigan_checkpoint=="[female]":
                cp_g = scan_checkpoint(self.pretrained_ckpt_female, 'g_')
                cp_do = scan_checkpoint(self.pretrained_ckpt_female, 'do_')



        self.target_delta = 0.0002
        self.graphs_json["stages"]["5"]["target_delta"] = self.target_delta

        self.training_steps = 0
        self.ckpts_finetuned = 0
        self.training_epoch = -1
        if cp_g is None or cp_do is None:
            raise # Don't ever train from scratch
            state_dict_do = None
            self.training_epoch = -1
        else:
            cp_g = cp_g.replace("\\", "/")
            self.print_and_log(f'Loading checkpoint from: {cp_g}', save_to_file=self.dataset_output)

            try:
                state_dict_g = self.load_checkpoint(cp_g, self.device)
                state_dict_do = self.load_checkpoint(cp_do, self.device)
            except:
                self.print_and_log(f'Failed to load the checkpoint! Maybe try the second-last checkpoint (delete the last one). Full error message: {traceback.format_exc()}', save_to_file=self.dataset_output)
                raise
            self.generator.load_state_dict(state_dict_g['generator'])
            self.mpd.load_state_dict(state_dict_do['mpd'])
            self.msd.load_state_dict(state_dict_do['msd'])

            self.training_steps = state_dict_do['steps'] + 1
            self.training_epoch = state_dict_do['epoch']
            self.ckpts_finetuned = state_dict_do['ckpts_finetuned'] if 'ckpts_finetuned' in list(state_dict_do.keys()) else 0

        self.optim_g = torch.optim.AdamW(self.generator.parameters(), self.h.learning_rate, betas=[self.h.adam_b1, self.h.adam_b2])
        self.optim_d = torch.optim.AdamW(itertools.chain(self.msd.parameters(), self.mpd.parameters()),
                                    self.h.learning_rate, betas=[self.h.adam_b1, self.h.adam_b2])

        if state_dict_do is not None:
            self.optim_g.load_state_dict(state_dict_do['optim_g'])
            self.optim_d.load_state_dict(state_dict_do['optim_d'])

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=self.h.lr_decay, last_epoch=self.training_epoch)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=self.h.lr_decay, last_epoch=self.training_epoch)

        input_training_file = f'{self.dataset_input}/metadata.csv'
        input_wavs_dir = f'{self.dataset_input}/wavs'

        training_filelist, not_found, dm = get_dataset_filelist(input_training_file, input_wavs_dir, use_embs=self.h.USE_EMB_CONDITIONING)
        self.print_and_log(f'Training items: {int(len(training_filelist)/dm)} | Data multiplier: {dm} | Not found: {not_found} | Total: {len(training_filelist)}', save_to_file=self.dataset_output)

        trainset = MelDataset(training_filelist, self.h.segment_size, self.h.n_fft, self.h.num_mels,
                              self.h.hop_size, self.h.win_size, self.h.sampling_rate, self.h.fmin, self.h.fmax, n_cache_reuse=0,
                              shuffle=True, fmax_loss=self.h.fmax_for_loss, device=self.device, h=self.h)

        # TODO, fix the script randomly dying at set interval when num_workers>0
        # self.train_loader = DataLoader(trainset, num_workers=self.h.num_workers, shuffle=True,
        self.train_loader = DataLoader(trainset, num_workers=0, shuffle=True,
                              sampler=None,
                              batch_size=self.h.batch_size,
                              pin_memory=True,
                              # pin_memory=True,
                              persistent_workers=False,
                              drop_last=True)

        self.sw = SummaryWriter(os.path.join(checkpoint_path, 'logs'))
        self.generator.train()
        self.mpd.train()
        self.msd.train()

        self.saved_ckpts = 0

        # https://github.com/pytorch/pytorch/issues/1917#issuecomment-433698337
        self.dataloader_iterator = iter(self.train_loader)
        self.start_new_epoch()

        self.is_init = True



    def init_logs (self, dataset_output):
        if self.logs_are_init:
            return
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
            self.print_and_log("No graphs.json file found. Starting anew.")

        self.logs_are_init = True


    async def start (self, data, gpus=None, resume=None):
        if self.running:
            return
        self.running = True

        if not resume:
            if gpus is not None:
                self.gpus = gpus


            self.dataset_input = data["dataset_path"]
            self.dataset_id = self.dataset_input.split("/")[-1]
            self.dataset_output = data["output_path"] + f'/{self.dataset_id}'
            os.makedirs(self.dataset_output, exist_ok=True)
            self.hifigan_checkpoint = data["hifigan_checkpoint"]

            self.workers = data["num_workers"]
            self.batch_size = data["batch_size"]
            self.epochs_per_checkpoint = data["epochs_per_checkpoint"]



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
                    if self.logger is not None:
                        self.logger.info(err_msg)
                    raise


    def pause (self, websocket=None):
        self.logger.info("pause")
        self.running = False
        torch.cuda.empty_cache()

    def resume (self, websocket=None):
        self.logger.info("resume")
        torch.cuda.empty_cache()
        if self.running:
            return
        self.running = True
        self.iteration()



    def start_new_epoch (self):
        self.epoch_start_time = time.time()
        self.avg_loss_per_epoch += [0.0]
        self.epoch_iter = 0
        self.iter_losses = []


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


        self.generator.zero_grad(set_to_none=True)
        self.mpd.zero_grad(set_to_none=True)
        self.msd.zero_grad(set_to_none=True)

        start_b = time.time()
        x, y, _, y_mel, _ = batch
        del batch
        x = torch.autograd.Variable(x.to(self.device, non_blocking=True))
        y = torch.autograd.Variable(y.to(self.device, non_blocking=True))
        y_mel = torch.autograd.Variable(y_mel.to(self.device, non_blocking=True))
        y = y.unsqueeze(1)

        y_g_hat = self.generator(x)
        y_g_hat_mel = self.mel_spectrogram(y_g_hat.squeeze(1), self.h.n_fft, self.h.num_mels, self.h.sampling_rate, self.h.hop_size, self.h.win_size, self.h.fmin, self.h.fmax_for_loss)

        del x, _,
        self.optim_d.zero_grad()



        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        del _

        loss_disc_all = loss_disc_s + loss_disc_f
        loss_disc_all.backward()
        self.optim_d.step()

        # Generator
        self.optim_g.zero_grad()

        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = self.generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = self.generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        loss_gen_all.backward()
        self.optim_g.step()

        with torch.no_grad():
            mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

        gen_loss = int(loss_gen_all*1000)/1000
        mel_loss = int(mel_error*1000)/1000
        self.epoch_iter += 1
        self.avg_loss_per_epoch[-1] += mel_loss

        # if self.training_steps % 5 == 0:
        s_per_b =  int((time.time() - start_b)*1000)/1000
        its_p_s = 1/(s_per_b/self.h.batch_size)
        its_p_s = int(its_p_s*100)/100

        avg_losses_print = ""
        if len(self.avg_loss_per_epoch)>=2:
            acc_epoch_deltas = []
            avg_loss_per_epoch_for_printing = [val for val in self.avg_loss_per_epoch]
            avg_loss_per_epoch_for_printing[-1] /= self.epoch_iter
            for vi, val in enumerate(avg_loss_per_epoch_for_printing):
                if vi:
                    delta = (avg_loss_per_epoch_for_printing[vi-1]-avg_loss_per_epoch_for_printing[vi])/avg_loss_per_epoch_for_printing[vi-1]
                    acc_epoch_deltas.append(delta)
            acc_epoch_deltas_avg20 = np.mean(acc_epoch_deltas if len(acc_epoch_deltas)<self.EPOCH_AVG_SPAN else acc_epoch_deltas[-self.EPOCH_AVG_SPAN:])
            avg_losses_print = int(acc_epoch_deltas_avg20*100000)/100000
            avg_losses_print = f'| Avg loss % delta: {avg_losses_print} | Target: {self.target_delta}'

        print_line = f'Stage 5 | Epoch: {self.training_epoch+1} | It: {(self.training_steps+1)%len(self.train_loader)}/{len(self.train_loader)} ({self.training_steps+1}) | Mel loss: {mel_loss} | its/s: {its_p_s} {avg_losses_print}'
        self.training_log_live_line = print_line
        self.print_and_log(save_to_file=self.dataset_output)


        del y_mel, y_g_hat, y_g_hat_mel, y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g, y
        del loss_fm_f, loss_fm_s, loss_gen_f, losses_gen_f, loss_gen_s, losses_gen_s, loss_mel
        loss_gen_all = loss_gen_all.item()

        # Tensorboard summary logging
        if self.training_steps % 100 == 0:
            self.sw.add_scalar("training/gen_loss_total", loss_gen_all, self.training_steps)
            self.sw.add_scalar("training/mel_spec_error", mel_error, self.training_steps)
            self.sw.add_scalar("training/d_lr", self.optim_d.param_groups[0]['lr'], self.training_steps)
            self.sw.add_scalar("training/g_lr", self.optim_g.param_groups[0]['lr'], self.training_steps)
        self.training_steps += 1


        del loss_gen_all, mel_error

        if self.running:
            await self.iteration()
        else:
            self.logger.info(f'[DEBUG] returning, not self.running')
            return


    def output_checkpoint (self):
        if self.training_epoch % self.epochs_per_checkpoint == 0:
            try:
                output_path = "{}/hifi/g_{:08d}".format(self.dataset_output, self.training_steps)
                self.save_checkpoint(output_path, {'generator': (self.generator.module if len(self.gpus)>1 else self.generator).state_dict()})
                self.print_and_log(f'Stage 5 |Epoch: {self.training_epoch} | It: {self.training_steps} | {output_path.split("/")[-1]} | Mel loss: {self.avg_loss_per_epoch[-1] / self.epoch_iter}', save_to_file=self.dataset_output)

                output_path = "{}/hifi/do_{:08d}".format(self.dataset_output, self.training_steps)
                self.ckpts_finetuned += 1
                ckpt_data = {
                    'mpd': (self.mpd.module if len(self.gpus)>1 else self.mpd).state_dict(),
                    'msd': (self.msd.module if len(self.gpus)>1 else self.msd).state_dict(),
                    'optim_g': self.optim_g.state_dict(), 'optim_d': self.optim_d.state_dict(), 'steps': self.training_steps,
                    'epoch': self.training_epoch,
                    # "avg_loss_per_epoch": self.avg_loss_per_epoch,
                    "avg_loss_per_epoch": [],
                    "ckpts_finetuned": self.ckpts_finetuned
                }
                self.save_checkpoint(output_path, ckpt_data)
                del ckpt_data

                # Clear old checkpoints
                ckpts = sorted([fname for fname in os.listdir(self.dataset_output+"/hifi") if "do_" in fname], key=sort_ckpt)[:-2]
                for ckpt in ckpts:
                    os.remove(f'{self.dataset_output}/hifi/{ckpt}')
                ckpts = sorted([fname for fname in os.listdir(self.dataset_output+"/hifi") if "g_" in fname], key=sort_ckpt)[:-2]
                for ckpt in ckpts:
                    os.remove(f'{self.dataset_output}/hifi/{ckpt}')

                # Save the output file, for ease
                output_path = f'{self.dataset_output}/{self.dataset_output.split("/")[-1]}.hg.pt'
                self.save_checkpoint(output_path, {'generator': (self.generator.module if len(self.gpus)>1 else self.generator).state_dict()})
            except:
                print(traceback.format_exc())
                raise


    def finish_epoch (self):

        self.scheduler_g.step()
        self.scheduler_d.step()
        self.training_epoch += 1

        self.output_checkpoint()

        self.avg_loss_per_epoch[-1] /= self.epoch_iter
        self.iter_start_time = None

        acc_epoch_deltas = []
        for vi, val in enumerate(self.avg_loss_per_epoch):
            if vi:
                delta = (self.avg_loss_per_epoch[vi-1]-self.avg_loss_per_epoch[vi])/self.avg_loss_per_epoch[vi-1]
                # delta = max(0, delta)
                acc_epoch_deltas.append(delta)

        acc_epoch_deltas_avg20 = None
        if len(acc_epoch_deltas)>=2:
            acc_epoch_deltas_avg20 = np.mean(acc_epoch_deltas if len(acc_epoch_deltas)<self.EPOCH_AVG_SPAN else acc_epoch_deltas[-self.EPOCH_AVG_SPAN:])
            self.sw.add_scalar(f'meta/stage_5_acc_epoch_deltas_avg20', acc_epoch_deltas_avg20, self.training_steps)

        self.graphs_json["stages"]["5"]["loss"].append([self.training_steps, self.avg_loss_per_epoch[-1]])

        if len(acc_epoch_deltas)>=2:
            acc_epoch_deltas_avg20 = np.mean(acc_epoch_deltas if len(acc_epoch_deltas)<self.EPOCH_AVG_SPAN else acc_epoch_deltas[-self.EPOCH_AVG_SPAN:])

            self.graphs_json["stages"]["5"]["loss_delta"].append([self.training_steps, acc_epoch_deltas_avg20])
            with open(f'{self.dataset_output}/graphs.json', "w+") as f:
                f.write(json.dumps(self.graphs_json))

            if acc_epoch_deltas_avg20 <= self.target_delta and len(acc_epoch_deltas)>=25:
                self.training_log_live_line = ""
                if self.logger is not None:
                    self.logger.info("[HiFi Trainer] END_OF_TRAINING...")
                self.print_and_log(f'HiFi-GAN training finished', save_to_file=self.dataset_output)
                self.END_OF_TRAINING = True
                raise




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Training', allow_abbrev=False)
    parser.add_argument('-gpus', type=str, default=f'0', help='CUDA devices')
    args, _ = parser.parse_known_args()

    gpus = [int(val) for val in args.gpus.split(",")]
    print(f'gpus, {gpus}')


    async def do_next_dataset_or_stage ():

        dataset_pairs = []
        dataset_pairs.append(["D:/FP_INPUT/...", "[male]"])
        dataset_pairs.append(["D:/FP_INPUT/...", "[female]"])

        while len(dataset_pairs):
            torch.cuda.empty_cache()

            TRAINER = HiFiTrainer(None, PROD=False, gpus=gpus, models_manager=None)

            try:
                init_data = {}
                init_data["dataset_path"] = dataset_pairs[0][0]
                init_data["output_path"] = f'D:/FP_OUTPUT/{dataset_pairs[0][0].split("/")[-1]}'
                init_data["hifigan_checkpoint"] = dataset_pairs[0][1]
                init_data["num_workers"] = 4
                init_data["batch_size"] = 32
                init_data["epochs_per_checkpoint"] = 5

                print("start training")
                await TRAINER.start(init_data, gpus=gpus)
                print("end training")

            except KeyboardInterrupt:
                raise
            except RuntimeError as e:
                print(str(e))



    import asyncio
    try:
        asyncio.run(do_next_dataset_or_stage())
    except:
        import traceback
        print("last traceback " + traceback.format_exc())
        print("exit")
        quit()