import os
import json
import traceback

import torch


class ModelsManager(object):

    def __init__(self, logger, PROD, device="cpu"):
        super(ModelsManager, self).__init__()

        self.models_bank = {}
        self.logger = logger
        self.PROD = PROD
        self.device_label = device
        self.device = torch.device(device)

    async def init_model (self, model_key, websocket=None):
        model_key = model_key.lower()
        try:
            if model_key in list(self.models_bank.keys()) and self.models_bank[model_key].isReady:
                return

            if websocket is not None:
                await websocket.send(json.dumps({"key": "ass_model_loading", "data": "loading"}))

            self.logger.info(f'ModelsManager: Initializing model: {model_key}')

            if model_key=="ass":
                from python.audio_source_separation.model import ASS
                self.models_bank[model_key] = ASS(self.logger, self.PROD, self.device, self)

            if model_key=="diarization":
                from python.speaker_diarization.model import Diarization
                self.models_bank[model_key] = Diarization(self.logger, self.PROD, self.device, self)

            if model_key=="formatting":
                from python.audio_format.model import AudioFormatter
                self.models_bank[model_key] = AudioFormatter(self.logger, self.PROD, self.device, self)

            if model_key=="silence_cut":
                from python.silence_cut.model import SilenceCutter
                self.models_bank[model_key] = SilenceCutter(self.logger, self.PROD, self.device, self)

            if model_key=="hifigan":
                from python.hifigan.model import HiFi_GAN
                self.models_bank[model_key] = HiFi_GAN(self.logger, self.PROD, self.device, self)

            try:
                self.models_bank[model_key].model = self.models_bank[model_key].model.to(self.device)
            except:
                pass
        except:
            self.logger.info(traceback.format_exc())

    def load_model (self, model_key, ckpt_path, **kwargs):

        if model_key not in self.models_bank.keys():
            self.init_model(model_key)

        if not os.path.exists(ckpt_path):
            return "ENOENT"

        if self.models_bank[model_key].ckpt_path != ckpt_path:
            self.logger.info(f'ModelsManager: Loading model checkpoint: {model_key}, {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.models_bank[model_key].load_state_dict(ckpt_path, ckpt, **kwargs)

    def set_device (self, device):
        if device=="gpu":
            device = "cuda"
        if self.device_label==device:
            return
        self.device_label = device
        self.device = torch.device(device)
        self.logger.info(f'ModelsManager: Changing device to: {device}')
        for model_key in list(self.models_bank.keys()):
            self.models_bank[model_key].set_device(self.device)

    def models (self, key):
        return self.models_bank[key.lower()]
