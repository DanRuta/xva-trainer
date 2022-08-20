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

            # if websocket is not None:
            #     await websocket.send(json.dumps({"key": "ass_model_loading", "data": "loading"}))

            self.logger.info(f'ModelsManager: Initializing model: {model_key}')

            # Tools
            if model_key=="ass":
                from python.audio_source_separation.model import ASS
                self.models_bank[model_key] = ASS(self.logger, self.PROD, self.device, self)

            if model_key=="diarization":
                from python.speaker_diarization.model import Diarization
                self.models_bank[model_key] = Diarization(self.logger, self.PROD, self.device, self)

            if model_key=="formatting":
                from python.audio_format.model import AudioFormatter
                self.models_bank[model_key] = AudioFormatter(self.logger, self.PROD, self.device, self)

            if model_key=="normalize":
                from python.audio_norm.model import AudioNormalizer
                self.models_bank[model_key] = AudioNormalizer(self.logger, self.PROD, self.device, self)

            if model_key=="wem2ogg":
                from python.wem2ogg.model import Wem2Ogg
                self.models_bank[model_key] = Wem2Ogg(self.logger, self.PROD, self.device, self)

            if model_key=="cluster_speakers":
                from python.cluster_speakers.model import ClusterSpeakers
                self.models_bank[model_key] = ClusterSpeakers(self.logger, self.PROD, self.device, self)

            if model_key=="speaker_search":
                from python.speaker_search.model import SpeakerSearch
                self.models_bank[model_key] = SpeakerSearch(self.logger, self.PROD, self.device, self)

            if model_key=="speaker_cluster_search":
                from python.speaker_cluster_search.model import SpeakerClusterSearch
                self.models_bank[model_key] = SpeakerClusterSearch(self.logger, self.PROD, self.device, self)

            if model_key=="transcribe":
                from python.transcribe.model import Wav2Vec2PlusPuncTranscribe
                self.models_bank[model_key] = Wav2Vec2PlusPuncTranscribe(self.logger, self.PROD, self.device, self)

            if model_key=="wer_evaluation":
                from python.wer_evaluation.model import WER_evaluation
                self.models_bank[model_key] = WER_evaluation(self.logger, self.PROD, self.device, self)

            if model_key=="silence_cut":
                from python.silence_cut.model import SilenceCutter
                self.models_bank[model_key] = SilenceCutter(self.logger, self.PROD, self.device, self)

            if model_key=="noise_removal":
                from python.noise_removal.model import NoiseRemoval
                self.models_bank[model_key] = NoiseRemoval(self.logger, self.PROD, self.device, self)

            if model_key=="silence_split":
                from python.silence_split.model import SilenceSplit
                self.models_bank[model_key] = SilenceSplit(self.logger, self.PROD, self.device, self)

            if model_key=="cut_padding":
                from python.cut_padding.model import CutPadding
                self.models_bank[model_key] = CutPadding(self.logger, self.PROD, self.device, self)

            if model_key=="srt_split":
                from python.srt_split.model import SRTSplit
                self.models_bank[model_key] = SRTSplit(self.logger, self.PROD, self.device, self)

            if model_key=="make_srt":
                from python.make_srt.model import MakeSRT
                await self.init_model("diarization")
                await self.init_model("transcribe")
                self.models_bank[model_key] = MakeSRT(self.logger, self.PROD, self.device, self)

            # Models
            if model_key=="fastpitch1_1":
                from python.fastpitch1_1.xva_train import FastPitchTrainer
                self.models_bank[model_key] = FastPitchTrainer(self.logger, self.PROD, self.device, self)
            if model_key=="hifigan":
                from python.hifigan.model import HiFi_GAN
                self.models_bank[model_key] = HiFi_GAN(self.logger, self.PROD, self.device, self)

            try:
                self.models_bank[model_key].model = self.models_bank[model_key].model.to(self.device)
            except:
                pass
        except:
            self.logger.info(traceback.format_exc())

    def sync_init_model (self, model_key, websocket=None, gpus=[0]):
        model_key = model_key.lower()
        try:
            if model_key=="fastpitch1_1" and "fastpitch1_1" not in list(self.models_bank.keys()):
                from python.fastpitch1_1.xva_train import FastPitchTrainer
                self.models_bank[model_key] = FastPitchTrainer(self.logger, self.PROD, gpus, self, websocket=websocket)
            if model_key=="hifigan" and "hifigan" not in list(self.models_bank.keys()):
                from python.hifigan.xva_train import HiFiTrainer
                self.models_bank[model_key] = HiFiTrainer(self.logger, self.PROD, gpus, self, websocket=websocket)
        except:
            self.logger.info(traceback.format_exc())

    def load_model (self, model_key, ckpt_path, **kwargs):

        if model_key not in self.models_bank.keys():
            # Do here, to avoid having to handle async init_model code in the HTTP server
            if model_key=="infer_fastpitch1_1":
                from python.fastpitch1_1.xva_train import FastPitch1_1
                self.models_bank[model_key] = FastPitch1_1(self.logger, self.PROD, self.device, self)
            if model_key=="infer_hifigan":
                from python.hifigan.models import HiFi_GAN
                self.models_bank[model_key] = HiFi_GAN(self.logger, self.PROD, self.device, self)

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
