import os
import sys
import json
import torch
import traceback

# def returnFalse():
#     return False
# torch.cuda.is_available = returnFalse

from scipy.io import wavfile

# from python.speaker_diarization.pipeline.speaker_diarization import SpeakerDiarization

class Diarization(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(Diarization, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        # self.model = torch.hub.load('pyannote/pyannote-audio', 'dia')
        self.model = load_model(f'{"./resources/app" if self.PROD else "."}/python/speaker_diarization/hub/')

        # self.logger.info(str(self.model))
        # self.model = BaseModel.from_pretrained(f'{"./resources/app" if self.PROD else "."}/python/audio_source_separation/assModel.pt')


        self.isReady = True


    def set_device (self, device):
        self.device = device
        self.model = self.model.to(device)
        self.model.device = device

    def runTask (self, data, websocket=None):
        return self.diarize(data, websocket)


    async def diarize (self, data, websocket):

        inPath = data["inPath"]
        mergeSameOutput = data["toolSettings"]["mergeSingleOutputFolder"]
        outputAudacityLabels = data["toolSettings"]["outputAudacityLabels"]


        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": "Reading file"}))

        audacity_file = []

        try:
            rate, data = wavfile.read(inPath)

            if websocket is not None:
                await websocket.send(json.dumps({"key": "task_info", "data": "Splitting file"}))

            diarization = self.model({'audio': inPath})

            out_file_counter = 0
            total_tracks = len(diarization._tracks)

            for turn, _, speaker in diarization.itertracks(yield_label=True):

                if websocket is not None:
                    await websocket.send(json.dumps({"key": "task_info", "data": f'Outputting chunks: {out_file_counter+1}/{total_tracks}'}))

                start_s = turn.start
                end_s = turn.end

                # Skip audio chunks less than 1 second long
                if end_s-start_s < 1:
                    continue

                if outputAudacityLabels:
                    audacity_file.append('{:.6f}\t{:.6f}\tSpeaker_{}'.format(start_s, end_s, speaker))

                split_data = data[int(start_s*rate):int(end_s*rate)]

                folder_name = ".".join(inPath.split("/")[-1].split(".")[:-1]).replace(".", "_")
                if mergeSameOutput:
                    out_folder = f'{"./resources/app" if self.PROD else "."}/python/speaker_diarization/output/'
                else:
                    out_folder = f'{"./resources/app" if self.PROD else "."}/python/speaker_diarization/output/{folder_name}/speaker {speaker}'

                os.makedirs(out_folder, exist_ok=True)
                if mergeSameOutput:
                    wavfile.write(f'{out_folder}/{folder_name}_{str(out_file_counter).zfill(7)}.wav', rate, split_data)
                else:
                    wavfile.write(f'{out_folder}/{folder_name}_{speaker}_{str(out_file_counter).zfill(7)}.wav', rate, split_data)
                out_file_counter += 1
        except:
            self.logger.info(traceback.format_exc())

        if outputAudacityLabels:
            with open(f'{out_folder}/audacity.txt', "w+", encoding="utf8") as f:
                f.write("\n".join(audacity_file))

        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))




#
#
#
#  This is a huge mess, but pyannote very much wants models to be downloaded from the internet
#  For future-proofing reasons, I don't want that, so I had to change a lot of the library code,
#  and the way the models were loaded, such that torchhub isn't used, and instead the local model files are used.
#
#
#

def load_model (_HUB_DIR):
    import typing
    import shutil
    import functools
    import yaml
    import zipfile

    from pyannote.audio.features import Pretrained as _Pretrained
    from pyannote.pipeline import Pipeline as _Pipeline

    dependencies = ['pyannote.audio', 'torch']

    _HUB_REPO = 'https://github.com/pyannote/pyannote-audio-hub'
    _ZIP_URL = f'{_HUB_REPO}/raw/master/{{kind}}s/{{name}}.zip'
    _PRETRAINED_URL = f'{_HUB_REPO}/raw/master/pretrained.yml'

    # path where pre-trained models and pipelines are downloaded and cached
    # _HUB_DIR = f'{"./resources/app" if self.PROD else "."}/python/speaker_diarization/hub'
    # _HUB_DIR = pathlib.Path(os.environ.get("PYANNOTE_AUDIO_HUB",
    #                                        "~/.pyannote/hub")).expanduser().resolve()

    # download pretrained.yml if needed
    _PRETRAINED_YML = _HUB_DIR + 'pretrained.yml'
    print(f'_PRETRAINED_YML, {_PRETRAINED_YML}')

    # if not _PRETRAINED_YML.exists():
    #     msg = (
    #         f'Downloading list of pretrained models and pipelines '
    #         f'to "{_PRETRAINED_YML}".'
    #     )
    #     print(msg)
    #     from pyannote.audio.utils.path import mkdir_p
    #     mkdir_p(_PRETRAINED_YML.parent)
    #     torch.hub.download_url_to_file(_PRETRAINED_URL,
    #                                    _PRETRAINED_YML,
    #                                    progress=True)

    def _generic(name: str,
                 duration: float = None,
                 step: float = 0.25,
                 batch_size: int = 32,
                 device: typing.Optional[typing.Union[typing.Text, torch.device]] = None,
                 pipeline: typing.Optional[bool] = None,
                 force_reload: bool = False) -> typing.Union[_Pretrained, _Pipeline]:
        """Load pretrained model or pipeline

        Parameters
        ----------
        name : str
            Name of pretrained model or pipeline
        duration : float, optional
            Override audio chunks duration.
            Defaults to the one used during training.
        step : float, optional
            Ratio of audio chunk duration used for the internal sliding window.
            Defaults to 0.25 (i.e. 75% overlap between two consecutive windows).
            Reducing this value might lead to better results (at the expense of
            slower processing).
        batch_size : int, optional
            Batch size used for inference. Defaults to 32.
        device : torch.device, optional
            Device used for inference.
        pipeline : bool, optional
            Wrap pretrained model in a (not fully optimized) pipeline.
        force_reload : bool
            Whether to discard the existing cache and force a fresh download.
            Defaults to use existing cache.

        Returns
        -------
        pretrained: `Pretrained` or `Pipeline`

        Usage
        -----
        >>> sad_pipeline = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')
        >>> scores = model({'audio': '/path/to/audio.wav'})
        """
        # print("name", name)


        model_exists = name in _MODELS
        pipeline_exists = name in _PIPELINES

        # print(f'PRE model_exists, {model_exists}')
        # print(f'PRE pipeline_exists, {pipeline_exists}')

        if model_exists and pipeline_exists:
            # print(f'model_exists and pipeline_exists')
            # pass

            # if pipeline is None:
            #     msg = (
            #         f'Both a pretrained model and a pretrained pipeline called '
            #         f'"{name}" are available. Use option "pipeline=True" to '
            #         f'load the pipeline, and "pipeline=False" to load the model.')
            #     raise ValueError(msg)

            if pipeline:
                kind = 'pipeline'
                # zip_url = _ZIP_URL.format(kind=kind, name=name)
                # sha256 = _PIPELINES[name]
                return_pipeline = True

            else:
                kind = 'model'
                # zip_url = _ZIP_URL.format(kind=kind, name=name)
                # sha256 = _MODELS[name]
                return_pipeline = False

        elif pipeline_exists:
        # elif False:
            # print(f'pipeline_exists')
            # pass

            # pass
            if pipeline is None:
                pipeline = True

            if not pipeline:
                msg = (
                    f'Could not find any pretrained "{name}" model. '
                    f'A pretrained "{name}" pipeline does exist. '
                    f'Did you mean "pipeline=True"?'
                )
                raise ValueError(msg)

            kind = 'pipeline'
            # zip_url = _ZIP_URL.format(kind=kind, name=name)
            # sha256 = _PIPELINES[name]
            return_pipeline = True

        elif model_exists:

            # print(f'model_exists')
            # pass
            if pipeline is None:
                pipeline = False

            kind = 'model'
            # zip_url = _ZIP_URL.format(kind=kind, name=name)
            # sha256 = _MODELS[name]
            return_pipeline = pipeline

            if name.startswith('emb_') and return_pipeline:
                msg = (
                    f'Pretrained model "{name}" has no associated pipeline. Use '
                    f'"pipeline=False" or remove "pipeline" option altogether.'
                )
                raise ValueError(msg)

        else:
            # print("ERROR====================")
            pass
            # msg = (
            #     f'Could not find any pretrained model nor pipeline called "{name}".'
            # )
            # raise ValueError(msg)

        # if sha256 is None:
        #     msg = (
        #         f'Pretrained {kind} "{name}" is not available yet but will be '
        #         f'released shortly. Stay tuned...'
        #     )
        #     raise NotImplementedError(msg)

        pretrained_dir = _HUB_DIR + f'/{kind}s'
        pretrained_subdir = pretrained_dir + f'/{name}'
        pretrained_zip = pretrained_dir + f'/{name}.zip'

        # import pathlib
        # pretrained_subdir = pathlib.Path(pretrained_subdir)

        # if not pretrained_subdir.exists() or force_reload:

        #     if pretrained_subdir.exists():
        #         shutil.rmtree(pretrained_subdir)

        #     from pyannote.audio.utils.path import mkdir_p
        #     mkdir_p(pretrained_zip.parent)
        #     try:
        #         msg = (
        #             f'Downloading pretrained {kind} "{name}" to "{pretrained_zip}".'
        #         )
        #         print(msg)
        #         torch.hub.download_url_to_file(zip_url,
        #                                        pretrained_zip,
        #                                        hash_prefix=sha256,
        #                                        progress=True)
        #     except RuntimeError as e:
        #         shutil.rmtree(pretrained_subdir)
        #         msg = (
        #             f'Failed to download pretrained {kind} "{name}".'
        #             f'Please try again.')
        #         raise RuntimeError(msg)

        #     # unzip downloaded file
        #     with zipfile.ZipFile(pretrained_zip) as z:
        #         z.extractall(path=pretrained_dir)

        if kind == 'model':

            params_yml = None
            params_yml_parent = None
            params_yml_c1 = os.listdir(pretrained_subdir)
            for c1 in params_yml_c1:
                params_yml_c2 = [fold for fold in os.listdir(f'{pretrained_subdir}/{c1}') if os.path.isdir(f'{pretrained_subdir}/{c1}/{fold}')]
                for c2 in params_yml_c2:
                    params_yml_c3 = os.listdir(f'{pretrained_subdir}/{c1}/{c2}')
                    for c3 in params_yml_c3:
                        params_yml_c4 = os.listdir(f'{pretrained_subdir}/{c1}/{c2}/{c3}')
                        for c4 in params_yml_c4:
                            if c4=="params.yml":
                                params_yml_parent = f'{pretrained_subdir}/{c1}/{c2}/{c3}'
                                params_yml = f'{pretrained_subdir}/{c1}/{c2}/{c3}/params.yml'
                                break
            # print(f'----------params_yml, {params_yml}')
            # print(f'----------params_yml_parent, {params_yml_parent}')




            # params_yml, = pretrained_subdir.glob('*/*/*/*/params.yml')
            # pretrained =  _Pretrained(validate_dir=params_yml.parent,
            pretrained =  _Pretrained(validate_dir=params_yml_parent,
                                      duration=duration,
                                      step=step,
                                      batch_size=batch_size,
                                      device=device)

            # if return_pipeline:
            #     if name.startswith('sad_'):
            #         from pyannote.audio.pipeline.speech_activity_detection import SpeechActivityDetection
            #         print("HERE PRE SpeechActivityDetection")
            #         pipeline = SpeechActivityDetection(scores=pretrained)
            #         print("HERE POST")
            #     elif name.startswith('scd_'):
            #         from pyannote.audio.pipeline.speaker_change_detection import SpeakerChangeDetection
            #         print("HERE PRE SpeakerChangeDetection")
            #         pipeline = SpeakerChangeDetection(scores=pretrained)
            #         print("HERE POST")
            #     elif name.startswith('ovl_'):
            #         from pyannote.audio.pipeline.overlap_detection import OverlapDetection
            #         print("HERE PRE OverlapDetection")
            #         pipeline = OverlapDetection(scores=pretrained)
            #         print("HERE POST")
            #     else:
            #         # this should never happen
            #         msg = (
            #             f'Pretrained model "{name}" has no associated pipeline. Use '
            #             f'"pipeline=False" or remove "pipeline" option altogether.'
            #         )
            #         raise ValueError(msg)

            #     return pipeline.load_params(params_yml)

            return pretrained

        elif kind == 'pipeline':

            from pyannote.audio.pipeline.utils import load_pretrained_pipeline
            params_yml = None
            params_yml_parent = None
            # print(f'START     pretrained_subdir, {pretrained_subdir}')
            # params_yml_c1 = os.listdir(pretrained_subdir)
            params_yml_c1 = [fold for fold in os.listdir(f'{pretrained_subdir}') if os.path.isdir(f'{pretrained_subdir}/{fold}')]
            for c1 in params_yml_c1:
                # params_yml_c2 = os.listdir(f'{pretrained_subdir}/{c1}'.replace("//","/"))
                params_yml_c2 = [fold for fold in os.listdir(f'{pretrained_subdir}/{c1}') if os.path.isdir(f'{pretrained_subdir}/{c1}/{fold}')]
                for c2 in params_yml_c2:
                    params_yml_c3 = os.listdir(f'{pretrained_subdir}/{c1}/{c2}')
                    for c3 in params_yml_c3:
                        if c3=="params.yml":
                            params_yml_parent = f'{pretrained_subdir}/{c1}/{c2}'
                            params_yml = f'{pretrained_subdir}/{c1}/{c2}/params.yml'
                            break

            # params_yml, *_ = pretrained_subdir.glob('*/*/params.yml')
            # return load_pretrained_pipeline(params_yml.parent)
            # print("=== ptp PRE")
            ptp = load_pretrained_pipeline(params_yml_parent)
            # print("=== ptp POST")
            return ptp

    with open(_PRETRAINED_YML, 'r') as fp:
        _pretrained = yaml.load(fp, Loader=yaml.SafeLoader)
    # print(f'_pretrained, {_pretrained}')

    ___stuff = {}



    _MODELS = _pretrained['models']
    # print(f'_MODELS, {_MODELS}')
    for name in _MODELS:
        # print(f'_MODELS name, {name}')
        # locals()[name] = functools.partial(_generic, name)
        ___stuff[name] = functools.partial(_generic, name)

    _PIPELINES = _pretrained['pipelines']
    # print(f'_PIPELINES, {_PIPELINES}')
    for name in _PIPELINES:
        # print(f'_PIPELINES name, {name}')
        # locals()[name] = functools.partial(_generic, name)
        ___stuff[name] = functools.partial(_generic, name)

    _SHORTCUTS = _pretrained['shortcuts']
    # print(f'_SHORTCUTS, {_SHORTCUTS}')
    for shortcut, name in _SHORTCUTS.items():
        # print(f'_SHORTCUTS name, {name}')
        # locals()[shortcut] = locals()[name]
        ___stuff[shortcut] = ___stuff[name]

    return ___stuff["dia"]()
