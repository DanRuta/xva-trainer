
import os
import shutil
import json
import traceback

import faiss
from pathlib import Path
import numpy as np
import sklearn

# Not a model, but it was easier to just integrate the code this way

# Temporary
import torch
# def returnFalse():
#     return False
# torch.cuda.is_available = returnFalse

# Transcription
from python.transcribe.wav2vec2.model import Wav2Vec2
# Punctuation restoration
# from python.transcribe.NLP_Toolkit.nlptoolkit.utils.config import Config
# from python.transcribe.NLP_Toolkit.nlptoolkit.punctuation_restoration.infer import infer_from_trained

class Wav2Vec2PlusPuncTranscribe(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(Wav2Vec2PlusPuncTranscribe, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.wav2vec = None
        self.model = None
        self.isReady = True
        self.lazy_loaded = False


    def load_state_dict (self, ckpt_path, sd):
        pass

    def set_device (self, device):
        pass

    def runTask (self, data, websocket=None):
        return self.transcribe(data, websocket)


    def lazy_load_models (self, language):
        if self.lazy_loaded and self.wav2vec.language==language:
            return

        if self.wav2vec:
            del self.wav2vec
        self.wav2vec = Wav2Vec2(self.logger, self.PROD, self.device, None, language)
        self.lazy_loaded = True

    async def transcribe(self, data, websocket):

        ignore_existing_transcript = data["toolSettings"]["ignore_existing_transcript"] if "ignore_existing_transcript" in data["toolSettings"] else False
        language = data["toolSettings"]["language"] if "language" in data["toolSettings"] else "en"

        try:
            if websocket is not None:
                await websocket.send(json.dumps({"key": "task_info", "data": f'Initializing...'}))

            self.lazy_load_models(language)

            inPath, outputDirectory = data["inPath"], data["outputDirectory"]
            if outputDirectory:
                shutil.rmtree(outputDirectory, ignore_errors=True)
        except:
            self.logger.info(traceback.format_exc())
            return


        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Preparing data...'}))
        else:
            self.logger.info("No websocket for: Preparing data...")


        finished_transcript = {}

        # Check any existing transcriptions, and use them instead of generating new ones
        self.logger.info(f'ignore_existing_transcript: {ignore_existing_transcript}')
        if not ignore_existing_transcript:
            inPathParent = "/".join(inPath.split("/")[:-2])
            potential_metadata_path = inPathParent+"/metadata.csv"
            if os.path.exists(potential_metadata_path):
                with open(potential_metadata_path) as f:
                    existing_data = f.read().split("\n")
                    for line in existing_data:
                        if len(line.strip()):
                            fname = line.split("|")[0]
                            text = line.split("|")[1].strip() if len(line.split("|"))>1 else ""
                            if len(text):
                                finished_transcript[fname] = text

        input_files = [f'{inPath}/{file}' for file in list(os.listdir(inPath)) if ".wav" in file and "_16khz.wav" not in file]

        try:
            for fi, file in enumerate(input_files):

                if websocket is not None:
                    await websocket.send(json.dumps({"key": "task_info", "data": f'Transcribing audio files: {fi+1}/{len(input_files)}  ({(int(fi+1)/len(input_files)*100*100)/100}%)  - {file.split("/")[-1]} '}))
                else:
                    self.logger.info("No websocket for: Transcribing audio files...")

                new_name = file.split("/")[-1].replace(".wem", "") # Helps avoid some issues, later down the line
                if new_name in list(finished_transcript.keys()):
                    continue

                transcript = self.wav2vec.infer(file)

                # Generate punctuation for the line, and apply some manual post-processing to fix common issues
                # try:
                #     # transcript_punct = self.inferer.infer_sentence(transcript)
                #     # transcript = transcript_punct
                #     pass
                # except:
                #     self.logger.info(traceback.format_exc())
                #     # transcript_punct = transcript

                transcript = (" "+transcript.lower()+" ")\
                    .replace(" on't ", " don't ")\
                    .replace(" on't ", " don't ")\
                    .replace(" do n't ", " don't ")\
                    .replace(" i 'm ", " i'm ")\
                    .replace('"', "")\
                    .replace("hasn '. T", "hansn't")\
                    .replace("hasn '. t", "hansn't")\
                    .replace("you 've", "you've")\
                    .replace("you 're", "you're")\
                    .replace("does n't", "doesn't")\
                    .replace(" will' ", " we'll ")\
                    .replace("i don '", "i don't")\
                    .replace("it ' ", "it's")\
                    .replace(" '", "'")\
                    .replace("i,'ve' ", "i've")\
                    .replace("would n't", "wouldn't")\
                    .replace("ca n't", "can't")\
                    .replace("that,'s", "that's")\
                    .replace("they ve", "they've")\
                    .replace("we,'re", "we're")\
                    .replace("did n't", "didn't")\
                    .replace(" wo n't ", " won't ")\
                    .replace(" is n't ", " isn't ")\
                    .replace(" should n't ", " shouldn't ")\
                    .replace("it s ", "it's ")\
                    .replace(" have n't ", " haven't ")\
                    .replace(" was n't ", " wasn't ")\
                    .replace(" there s ", " there's ")\
                    .replace(" are n't ", " aren't ")\
                    .replace(" ai n't ", " ain't ")\
                    .replace(" i ve ", " i've ")\
                    .replace(" was nt ", " wasn't ")\
                    .replace(" didn t ", " didn't ")\
                    .replace(" weren t ", " weren't ")\
                    .replace(" you re ", " you're ")\
                    .replace(" ddon't ", " don't ")\
                    .strip()

                transcript = transcript if transcript.endswith("?") or transcript.endswith("!") or transcript.endswith(".") or transcript.endswith(",") else f'{transcript}.'
                transcript_punct = transcript if transcript.endswith("?") or transcript.endswith("!") or transcript.endswith(".")  or transcript.endswith(",") else f'{transcript}.'
                finished_transcript[new_name] = transcript_punct
        except KeyboardInterrupt :
            raise
        except:
            self.logger.info(f'file: {file}')
            self.logger.info(traceback.format_exc())
            pass


        metadata = []
        for fname in list(finished_transcript.keys()):
            metadata.append(f'{fname if ".wav" in fname else fname+".wav"}|{finished_transcript[fname]}')

        if outputDirectory:
            os.makedirs(outputDirectory, exist_ok=True)
        with open(f'{outputDirectory or inPathParent}/metadata.csv', "w+") as f:
            f.write("\n".join(metadata))


        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Cleaning up...'}))
        to_delete = [f'{inPath}/{file}' for file in list(os.listdir(inPath)) if "_16khz.wav" in file]
        for fpath in to_delete:
            os.remove(fpath)



        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))