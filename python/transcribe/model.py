
import os
import json
import traceback


# Not a model, but it was easier to just integrate the code this way


import whisper

lang_names_to_codes = {
    "amharic": "am",
    "arabic": "ar",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "finnish": "fi",
    "french": "fr",
    "hausa": "ha",
    "hindi": "hi",
    "hungarian": "hu",
    "italian": "it",
    "japanese": "jp",
    "korean": "ko",
    "latin": "la",
    "mongolian": "mn",
    "dutch": "nl",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "kiswahili": "sw",
    "swedish": "sv",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "vietnamese": "vi",
    "wolof": "wo",
    "yoruba": "yo",
    "chinese Mandarin": "zh",
}


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

        self.ffmpeg_path = f'{"./resources/app" if PROD else "."}/python/ffmpeg.exe'

        self.model = None



    def load_state_dict (self, ckpt_path, sd):
        pass

    def set_device (self, device):
        pass

    def runTask (self, data, websocket=None):
        return self.transcribe(data, websocket)


    def lazy_load_wav2vec2_model (self, language):
        if self.lazy_loaded and self.wav2vec.language==language:
            return

        if self.wav2vec:
            del self.wav2vec

        from python.transcribe.wav2vec2.model import Wav2Vec2
        self.wav2vec = Wav2Vec2(self.logger, self.PROD, self.device, None, language)
        self.lazy_loaded = True

    async def transcribe(self, data, websocket):

        ignore_existing_transcript = data["toolSettings"]["ignore_existing_transcript"] if "ignore_existing_transcript" in data["toolSettings"] else False
        transcription_model = data["toolSettings"]["transcription_model"] if "transcription_model" in data["toolSettings"] else "whisper_medium"
        whisper_lang = data["toolSettings"]["whisper_lang"] if "whisper_lang" in data["toolSettings"] else "en"
        # useMP = data["toolSettings"]["useMP"] if "useMP" in data["toolSettings"].keys() else False
        # useMP_num_workers = int(data["toolSettings"]["useMP_num_workers"]) if "useMP_num_workers" in data["toolSettings"].keys() else 2
        # processes = max(1, int(mp.cpu_count()/2)-5) # TODO, figure out why more processes break the websocket
        self.websocket = websocket

        try:
            os.remove(f'{"./resources/app" if self.PROD else "."}/python/transcribe/.progress.txt')
        except:
            pass

        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Transcribing (model: {transcription_model}{" "+whisper_lang if transcription_model.startswith("whisper_")  else ""})... (can be quite slow)'}))

        inPath, outputDirectory = data["inPath"], data["outputDirectory"]
        self.outputDirectory = outputDirectory
        # if outputDirectory:
        #     shutil.rmtree(outputDirectory, ignore_errors=True)


        finished_transcript = {}

        # Check any existing transcriptions, and use them instead of generating new ones
        inPathParent = None
        self.logger.info(f'ignore_existing_transcript: {ignore_existing_transcript}')
        if not ignore_existing_transcript:
            # inPathParent = "/".join(inPath.split("/")[:-2])
            inPathParent = "/".join(self.outputDirectory.split("/")[:-2])
            # potential_metadata_path = inPathParent+"/metadata.csv"
            potential_metadata_path = f'{self.outputDirectory or inPathParent}/metadata.csv'
            self.logger.info(f"[WHISPER] potential_metadata_path: {potential_metadata_path}")
            if os.path.exists(potential_metadata_path):
                with open(potential_metadata_path, encoding="utf8") as f:
                    existing_data = f.read().split("\n")
                    for line in existing_data:
                        if len(line.strip()):
                            fname = line.split("|")[0]
                            text = line.split("|")[1].strip() if len(line.split("|"))>1 else ""
                            if len(text):
                                finished_transcript[fname] = text
        self.inPathParent = inPathParent




        input_files = [f'{inPath}/{file}' for file in list(os.listdir(inPath)) if ".wav" in file and "_16khz.wav" not in file]



        # Use either whisper, or Wav2Vec2
        #
        if transcription_model.startswith("whisper_"):

            _, size = transcription_model.split("_")
            # lang = lang_names_to_codes[lang.lower()]

            if self.model is None:
                with open(f'{"./resources/app" if self.PROD else "."}/python/transcribe/.progress.txt', "w+") as f:
                    f.write(f'Initializing whisper model...')

                model_path = f'{"./resources/app" if self.PROD else "."}/python/transcribe/whisper/{size}.pt'

                self.model = whisper.load_model(model_path)
                self.model = self.model.to(self.device)

            with open(f'{"./resources/app" if self.PROD else "."}/python/transcribe/.progress.txt', "w+") as f:
                f.write(f'Setting up whisper model...')

            self.options = whisper.DecodingOptions(language=whisper_lang, fp16=False)


            try:
                finished_transcript = self.handle_whisper(finished_transcript, input_files)
            except KeyboardInterrupt:
                raise
            except:
                # self.logger.info(f'file: {file}')
                self.logger.info(traceback.format_exc())

        else:
            _, lang = transcription_model.split("_")

            self.lazy_load_wav2vec2_model(lang)

            try:
                finished_transcript = self.handle_wav2vec(finished_transcript, input_files)
            except KeyboardInterrupt:
                raise
            except:
                # self.logger.info(f'file: {file}')
                self.logger.info(traceback.format_exc())





        self.dump_to_file(finished_transcript)



        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Cleaning up...'}))
        to_delete = [f'{inPath}/{file}' for file in list(os.listdir(inPath)) if "_16khz.wav" in file]
        for fpath in to_delete:
            os.remove(fpath)
        try:
            os.remove(f'{"./resources/app" if self.PROD else "."}/python/transcribe/.progress.txt')
        except:
            pass

        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))



    def dump_to_file (self, transcript):
        metadata = []
        for fname in list(transcript.keys()):
            metadata.append(f'{fname if ".wav" in fname else fname+".wav"}|{transcript[fname]}')

        if self.outputDirectory and len(self.outputDirectory):
            os.makedirs(self.outputDirectory, exist_ok=True)
        with open(f'{self.outputDirectory if self.outputDirectory and len(self.outputDirectory) else self.inPathParent}/metadata.csv', "w+", encoding="utf8") as f:
            f.write("\n".join(metadata))


    def handle_whisper(self, finished_transcript, input_files):

        finished_transcript_keys = list(finished_transcript.keys())

        self.logger.info(f"[WHISPER] transcript finished_transcript_keys {finished_transcript_keys}")


        for fi, file in enumerate(input_files):

            if fi%3==0 and self.websocket is not None:
                with open(f'{"./resources/app" if self.PROD else "."}/python/transcribe/.progress.txt', "w+") as f:
                    f.write(f'{fi+1}/{len(input_files)} | {round(  int(fi+1) / len(input_files)*100  , 2)}%')

            new_name = file.split("/")[-1]
            if new_name in finished_transcript_keys:
                continue

            # load audio and pad/trim it to fit 30 seconds

            audio = whisper.load_audio(file, ffmpeg_path=self.ffmpeg_path)
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            result = whisper.decode(self.model, mel, self.options)
            transcript = result.text


            transcript = transcript if transcript.endswith("?") or transcript.endswith("!") or transcript.endswith(".") or transcript.endswith(",") else f'{transcript}.'
            transcript_punct = transcript if transcript.endswith("?") or transcript.endswith("!") or transcript.endswith(".")  or transcript.endswith(",") else f'{transcript}.'
            finished_transcript[new_name] = transcript_punct



            if fi==0 or (fi+1)%10==0:
                self.dump_to_file(finished_transcript)

        return finished_transcript

    def handle_wav2vec(self, finished_transcript, input_files):

        for fi, file in enumerate(input_files):

            if fi%3==0 and self.websocket is not None:
                with open(f'{"./resources/app" if self.PROD else "."}/python/transcribe/.progress.txt', "w+") as f:
                    f.write(f'{fi+1}/{len(input_files)} | {round(  int(fi+1) / len(input_files)*100  , 2)}%')

            new_name = file.split("/")[-1].replace(".wem", "") # Helps avoid some issues, later down the line
            if new_name in list(finished_transcript.keys()):
                continue
            new_name = new_name.replace("_16khz", "")

            transcript = self.wav2vec.infer(file)

            if self.wav2vec.language=="en":
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
            else:
                transcript = transcript.lower().strip()

            transcript = transcript if transcript.endswith("?") or transcript.endswith("!") or transcript.endswith(".") or transcript.endswith(",") else f'{transcript}.'
            transcript_punct = transcript if transcript.endswith("?") or transcript.endswith("!") or transcript.endswith(".")  or transcript.endswith(",") else f'{transcript}.'
            finished_transcript[new_name] = transcript_punct

            if fi==0 or (fi+1)%10==0:
                self.dump_to_file(finished_transcript)

        return finished_transcript


