
import os
import shutil
import json
import traceback

# import faiss
# from pathlib import Path
# import numpy as np
# import sklearn

# Not a model, but it was easier to just integrate the code this way

import ffmpeg

# Temporary
# import torch
# def returnFalse():
#     return False
# torch.cuda.is_available = returnFalse

# Transcription

# Punctuation restoration
# from python.transcribe.NLP_Toolkit.nlptoolkit.utils.config import Config
# from python.transcribe.NLP_Toolkit.nlptoolkit.punctuation_restoration.infer import infer_from_trained


import multiprocessing as mp

def transcribeTask (data):
    [language, PROD, worker_i, files] = data
    transcripts = []

    from python.transcribe.wav2vec2.model import Wav2Vec2
    wav2vec = Wav2Vec2(None, PROD, "cpu", None, language)

    for fdi, file_data in enumerate(files):
        [new_name, fpath] = file_data
        transcript = wav2vec.infer(fpath)

        if language=="en":
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
        transcripts.append([new_name, transcript_punct])

        if worker_i==0 and fdi%3==0:
            with open(f'{"./resources/app" if PROD else "."}/python/transcribe/.progress.txt', "w+") as f:
                f.write(f'{(int(fdi+1)/len(files)*100*100)/100}')

    return transcripts



def formatTask (data):
    [inPath, outPath, formatting_hz, ffmpeg_path] = data

    try:
        stream = ffmpeg.input(inPath)
        ffmpeg_options = {"ar": formatting_hz, "ac": 1} # 22050Hz mono
        stream = ffmpeg.output(stream, outPath, **ffmpeg_options)
        out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))
    except ffmpeg.Error as e:
        return "ffmpeg err: "+ e.stderr.decode('utf8')
    except:
        return traceback.format_exc()


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

        from python.transcribe.wav2vec2.model import Wav2Vec2
        self.wav2vec = Wav2Vec2(self.logger, self.PROD, self.device, None, language)
        self.lazy_loaded = True

    async def transcribe(self, data, websocket):

        ignore_existing_transcript = data["toolSettings"]["ignore_existing_transcript"] if "ignore_existing_transcript" in data["toolSettings"] else False
        language = data["toolSettings"]["language"] if "language" in data["toolSettings"] else "en"
        useMP = data["toolSettings"]["useMP"] if "useMP" in data["toolSettings"].keys() else False
        useMP_num_workers = int(data["toolSettings"]["useMP_num_workers"]) if "useMP_num_workers" in data["toolSettings"].keys() else 2
        processes = max(1, int(mp.cpu_count()/2)-5) # TODO, figure out why more processes break the websocket

        try:
            os.remove(f'{"./resources/app" if self.PROD else "."}/python/transcribe/.progress.txt')
        except:
            pass

        try:
            if websocket is not None:
                await websocket.send(json.dumps({"key": "task_info", "data": f'Transcribing... (can be quite slow)'}))

            self.lazy_load_models(language)

            inPath, outputDirectory = data["inPath"], data["outputDirectory"]
            if outputDirectory:
                shutil.rmtree(outputDirectory, ignore_errors=True)
        except:
            self.logger.info(traceback.format_exc())
            return


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

        if useMP:
            input_paths = input_files
            ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'

            workItems = []
            for ip, fpath in enumerate(input_paths):
                workItems.append([fpath, fpath.replace(".wav", "_16khz.wav"), 16000, ffmpeg_path])

            workers = processes if processes>0 else max(1, mp.cpu_count()-1)
            workers = min(len(workItems), workers)

            self.logger.info("[mp ffmpeg] workers: "+str(workers))

            pool = mp.Pool(workers)
            results = pool.map(formatTask, workItems)
            pool.close()
            pool.join()

            errs = [items for items in results if items is not None]

            errs = []
            input_files_new = []
            for ii, items in enumerate(results):
                if items is None:
                    input_files_new.append(input_paths[ii].replace(".wav", "_16khz.wav"))

            input_files = input_files_new
            if len(errs):
                self.logger.info(errs)

        try:

            self.logger.info(f'Transcribing {len(input_files)} files...')

            if useMP:
                num_workers = useMP_num_workers
                workItems = [[language, self.PROD, _, []] for _ in range(num_workers)]
                for fi, file in enumerate(input_files):
                    new_name = file.split("/")[-1].replace(".wem", "") # Helps avoid some issues, later down the line
                    if new_name in list(finished_transcript.keys()):
                        continue
                    workItems[fi%num_workers][3].append([new_name, file])


                self.logger.info("[mp ffmpeg] transcribe workers: "+str(num_workers))

                self.logger.info(f'Loading Wav2Vec2 model for language: {language}')

                pool = mp.Pool(workers)
                results = pool.map(transcribeTask, workItems)
                pool.close()
                pool.join()
                try:
                    os.remove(f'{"./resources/app" if self.PROD else "."}/python/transcribe/.progress.txt')
                except:
                    pass


                for result in results:
                    for item in result:
                        [new_name, transcript] = item
                        new_name = new_name.replace("_16khz", "")
                        finished_transcript[new_name] = transcript

            else:

                try:
                    for fi, file in enumerate(input_files):

                        if fi%3==0 and websocket is not None:
                            with open(f'{"./resources/app" if self.PROD else "."}/python/transcribe/.progress.txt', "w+") as f:
                                f.write(f'{(int(fi+1)/len(input_files)*100*100)/100}%')

                        new_name = file.split("/")[-1].replace(".wem", "") # Helps avoid some issues, later down the line
                        if new_name in list(finished_transcript.keys()):
                            continue
                        new_name = new_name.replace("_16khz", "")

                        transcript = self.wav2vec.infer(file)

                        # Generate punctuation for the line, and apply some manual post-processing to fix common issues
                        # try:
                        #     # transcript_punct = self.inferer.infer_sentence(transcript)
                        #     # transcript = transcript_punct
                        #     pass
                        # except:
                        #     self.logger.info(traceback.format_exc())
                        #     # transcript_punct = transcript

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
                except KeyboardInterrupt:
                    raise
                except:
                    self.logger.info(f'file: {file}')
                    self.logger.info(traceback.format_exc())
                    pass

        except KeyboardInterrupt:
            raise
        except:
            self.logger.info(traceback.format_exc())
            pass


        metadata = []
        for fname in list(finished_transcript.keys()):
            metadata.append(f'{fname if ".wav" in fname else fname+".wav"}|{finished_transcript[fname]}')

        if outputDirectory:
            os.makedirs(outputDirectory, exist_ok=True)
        with open(f'{outputDirectory or inPathParent}/metadata.csv', "w+", encoding="utf8") as f:
            f.write("\n".join(metadata))


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