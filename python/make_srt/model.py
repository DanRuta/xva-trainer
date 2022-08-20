import os
import json
import shutil
import traceback

import subprocess
import ffmpeg

# Not a model, but it was easier to just integrate the code this way

def format_time(time):
    hours = int(time/(60*60))
    time = time - hours*60*60
    minutes = int(time/60)
    time = time - minutes*60

    timestamp = f'{str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(int(time)).zfill(2)},{round(time-int(time), 3)}'
    return timestamp

def initializer(*arguments):
    global progress, total, lock
    progress, total, lock = arguments

class MakeSRT(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(MakeSRT, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.model = None
        self.isReady = True
        self.ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'


    def load_state_dict (self, ckpt_path, sd):
        pass

    def set_device (self, device):
        pass

    def runTask (self, data, websocket=None):
        return self.make_srt(data, websocket)


    async def make_srt(self, data, websocket):

        inPath, outputDirectory = data["inPath"], data["outputDirectory"]
        lang = data["toolSettings"]["language"] if "language" in data["toolSettings"].keys() else False

        files = [fname for fname in sorted(os.listdir(inPath)) if fname.endswith(".mp4")]

        try:
            os.remove(f'{"./resources/app" if self.PROD else "."}/python/make_srt/.progress.txt')
        except:
            pass

        ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'

        os.makedirs(f'{outputDirectory}/', exist_ok=True)

        for fi,mp4_file in enumerate(files):
            base_mp4_name = ".mp4".join(mp4_file.split(".mp4")[:-1])

            try:
                shutil.rmtree(f'{"./resources/app" if self.PROD else "."}/python/speaker_diarization/output')
            except:
                pass
            os.makedirs(f'{"./resources/app" if self.PROD else "."}/python/speaker_diarization/output', exist_ok=True)


            # Convert the file to 16khz .wav
            stream = ffmpeg.input(f'{inPath}/{mp4_file}')
            ffmpeg_options = {"ar": "16000", "ac": 1} # 16000Hz mono
            stream = ffmpeg.output(stream, f'{outputDirectory}/{base_mp4_name}.wav', **ffmpeg_options)
            out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))

            # Run speaker diarization on it, to get the speaker activity detection time intervals
            # Also splits the long .wav into small chunks
            with open(f'{"./resources/app" if self.PROD else "."}/python/make_srt/.progress.txt', "w+") as f:
                f.write(f'[File {fi+1}/{len(files)}] Running speaker diarization for: {mp4_file} ')
            data = {}
            data["inPath"] = f'{"./resources/app" if self.PROD else "."}/python/make_srt/output/{base_mp4_name}.wav'
            data["toolSettings"] = {}
            data["toolSettings"]["mergeSingleOutputFolder"] = True
            data["toolSettings"]["outputAudacityLabels"] = True

            await self.models_manager.models_bank["diarization"].runTask(data)

            with open(f'{"./resources/app" if self.PROD else "."}/python/speaker_diarization/output/audacity.txt') as f:
                timings = f.read().split("\n")
                timings = [[line.split("\t")[0],line.split("\t")[1]] for line in timings if len(line.strip())]

            # Run ASR on all chunks, to get transcript for every line
            with open(f'{"./resources/app" if self.PROD else "."}/python/make_srt/.progress.txt', "w+") as f:
                f.write(f'[File {fi+1}/{len(files)}] Running auto-transcribe for: {mp4_file} ')
            data = {}
            data["inPath"] = f'{"./resources/app" if self.PROD else "."}/python/speaker_diarization/output/'
            data["outputDirectory"] = f'{"./resources/app" if self.PROD else "."}/python/make_srt/output/metadata'
            data["toolSettings"] = {}
            data["toolSettings"]["language"] = lang
            data["toolSettings"]["useMP"] = False
            data["toolSettings"]["ignore_existing_transcript"] = True
            await self.models_manager.models_bank["transcribe"].runTask(data)

            with open(data["outputDirectory"]+"/metadata.csv", encoding="utf8") as f:
                transcript = [line for line in f.read().split("\n") if len(line)]


            # Format an .srt file using the time stamps and the transcript
            srt_file = []

            for ti,transcript_line in enumerate(transcript):
                srt_file.append(f'{ti}')
                srt_file.append(f'{format_time(float(timings[ti][0]))} --> {format_time(float(timings[ti][1]))}')
                srt_file.append(f'{transcript_line.split("|")[1]}')
                srt_file.append(f'')


            shutil.rmtree(data["outputDirectory"])
            os.remove(f'{"./resources/app" if self.PROD else "."}/python/make_srt/output/{base_mp4_name}.wav')

            with open(f'{"./resources/app" if self.PROD else "."}/python/make_srt/output/{base_mp4_name}.srt', "w+", encoding="utf8") as f:
                f.write("\n".join(srt_file))

        try:
            os.remove(f'{"./resources/app" if self.PROD else "."}/python/make_srt/.progress.txt')
        except:
            pass

        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))


