import os
import shutil
import json
import traceback

# import ffmpeg
import subprocess

# Not a model, but it was easier to just integrate the code this way

import multiprocessing as mp

def splitTask (data):
    [inPath, min_dB, silence_duration, ffmpeg_path, outputDirectory] = data
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    command = ""

    try:
        command = f'{ffmpeg_path} -i "{inPath}" -af silencedetect=noise={min_dB}dB:d={silence_duration} -f null -'
        command_process = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = command_process.communicate()
        stderr = stderr.decode("utf-8")
        # if len(stderr):
        #     self.logger.info(f'ffmpeg Command: {command}')
        #     self.logger.info(f'ffmpeg ERROR: {stderr}')
        #     if websocket is not None:
        #         await websocket.send(json.dumps({"key": "tasks_error", "data": stderr}))
        lines = [line for line in stderr.split("\n") if "silence_end" in line]
        silences = []
        for line in lines:

            end = line.split("silence_end: ")[1].split(" ")[0]
            dur = line.split(": ")[-1]
            start = float(end)-float(dur)

            if float(dur)>2:
                silences.append([start, float(end), float(dur)])

        filename = inPath.split("/")[-1]

        if len(silences)<2:
            shutil.copyfile(inPath, f'{outputDirectory}/{filename}')
        else:
            for si, silence in enumerate(silences[:-1]):
                silence_end = silence[1]
                next_silence_start = silences[si+1][0]

                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW


                out_fname = f'{outputDirectory}/{filename.split(".wav")[0]}_{str(si).zfill(6)}.wav'

                command = f'{ffmpeg_path} -ss {silence_end} -t {next_silence_start - silence_end + 0.25} -i "{inPath}" "{out_fname}"'
                sp = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = sp.communicate()
    except:
        return f'Command: {command} | Error: {traceback.format_exc()}'


class SilenceSplit(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(SilenceSplit, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.model = None
        self.isReady = True


    def load_state_dict (self, ckpt_path, sd):
        pass

    def set_device (self, device):
        pass

    def runTask (self, data, websocket=None):
        return self.remove_noise(data, websocket)


    async def remove_noise(self, data, websocket):

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW


        inPath, outputDirectory = data["inPath"], data["outputDirectory"]
        min_dB = data["toolSettings"]["min_dB"] if "min_dB" in data["toolSettings"].keys() else "-15"
        silence_duration = data["toolSettings"]["silence_duration"] if "silence_duration" in data["toolSettings"].keys() else "0.5"
        useMP = data["toolSettings"]["useMP"] if "useMP" in data["toolSettings"].keys() else False
        processes = max(1, int(mp.cpu_count()/2)-5) # TODO, figure out why more processes break the websocket


        ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'


        if useMP:

            input_paths = sorted(os.listdir(inPath))
            input_paths = [fpath for fpath in input_paths if not fpath.endswith(".ini")]

            workItems = []
            for ip, path in enumerate(input_paths):
                workItems.append([f'{inPath}/{path}', min_dB, silence_duration, ffmpeg_path, outputDirectory])

            workers = processes if processes>0 else max(1, mp.cpu_count()-1)
            workers = min(len(workItems), workers)
            self.logger.info("[mp ffmpeg] workers: "+str(workers))

            pool = mp.Pool(workers)
            results = pool.map(splitTask, workItems)
            pool.close()
            pool.join()

            errs = [items for items in results if items is not None]
            if len(errs):
                self.logger.info(errs)
                if websocket is not None:
                    await websocket.send(json.dumps({"key": "tasks_error", "data": f'Task done. {len(errs)} items failed (out of: {len(input_paths)})<br>First error (check the server.log for all):<br>{errs[0]}'}))

            if websocket is not None:
                await websocket.send(json.dumps({"key": "tasks_next"}))



        else:
            command = f'{ffmpeg_path} -i "{inPath}" -af silencedetect=noise={min_dB}dB:d={silence_duration} -f null -'
            command_process = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = command_process.communicate()
            stderr = stderr.decode("utf-8")
            if len(stderr) and "silence_end" not in stderr:
                self.logger.info(f'ffmpeg split ERROR: {stderr}')
            lines = [line for line in stderr.split("\n") if "silence_end" in line]
            silences = []
            for line in lines:

                end = line.split("silence_end: ")[1].split(" ")[0]
                dur = line.split(": ")[-1]
                start = float(end)-float(dur)

                if float(dur)>2:
                    silences.append([start, float(end), float(dur)])

            filename = inPath.split("/")[-1]

            if len(silences)<2:
                shutil.copyfile(inPath, f'{outputDirectory}/{filename}')
            else:
                for si, silence in enumerate(silences[:-1]):
                    silence_end = silence[1]
                    next_silence_start = silences[si+1][0]

                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW


                    out_fname = f'{outputDirectory}/{filename.split(".wav")[0]}_{str(si).zfill(6)}.wav'

                    command = f'{ffmpeg_path} -ss {silence_end} -t {next_silence_start - silence_end + 0.25} -i "{inPath}" "{out_fname}"'
                    sp = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = sp.communicate()
                    stderr = stderr.decode("utf-8")

            if websocket is not None:
                await websocket.send(json.dumps({"key": "tasks_next"}))

