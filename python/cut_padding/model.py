import os
import shutil
import json
import traceback

import subprocess

# Not a model, but it was easier to just integrate the code this way

import multiprocessing as mp

def splitTask (data):
    [inPath, min_dB, ffmpeg_path, outputDirectory] = data
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    command = ""

    try:
        outputPath = f'{outputDirectory}/{".".join(inPath.split("/")[-1].split(".")[:-1])}.wav'

        command = f'{ffmpeg_path} -i "{inPath}" -af "silenceremove=start_periods=1:start_duration=1:start_threshold={min_dB}dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_duration=1:start_threshold={min_dB}dB:detection=peak,aformat=dblp,areverse" "{outputPath}"'
        command_process = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = command_process.communicate()
        stderr = stderr.decode("utf-8")

    except:
        return f'Command: {command} | Error: {traceback.format_exc()}'


class CutPadding(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(CutPadding, self).__init__()

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
        return self.cut_padding(data, websocket)


    async def cut_padding(self, data, websocket):

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        inPath, outputDirectory = data["inPath"], data["outputDirectory"]
        min_dB = data["toolSettings"]["min_dB"] if "min_dB" in data["toolSettings"].keys() else "-15"
        useMP = data["toolSettings"]["useMP"] if "useMP" in data["toolSettings"].keys() else False
        processes = max(1, int(mp.cpu_count()/2)-5) # TODO, figure out why more processes break the websocket

        ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'



        if useMP:

            input_paths = sorted(os.listdir(inPath))
            input_paths = [fpath for fpath in input_paths if not fpath.endswith(".ini")]

            workItems = []
            for ip, path in enumerate(input_paths):
                workItems.append([f'{inPath}/{path}', min_dB, ffmpeg_path, outputDirectory])

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
            outputPath = f'{outputDirectory}/{".".join(inPath.split("/")[-1].split(".")[:-1])}.wav'

            command = f'{ffmpeg_path} -i "{inPath}" -af "silenceremove=start_periods=1:start_duration=1:start_threshold={min_dB}dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_duration=1:start_threshold={min_dB}dB:detection=peak,aformat=dblp,areverse" {outputPath}'
            self.logger.info(command)
            command_process = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = command_process.communicate()
            stderr = stderr.decode("utf-8")

            if websocket is not None:
                await websocket.send(json.dumps({"key": "tasks_next"}))

