import os
import json
import traceback

# import ffmpeg
import subprocess

# Not a model, but it was easier to just integrate the code this way

import multiprocessing as mp


def normalizeTask (data):
    [inPath, outPath] = data
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    # sp = subprocess.Popen(f'ffmpeg-normalize -ar 22050 "{inPath}" -o "{outPath}"', startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sp = subprocess.Popen(f'ffmpeg-normalize -ar 16000 "{inPath}" -o "{outPath}"', startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = sp.communicate()
    stderr = stderr.decode("utf-8")

    if len(stderr) and "duration of less than 3 seconds" not in stderr:
        print("stderr", stderr)
        return "stderr: "+ stderr

    return None

class AudioNormalizer(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(AudioNormalizer, self).__init__()

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
        return self.normalize(data, websocket)


    async def normalize(self, data, websocket):

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        inPath, outputDirectory = data["inPath"], data["outputDirectory"]
        useMP = data["toolSettings"]["useMP"]
        # processes = data["toolSettings"]["mpProcesses"]
        processes = max(1, mp.cpu_count()-1) # TODO

        if useMP:

            input_paths = sorted(os.listdir(inPath))
            output_paths = [f'{outputDirectory}/{fpath}' for fpath in input_paths]
            input_paths = [f'{inPath}/{fpath}' for fpath in input_paths]


            workItems = []
            for ip, path in enumerate(input_paths):
                workItems.append([path, output_paths[ip]])

            workers = processes if processes>0 else max(1, mp.cpu_count()-1)
            workers = min(len(workItems), workers)

            self.logger.info("[mp ffmpeg] workers: "+str(workers))

            pool = mp.Pool(workers)
            results = pool.map(normalizeTask, workItems)
            pool.close()
            pool.join()
            self.logger.info("Finished mp job")

            errs = [items for items in results if items is not None]
            if len(errs):
                self.logger.info(errs)
                if websocket is not None:
                    await websocket.send(json.dumps({"key": "tasks_error", "data": f'Task done. {len(errs)} items failed (out of: {len(input_paths)})<br>First error (check the server.log for all):<br>{errs[0]}'}))

        else:

            outputPath = f'{outputDirectory}/{inPath.split("/")[-1].split(".")[0]}.wav'

            try:
                sp = subprocess.Popen(f'ffmpeg-normalize -ar 22050 "{inPath}" -o "{outputPath}"', startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # sp = subprocess.Popen(f'ffmpeg-normalize -ar 16000 "{inPath}" -o "{outputPath}"', startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = sp.communicate()
                stderr = stderr.decode("utf-8")

                if len(stderr) and "duration of less than 3 seconds" not in stderr:
                    print("stderr", stderr)
                    self.logger.info("stderr: "+ stderr)

                    if websocket is not None:
                        await websocket.send(json.dumps({"key": "tasks_error", "data": stderr}))

            except:
                self.logger.info(traceback.format_exc())
                if websocket is not None:
                    await websocket.send(json.dumps({"key": "tasks_error", "data": traceback.format_exc()}))


            if websocket is not None:
                await websocket.send(json.dumps({"key": "tasks_next"}))

