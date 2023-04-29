import os
import json
import traceback

import ffmpeg

# Not a model, but it was easier to just integrate the code this way

import multiprocessing as mp

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


class AudioFormatter(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(AudioFormatter, self).__init__()

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
        return self.format(data, websocket)


    async def format(self, data, websocket):

        inPath, outputDirectory = data["inPath"], data["outputDirectory"]

        outputPath = f'{outputDirectory}/{".".join(inPath.split("/")[-1].split(".")[:-1])}.wav'

        useMP = data["toolSettings"]["useMP"] if "useMP" in data["toolSettings"].keys() else False
        formatting_hz = data["toolSettings"]["formatting_hz"] if "formatting_hz" in data["toolSettings"].keys() else "22050"
        formatting_hz = int(formatting_hz)
        # processes = data["toolSettings"]["mpProcesses"]
        processes = max(1, int(mp.cpu_count()/2)-5) # TODO, figure out why more processes break the websocket

        # TODO, make this an checkbox toggle
        # also TODO, need to add checkbox toggle for not deleting the output director first, before kicking off the tool
        if os.path.exists(outputPath):
            if websocket is not None:
                await websocket.send(json.dumps({"key": "tasks_next"}))
            return

        if useMP:
            input_paths = sorted(os.listdir(inPath))
            input_paths = [fpath for fpath in input_paths if not fpath.endswith(".ini")]
            output_paths = [f'{outputDirectory}/{".".join(fpath.split(".")[:-1])+".wav"}' for fpath in input_paths]
            input_paths = [f'{inPath}/{fpath}' for fpath in input_paths]
            ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'

            workItems = []
            for ip, path in enumerate(input_paths):
                workItems.append([path, output_paths[ip], formatting_hz, ffmpeg_path])

            workers = processes if processes>0 else max(1, mp.cpu_count()-1)
            workers = min(len(workItems), workers)

            self.logger.info("[mp ffmpeg] workers: "+str(workers))

            pool = mp.Pool(workers)
            results = pool.map(formatTask, workItems)
            pool.close()
            pool.join()

            errs = [items for items in results if items is not None]
            if len(errs):
                self.logger.info(errs)
                if websocket is not None:
                    await websocket.send(json.dumps({"key": "tasks_error", "data": f'Task done. {len(errs)} items failed (out of: {len(input_paths)})<br>First error (check the server.log for all):<br>{errs[0]}'}))

        else:
            try:
                stream = ffmpeg.input(inPath)

                ffmpeg_options = {"ar": formatting_hz, "ac": 1} # 22050Hz mono

                stream = ffmpeg.output(stream, outputPath, **ffmpeg_options)
                ffmpeg_path = f'{"./resources/app" if self.PROD else "."}/python/ffmpeg.exe'

                out, err = (ffmpeg.run(stream, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True, overwrite_output=True))

            except ffmpeg.Error as e:
                self.logger.info("ffmpeg err: "+ e.stderr.decode('utf8'))

                if websocket is not None:
                    await websocket.send(json.dumps({"key": "tasks_error", "data": e.stderr.decode('utf8')}))

            except:
                self.logger.info(traceback.format_exc())
                if websocket is not None:
                    await websocket.send(json.dumps({"key": "tasks_error", "data": traceback.format_exc()}))

            if websocket is not None:
                await websocket.send(json.dumps({"key": "tasks_next"}))

