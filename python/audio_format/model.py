import os
import json
import traceback

import ffmpeg

# Not a model, but it was easier to just integrate the code this way

import multiprocessing as mp

def formatTask (data):
    [inPath, outPath] = data

    try:
        stream = ffmpeg.input(inPath)
        ffmpeg_options = {"ar": 22050, "ac": 1} # 22050Hz mono
        # ffmpeg_options = {"ar": 16000, "ac": 1} # 22050Hz mono
        stream = ffmpeg.output(stream, outPath, **ffmpeg_options)
        out, err = (ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True))
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

        self.logger.info(f'format task')

        useMP = data["toolSettings"]["useMP"] if "useMP" in data["toolSettings"].keys() else False
        # processes = data["toolSettings"]["mpProcesses"]
        processes = max(1, mp.cpu_count()-1) # TODO

        # TODO, make this an checkbox toggle
        # also TODO, need to add checkbox toggle for not deleting the output director first, before kicking off the tool
        if os.path.exists(outputPath):
            if websocket is not None:
                await websocket.send(json.dumps({"key": "tasks_next"}))
            return

        if useMP:
            input_paths = sorted(os.listdir(inPath))
            output_paths = [f'{outputDirectory}/{".".join(fpath.split(".")[:-1])+".wav"}' for fpath in input_paths]
            input_paths = [f'{inPath}/{fpath}' for fpath in input_paths]


            workItems = []
            for ip, path in enumerate(input_paths):
                workItems.append([path, output_paths[ip]])

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

                ffmpeg_options = {"ar": 22050, "ac": 1} # 22050Hz mono
                # ffmpeg_options = {"ar": 16000, "ac": 1} # 22050Hz mono

                stream = ffmpeg.output(stream, outputPath, **ffmpeg_options)

                out, err = (ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True))

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

