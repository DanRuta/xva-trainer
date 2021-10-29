
import json
import traceback

import ffmpeg

# Not a model, but it was easier to just integrate the code this way


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

        outputPath = f'{outputDirectory}/{inPath.split("/")[-1].split(".")[0]}.wav'

        try:
            stream = ffmpeg.input(inPath)

            ffmpeg_options = {"ar": 22050, "ac": 1} # 22050Hz mono

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

