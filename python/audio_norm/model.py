
import json
import traceback

# import ffmpeg
import subprocess

# Not a model, but it was easier to just integrate the code this way


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

        outputPath = f'{outputDirectory}/{inPath.split("/")[-1].split(".")[0]}.wav'

        try:
            sp = subprocess.Popen(f'ffmpeg-normalize -ar 22050 "{inPath}" -o "{outputPath}"', startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

