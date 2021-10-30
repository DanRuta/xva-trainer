
import json
import traceback

import subprocess

# Not a model, but it was easier to just integrate the code this way


class Wem2Ogg(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(Wem2Ogg, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.tool_path = f'{"./resources/app" if self.PROD else "."}/python/wem2ogg'

        self.model = None
        self.isReady = True


    def load_state_dict (self, ckpt_path, sd):
        pass

    def set_device (self, device):
        pass

    def runTask (self, data, websocket=None):
        return self.convert(data, websocket)


    async def convert(self, data, websocket):

        inPath, outputDirectory = data["inPath"], data["outputDirectory"]
        outputPath = f'{outputDirectory}/{inPath.split("/")[-1].split(".")[0]}.ogg'

        # Part 1
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        try:
            sp = subprocess.Popen(f'{self.tool_path}/ww2ogg/ww2ogg.exe "{inPath}" -o "{outputPath}" --pcb {self.tool_path}/ww2ogg/packed_codebooks_aoTuV_603.bin', startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = sp.communicate()
            stderr = stderr.decode("utf-8")

            self.logger.info("stdout: "+ stdout.decode("utf-8"))

            if len(stderr):
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
