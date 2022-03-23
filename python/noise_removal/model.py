import os
import json
import traceback

# import ffmpeg
import subprocess

# Not a model, but it was easier to just integrate the code this way


class NoiseRemoval(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(NoiseRemoval, self).__init__()

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


        inPath, inPath2, outputDirectory = data["inPath"], data["inPath2"], data["outputDirectory"]

        removeNoiseStrength = 0.25 # TODO, make configurable

        # Create noise profile
        noise_wav = sorted([fname for fname in os.listdir(inPath2) if fname.endswith(".wav")])[0]
        noise_profile_file = f'{noise_wav.replace(".wav", "")}.noise_profile_file'
        if not os.path.exists(noise_profile_file):
            command = f'sox {inPath2}/{noise_wav} -n noiseprof {inPath2}/{noise_profile_file}'
            sox = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = sox.communicate()
            stderr = stderr.decode("utf-8")
            if len(stderr):
                self.logger.info(f'SOX Command: {command}')
                self.logger.info(f'SOX ERROR: {stderr}')
                if websocket is not None:
                    await websocket.send(json.dumps({"key": "tasks_error", "data": stderr}))


        input_files = sorted(os.listdir(inPath))
        input_files = [fname for fname in input_files if fname.endswith(".wav")]

        for fni, fname in enumerate(input_files):

            if fni%3==0 and websocket is not None:
                await websocket.send(json.dumps({"key": "task_info", "data": f'Removing noise: {fni+1}/{len(input_files)}  ({(int(fni+1)/len(input_files)*100*100)/100}%)'}))

            command = f'sox {inPath}/{fname} {outputDirectory}/{fname} noisered {inPath2}/{noise_profile_file} {removeNoiseStrength}'
            sox = subprocess.Popen(command, startupinfo=startupinfo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = sox.communicate()
            stderr = stderr.decode("utf-8")
            if len(stderr):
                self.logger.info(f'SOX Command: {command}')
                self.logger.info(f'SOX ERROR: {stderr}')
                if websocket is not None:
                    await websocket.send(json.dumps({"key": "tasks_error", "data": stderr}))

        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))

