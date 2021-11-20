import sys
import json
import traceback


class ASS(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(ASS, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        from asteroid.models import BaseModel
        # self.model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
        self.model = BaseModel.from_pretrained(f'{"./resources/app" if self.PROD else "."}/python/audio_source_separation/assModel.pt')


        self.isReady = True


    def load_state_dict (self, ckpt_path, sd):
        self.ckpt_path = ckpt_path
        self.model.load_state_dict(sd["generator"])

    def set_device (self, device):
        self.device = device
        self.model = self.model.to(device)
        self.model.device = device

    def runTask (self, data, websocket=None):
        return self.separate(data, websocket)


    async def separate (self, data, websocket):
        inPath, outputDirectory = data["inPath"], data["outputDirectory"]

        try:
            self.model.separate(inPath, output_dir=outputDirectory, resample=True)
        except:
            self.logger.info(traceback.format_exc())

        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))
