
import os
import shutil
import json
import traceback

from jiwer import wer

# Not a model, but it was easier to just integrate the code this way


class WER_evaluation(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(WER_evaluation, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.model = None
        self.isReady = True
        self.lazy_loaded = False


    def load_state_dict (self, ckpt_path, sd):
        pass

    def set_device (self, device):
        pass

    def runTask (self, data, websocket=None):
        return self.evaluate_wer(data, websocket)

    async def evaluate_wer(self, data, websocket):

        inPath, inPath2, outputDirectory = data["inPath"], data["inPath2"], data["outputDirectory"]

        data = {}

        inPath = f'{inPath}/metadata.csv' if ".csv" not in inPath else inPath

        with open(inPath.replace("//", "/")) as f:
            lines = f.read().split("\n")

            for line in lines:
                if len(line.strip()):
                    file = line.split("|")[0]
                    text = line.split("|")[1]
                    data[file] = {
                        "orig": text
                    }

        with open(f'{inPath2}/metadata.csv' if ".csv" not in inPath2 else inPath2) as f:
            lines = f.read().split("\n")

            for line in lines:
                if len(line.strip()):
                    print(f'line, {line}')
                    file = line.split("|")[0]
                    text = line.split("|")[1]
                    if file in data.keys():
                        data[file]["asr"] = text


        out = []
        for fname in list(data.keys()):
            if "asr" in data[fname].keys():
                err = wer(data[fname]["asr"], data[fname]["orig"])
            else:
                err = "0"
                data[fname]["asr"] = ""
            out.append("|".join([fname, str(err), data[fname]["orig"], data[fname]["asr"]]))

        out = sorted(out, key=sort_err, reverse=True)

        os.makedirs(outputDirectory, exist_ok=True)

        with open(f'{outputDirectory}/wer_eval_results.txt', "w+", encoding="utf8") as f:
            f.write("\n".join(out))


        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))

def sort_err(x):
    return float(x.split("|")[1])