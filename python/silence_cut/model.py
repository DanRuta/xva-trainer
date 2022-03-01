import os
import json
import traceback

from pydub import AudioSegment

# Not a model, but it was easier to just integrate the code this way


import multiprocessing as mp


def processingTask(workItem):
    [inputPath, outputPath] = workItem

    threshold = -40 # tweak based on signal-to-noise ratio
    interval = 1 # ms, increase to speed up
    max_silence = 300 / interval

    audio = AudioSegment.from_wav(inputPath)

    # break into chunks
    chunks = [audio[i:i+interval] for i in range(0, len(audio), interval)]

    # find number of chunks with dBFS below threshold
    final_starting_silent_count = 0
    silent_blocks = 0

    index_cuts = []
    for ci, c in enumerate(chunks):
        if (c.dBFS == float('-inf') or c.dBFS < threshold):
            silent_blocks += 1
        else:
            if final_starting_silent_count>0 and silent_blocks > max_silence:
                index_cuts.append(ci-silent_blocks+int(max_silence/2))
                index_cuts.append(ci-int(max_silence/2))

            if final_starting_silent_count==0:
                final_starting_silent_count = silent_blocks
                index_cuts.append(ci)
            silent_blocks = 0

    index_cuts.append(len(chunks)-silent_blocks) # Add the index of the last bit of audio
    spliced_chunks = []
    for i in range(int(len(index_cuts)/2)):
        spliced_chunks.append(chunks[index_cuts[i*2]:index_cuts[i*2+1]])

    spliced_chunks = [item for sublist in spliced_chunks for item in sublist]
    combined_sound = sum(spliced_chunks, AudioSegment.empty())
    combined_sound = combined_sound.set_frame_rate(22050)

    combined_sound.export(f'{outputPath}', format="wav", bitrate=22050, parameters=["-ac", "1"])



class SilenceCutter(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(SilenceCutter, self).__init__()
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

        inputDirectory, outputDirectory = data["inputDirectory"], data["outputDirectory"]

        workItems = [[f'{inputDirectory}/{file_name}', f'{outputDirectory}/{file_name}'] for file_name in sorted(os.listdir(inputDirectory))]

        workers = max(1, mp.cpu_count()-1)
        workers = min(len(workItems), workers)
        self.logger.info("[mp silence_cutter] workers: "+str(workers))

        pool = mp.Pool(workers)
        pool.map(processingTask, workItems)
        pool.close()
        pool.join()

        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))

