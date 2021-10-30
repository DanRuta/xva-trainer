
import os
import shutil
import json
import traceback

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import sklearn

# Not a model, but it was easier to just integrate the code this way


class ClusterSpeakers(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(ClusterSpeakers, self).__init__()

        self.logger = logger
        self.PROD = PROD
        self.models_manager = models_manager
        self.device = device
        self.ckpt_path = None

        self.encoder = VoiceEncoder()

        self.model = None
        self.isReady = True


    def load_state_dict (self, ckpt_path, sd):
        pass

    def set_device (self, device):
        pass

    def runTask (self, data, websocket=None):
        return self.cluster(data, websocket)


    async def cluster(self, data, websocket):

        self.logger.info("data: "+ str(", ".join(list(data.keys()))))

        inPath, outputDirectory = data["inPath"], data["outputDirectory"]


        files = [f'{inPath}/{file}' for file in list(os.listdir(inPath)) if ".wav" in file]
        embeddings = []
        files_done = []

        for fi, file in enumerate(files):

            if websocket is not None:
                await websocket.send(json.dumps({"key": "task_info", "data": f'Encoding audio files: {fi+1}/{len(files)}  ({(int(fi+1)/len(files)*100*100)/100}%)   '}))

            fpath = Path(file)
            wav = preprocess_wav(fpath)
            embedding = self.encoder.embed_utterance(wav)
            embeddings.append(embedding)
            files_done.append(file)


        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Clustering...'}))

        embeddings = np.vstack(embeddings)
        standalone_ap = sklearn.cluster.AffinityPropagation()
        clusters = standalone_ap.fit_predict(embeddings)

        cluster_centers_indices = standalone_ap.cluster_centers_indices_
        labels = standalone_ap.labels_
        n_clusters_ = len(cluster_centers_indices)

        for ci, c in enumerate(clusters):
            if websocket is not None:
                await websocket.send(json.dumps({"key": "task_info", "data": f'Copying files into {n_clusters_} clusters...  {ci+1}/{len(clusters)}  ({(int(ci+1)/len(clusters)*100*100)/100}%)   '}))
            os.makedirs(f'{outputDirectory}/{c}', exist_ok=True)
            shutil.copyfile(files_done[ci], f'{outputDirectory}/{c}/{files_done[ci].split("/")[-1]}')

        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))