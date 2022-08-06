
import os
import shutil
import json
import traceback

import faiss
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import sklearn

# Not a model, but it was easier to just integrate the code this way


class SpeakerSearch(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(SpeakerSearch, self).__init__()

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
        return self.search(data, websocket)


    async def search(self, data, websocket):


        inPath, inPath2, outputDirectory = data["inPath"], data["inPath2"], data["outputDirectory"]
        shutil.rmtree(outputDirectory, ignore_errors=True)


        input_search_files = [f'{inPath}/{file}' for file in list(os.listdir(inPath)) if ".wav" in file]
        input_search_files_fnames = [fname.split("/")[-1] for fname in input_search_files]


        files = [f'{inPath}/{file}' for file in list(os.listdir(inPath)) if ".wav" in file]
        embeddings_queries = []

        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Encoding query audio files...'}))

        for fi, file in enumerate(files):
            fpath = Path(file)
            wav = preprocess_wav(fpath)
            embedding = self.encoder.embed_utterance(wav)
            embeddings_queries.append(embedding)


        files = [f'{inPath2}/{file}' for file in list(os.listdir(inPath2)) if ".wav" in file]
        SKIP_SAME_QUERY_NAMES = True
        if SKIP_SAME_QUERY_NAMES:
            files = [fname for fname in files if fname.split("/")[-1] not in input_search_files_fnames]

        embeddings_corpus = []
        files_done = []


        for fi, file in enumerate(files):

            try:
                if fi%10==0 or fi==len(files)-1:
                    if websocket is not None:
                        await websocket.send(json.dumps({"key": "task_info", "data": f'Encoding corpus audio files: {fi+1}/{len(files)}  ({(int(fi+1)/len(files)*100*100)/100}%)   '}))

                fpath = Path(file)
                wav = preprocess_wav(fpath)
                embedding = self.encoder.embed_utterance(wav)
                embeddings_corpus.append(embedding)
                files_done.append(file)
            except:
                self.logger.info(traceback.format_exc())




        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Building faiss index...'}))



        if len(embeddings_corpus)==0:
            await websocket.send(json.dumps({"key": "tasks_error", "data": f'No files given to search over ("corpus" folder)'}))
            return
        if len(embeddings_queries)==0:
            await websocket.send(json.dumps({"key": "tasks_error", "data": f'No query files given to search with ("query" folder)'}))
            return


        pool_scores = [0 for _ in range(len(embeddings_corpus))]

        pool_features = np.array(embeddings_corpus).astype(np.float32)
        query_features = np.array(embeddings_queries).astype(np.float32)
        index = faiss.IndexFlatL2(pool_features.shape[1])
        index.add(pool_features)
        num_results = int(pool_features.shape[0])


        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Running faiss...'}))

        D, I = index.search(query_features, num_results)

        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Post-processing faiss...'}))

        # For every query
        for query_i in range(query_features.shape[0]):
            # Go through all the sorted result indeces
            for res_i in range(num_results):
                # And increment a score for each data item in the pool, by the distance
                pool_scores[I[query_i][res_i]] += D[query_i][res_i]



        sort_index = np.argsort((np.array(pool_scores)))
        os.makedirs(outputDirectory)

        for sii, si in enumerate(sort_index):

            if websocket is not None:
                await websocket.send(json.dumps({"key": "task_info", "data": f'Sorting and outputting pool data...  {sii+1}/{len(sort_index)}  ({(int(sii+1)/len(sort_index)*100*100)/100}%)   '}))

            pool_file = files_done[si]
            out_name = f'{str(sii).zfill(6)}_{pool_file.split("/")[-1]}'
            shutil.copyfile(pool_file, f'{outputDirectory}/{out_name}')


        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))