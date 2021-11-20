
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


class SpeakerClusterSearch(object):
    def __init__(self, logger, PROD, device, models_manager):
        super(SpeakerClusterSearch, self).__init__()

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

        for fi, file in enumerate(files):

            if websocket is not None:
                await websocket.send(json.dumps({"key": "task_info", "data": f'Encoding query audio files: {fi+1}/{len(files)}  ({(int((fi+1)/len(files)*100*100))/100}%)   '}))

            fpath = Path(file)
            wav = preprocess_wav(fpath)
            embedding = self.encoder.embed_utterance(wav)
            embeddings_queries.append(embedding)




        # folders = [fdname for fdname in sorted(os.listdir(inPath2)) if "." not in fdname]
        folders = [fdname for fdname in sorted(os.listdir(inPath2))]
        folders_files = []
        embeddings_corpus = []
        files_done = []
        files_done_folderIndex = []

        # nested_structure = []

        folders_with_files = set()
        for fdname in folders:
            for file in list(os.listdir(f'{inPath2}/{fdname}')):
                if ".wav" in file:
                    folders_with_files.add(f'{inPath2}/{fdname}')
                    # nested_structure.


                    # Need to adjust for the nested structure of the folders now being flat, on the second leve. Need to assign indeces accordingly for when the files get copied across


                else:
                    sub_files = os.listdir(f'{inPath2}/{fdname}/{file}')
                    for sf_file in sub_files:
                        if ".wav" in sf_file:
                            folders_with_files.add(f'{inPath2}/{fdname}/{file}')
        folders_with_files = list(folders_with_files)


        # for fdi, fdname in enumerate(folders):
        for fdi, fdname in enumerate(folders_with_files):

            # files = []
            # for file in list(os.listdir(f'{inPath2}/{fdname}')):
            #     if ".wav" in file:
            #         files.append(f'{inPath2}/{fdname}/{file}')
            #     else:
            #         sub_files = os.listdir(f'{inPath2}/{fdname}/{file}')
            #         self.logger.info(f'folder: {inPath2}/{fdname}/{file} | sub_files: {len(sub_files)}')
            #         for sf_file in sub_files:
            #             if ".wav" in sf_file:
            #                 files.append(f'{inPath2}/{fdname}/{file}/{sf_file}')

            # self.logger.info(f'folder: {inPath2}/{fdname} | files: {len(files)}')

            if websocket is not None:
                await websocket.send(json.dumps({"key": "task_info", "data": f'Encoding corpus cluster audio files: {fdi+1}/{len(folders_with_files)}  ({(int(fdi+1)/len(folders_with_files)*100*100)/100}%)   '}))


            # files = [f'{inPath2}/{fdname}/{file}' for file in list(os.listdir(f'{inPath2}/{fdname}')) if ".wav" in file]
            files = [f'{fdname}/{file}' for file in list(os.listdir(fdname)) if ".wav" in file]

            SKIP_SAME_QUERY_NAMES = True
            # Don't use files with the same name as input query names
            if SKIP_SAME_QUERY_NAMES:
                files = [fname for fname in files if fname.split("/")[-1] not in input_search_files_fnames]

            folders_files.append(files)

            for fi, file in enumerate(files):
                try:
                    if websocket is not None:
                        await websocket.send(json.dumps({"key": "task_info", "data": f'Indexing corpus | Folder: {fdi+1}/{len(folders_with_files)} | File: {fi+1}/{len(files)}  ({(int((fdi+1)/len(folders_with_files)*100*100))/100}%)'}))
                    # print(f'\rIndexing pool | {fdi+1}/{len(folders)} | {fi+1}/{len(files)}  ({(int((fdi+1)/len(folders)*100*100))/100}%)   ', end="", flush=True)

                    fpath = Path(file)
                    wav = preprocess_wav(fpath)

                    embed = self.encoder.embed_utterance(wav)

                    embeddings_corpus.append(embed)
                    files_done.append(file)
                    files_done_folderIndex.append(fdi)
                except KeyboardInterrupt:
                    exit()
                except:
                    import traceback
                    print(traceback.format_exc())
                    self.logger.info(traceback.format_exc())





        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Building faiss index...'}))



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





        folder_scores = [[] for _ in folders_with_files]

        # sort_index = np.argsort((-np.array(pool_scores)))


        # Go through the summed distance scores for each line, and assign them to their cluster folder
        for psi, ps in enumerate(pool_scores):
            # folder_scores[psi].append(ps)
            folder_scores[files_done_folderIndex[psi]].append(ps)

        # Average up the scores for all cluster folders
        folder_scores = [np.mean(scores) for scores in folder_scores]

        # In ascending distance order, output the clusters out in the new order
        sort_index = np.argsort((np.array(folder_scores)))

        self.logger.info(f'sort_index: {len(sort_index)}')
        self.logger.info(f'folders_files: {len(folders_files)}')
        self.logger.info(f'folder_scores: {len(folder_scores)}')
        self.logger.info(f'pool_scores: {len(pool_scores)}')
        self.logger.info(f'files_done_folderIndex: {len(files_done_folderIndex)}')
        self.logger.info(f'folders_with_files: {len(folders_with_files)}')

        for sii, si in enumerate(sort_index):

            if websocket is not None:
                await websocket.send(json.dumps({"key": "task_info", "data": f'Sorting and outputting pool clusters data...  {sii+1}/{len(sort_index)}  ({(int((sii+1)/len(sort_index)*100*100))/100}%)   '}))

            os.makedirs(f'{outputDirectory}/{sii}', exist_ok=True)

            for fpath in folders_files[si]:
                shutil.copyfile(fpath, f'{outputDirectory}/{sii}/{fpath.split("/")[-1]}')


        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))