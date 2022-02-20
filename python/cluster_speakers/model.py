
import os
import shutil
import json
import traceback

import torch
import faiss
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import sklearn
from sklearn.cluster import KMeans

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

        inPath, outputDirectory = data["inPath"], data["outputDirectory"]
        do_search_reordering = data["toolSettings"]["do_search_reordering"]
        use_custom_k = data["toolSettings"]["use_custom_k"]
        custom_k = int(data["toolSettings"]["custom_k"])
        use_min_cluster_size = data["toolSettings"]["use_min_cluster_size"]
        min_cluster_size = int(data["toolSettings"]["min_cluster_size"])
        use_cluster_folder_prefix = data["toolSettings"]["use_cluster_folder_prefix"]
        cluster_folder_prefix = data["toolSettings"]["cluster_folder_prefix"]




        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Gathering audio files...'}))

        files = [f'{inPath}/{file}' for file in list(os.listdir(inPath)) if ".wav" in file]
        embeddings = []
        files_done = []

        with torch.no_grad():
            for fi, file in enumerate(files):

                if websocket is not None:
                    websocket.send(json.dumps({"key": "task_info", "data": f'Encoding audio files: {fi+1}/{len(files)}  ({int((fi+1)/len(files)*100*100)/100}%) '}))

                try:
                    fpath = Path(file)
                    wav = preprocess_wav(fpath)
                    embedding = self.encoder.embed_utterance(wav)
                    embeddings.append(embedding)
                    files_done.append(file)
                except:
                    self.logger.info(traceback.format_exc())




        if websocket is not None:
            await websocket.send(json.dumps({"key": "task_info", "data": f'Clustering...'}))

        embeddings = np.vstack(embeddings)

        if use_custom_k and custom_k>=2:
            kmeans = KMeans(n_clusters=custom_k, random_state=0).fit(embeddings)
            clusters = kmeans.labels_
            n_clusters_ = custom_k
        else:
            standalone_ap = sklearn.cluster.AffinityPropagation()
            clusters = standalone_ap.fit_predict(embeddings)
            cluster_centers_indices = standalone_ap.cluster_centers_indices_
            n_clusters_ = len(cluster_centers_indices)


        clusters_with_less_than_min = []
        if use_min_cluster_size:

            cluster_counts = [0 for c in range(n_clusters_)]

            for ci, c in enumerate(clusters):
                cluster_counts[c] += 1

            clusters_with_less_than_min = []

            for ci, c in enumerate(cluster_counts):
                if c<min_cluster_size:
                    clusters_with_less_than_min.append(ci)


        if do_search_reordering:

            if websocket is not None:
                await websocket.send(json.dumps({"key": "task_info", "data": f'Running principal cluster similarity re-ranking...'}))

            # Gather up all the files into their clusters
            clusters_files = [[] for c in range(n_clusters_)]
            clusters_embeddings = [[] for c in range(n_clusters_)]
            for ci, c in enumerate(clusters):
                if c not in clusters_with_less_than_min:
                    clusters_files[c].append(files_done[ci])
                    clusters_embeddings[c].append(embeddings[ci])


            # Figure out which cluster is the largest
            largest_c = 0
            largest_c_i = -1

            for ci, cf in enumerate(clusters_files):
                if len(cf) > largest_c:
                    largest_c = len(cf)
                    largest_c_i = ci

            # Separate embeddings into query and corpus, maintaining the individual corpus embeddings mappings to their cluster
            query_embs = clusters_embeddings[largest_c_i]
            corpus_embs = []
            corpus_embs_cluster_indices = []
            for ci, c_embs in enumerate(clusters_embeddings):
                if ci != largest_c_i:
                    for emb in c_embs:
                        corpus_embs.append(emb)
                        corpus_embs_cluster_indices.append(ci)


            # Build and run faiss
            pool_features = np.array(corpus_embs).astype(np.float32)
            query_features = np.array(query_embs).astype(np.float32)
            index = faiss.IndexFlatL2(pool_features.shape[1])
            index.add(pool_features)
            num_results = int(pool_features.shape[0])

            D, I = index.search(query_features, num_results)

            pool_scores = [0 for _ in range(len(corpus_embs))]
            # For every query
            for query_i in range(query_features.shape[0]):
                # Go through all the sorted result indeces
                for res_i in range(num_results):
                    # And increment a score for each data item in the pool, by the distance
                    pool_scores[I[query_i][res_i]] += D[query_i][res_i]


            cluster_scores = [[] for _ in clusters_files] # The scores, assigned to each original cluster (incl query, but we'll ignore that)
            # Go through the summed distance scores for each line, and assign them to their cluster folder
            for psi, ps in enumerate(pool_scores):
                cluster_scores[corpus_embs_cluster_indices[psi]].append(ps)

            # Average up the scores for all cluster folders
            cluster_scores = [np.mean(scores) for scores in cluster_scores]

            # In ascending distance order, output the clusters out in the new order
            sort_index = np.argsort((np.array(cluster_scores)))


            # Output the query first, as that will be #1
            os.makedirs(f'{outputDirectory}/0', exist_ok=True)
            for fpath in clusters_files[largest_c_i]:
                shutil.copyfile(fpath, f'{outputDirectory}/0/{fpath.split("/")[-1]}')


            n_clusters = len(cluster_scores)+1
            folder_out_index = 1
            for sii, si in enumerate(sort_index):

                # Ignore the query files, as we've already output them first
                if si==largest_c_i:
                    continue

                if websocket is not None:
                    websocket.send(json.dumps({"key": "task_info", "data": f'Copying files into {n_clusters} clusters...  {ci+1}/{n_clusters}  ({(int(ci+1)/n_clusters*100*100)/100}%)'}))

                if len(clusters_files[si]):
                    os.makedirs(f'{outputDirectory}/{cluster_folder_prefix+"_" if use_cluster_folder_prefix else ""}{folder_out_index}', exist_ok=True)

                    for fpath in clusters_files[si]:
                        shutil.copyfile(fpath, f'{outputDirectory}/{cluster_folder_prefix+"_" if use_cluster_folder_prefix else ""}{folder_out_index}/{fpath.split("/")[-1]}')

                    folder_out_index += 1

        else:

            # No clusters were found
            if n_clusters_==0:

                clusters_files = [[]]
                for ci, c in enumerate(clusters):
                    clusters_files[0].append(files_done[ci])

            else:
                # Gather up all the files into their clusters
                clusters_files = [[] for c in range(n_clusters_)]
                for ci, c in enumerate(clusters):
                    if c not in clusters_with_less_than_min:
                        clusters_files[c].append(files_done[ci])
                clusters_files = [cluster for cluster in clusters_files if len(cluster)]




            for ci, cf in enumerate(clusters_files):

                if websocket is not None:
                    websocket.send(json.dumps({"key": "task_info", "data": f'Copying files into {len(clusters_files)} clusters...  {ci+1}/{len(clusters_files)}  ({(int(ci+1)/len(clusters_files)*100*100)/100}%)'}))

                os.makedirs(f'{outputDirectory}/{cluster_folder_prefix+"_" if use_cluster_folder_prefix else ""}{ci}', exist_ok=True)

                for fpath in clusters_files[ci]:
                    shutil.copyfile(fpath, f'{outputDirectory}/{cluster_folder_prefix+"_" if use_cluster_folder_prefix else ""}{ci}/{fpath.split("/")[-1]}')


        del clusters, embeddings

        if websocket is not None:
            await websocket.send(json.dumps({"key": "tasks_next"}))


