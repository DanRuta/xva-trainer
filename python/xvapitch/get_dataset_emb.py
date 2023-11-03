import os
import numpy as np
from sklearn.cluster import KMeans


def get_emb(dataset_embs_path, main_emb_outpath, other_embs_outpath):

    if os.path.exists(main_emb_outpath) and os.path.exists(other_embs_outpath):
        with open(main_emb_outpath, "r") as f:
            centroid_emb = np.array([float(v) for v in f.read().split(",")])
        with open(other_embs_outpath, "r") as f:
            other_centroids = []
            for line in f.read().split("\n"):
                other_centroids.append(np.array([float(v) for v in line.split(",")]))

    else:
        embs = sorted(os.listdir(dataset_embs_path))
        embs_fnames = [emb for emb in embs if emb.endswith(".npy")]

        embs = []
        for ei,emb in enumerate(embs_fnames):
            print(f'\rLoading embeddings {ei+1}/{len(embs_fnames)}  ', end="", flush=True)
            embs.append(np.load(f'{dataset_embs_path}/{emb}'))
        print("")

        try:
            n_clusters = 10
            if len(embs)>5000:
                embs = random.sample(embs, 5000) # This should be enough - reduce memory limitations on some systems
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embs)

            clusters = kmeans.labels_
            cluster_centers_ = kmeans.cluster_centers_

            cluster_counts = {}

            for vi,val in enumerate(clusters):
                if val not in cluster_counts.keys():
                    cluster_counts[val] = []
                cluster_counts[val].append(val)

            largest_cluster = 0
            largest_cluster_count = 0
            for ki,key in enumerate(sorted(cluster_counts.keys())):
                if len(cluster_counts[key])>largest_cluster_count:
                    largest_cluster_count = len(cluster_counts[key])
                    largest_cluster = ki

            centroid_emb = cluster_centers_[largest_cluster]
            other_centroids = [cluster_centers_[i] for i in range(len(cluster_centers_)) if i!=largest_cluster]

        except BaseException:
            centroid_emb = random.sample(embs, 1)[0]
            other_centroids = random.sample(embs, 10)


        with open(main_emb_outpath, "w+") as f:
            f.write(",".join([str(val) for val in centroid_emb]))
        with open(other_embs_outpath, "w+") as f:
            out_text = []
            for emb in other_centroids:
                out_text.append(",".join([str(val) for val in emb]))
            f.write("\n".join(out_text))

    return centroid_emb, other_centroids



import pickle
def get_similar_priors(target_emb, dataset_roots, output_path, languages):
    import faiss

    cache_samples_path = f'{output_path}/similar_priors_datalist.txt'
    if os.path.exists(cache_samples_path):
        with open(cache_samples_path, encoding="utf8") as f:
            datalist = f.read().split("\n")
        return datalist

    datalist = []
    langs_datasets = {}
    for dataset_root in dataset_roots:
        datasets = sorted(os.listdir(dataset_root))
        for dataset in datasets:
            if "_" in dataset and "." not in dataset and not dataset.startswith("_"):
                if dataset.split("_")[0] not in langs_datasets.keys():
                    langs_datasets[dataset.split("_")[0]] = []
                langs_datasets[dataset.split("_")[0]].append(f'{dataset_root}/{dataset}')

    languages = [lang for lang in sorted(list(langs_datasets.keys())) if lang in languages]
    for lang in languages:
        # language_samples = []
        # print(f'Gathering similar sounding samples to target voice from the multi-lingual priors datasets | Language: {lang}')

        cache_path = f'{output_path}/emb_cache_{lang}.pkl'
        if os.path.exists(cache_path):
            print(f'\rLoading cached similar sounding samples to target voice from the multi-lingual priors datasets | Language: {lang}    ', end="", flush=True)
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                language_data_transcript, language_data_paths, language_data_embs = data
        else:
            language_data_transcript = {}
            language_data_paths = []
            language_data_embs = []

            for di,dataset in enumerate(langs_datasets[lang]):
                emb_files = sorted(os.listdir(f'{dataset}/se_embs'))
                language_data_transcript[dataset.split("/")[-1]] = {}

                with open(f'{dataset}/metadata.csv') as f:
                    metadata = f.read().split("\n")
                    for line in metadata:
                        if "|" in line:
                            language_data_transcript[dataset.split("/")[-1]][line.split("|")[0]] = line.split("|")[1]

                for ei,emb_file in enumerate(emb_files):
                    if emb_file.replace(".npy", ".wav") in language_data_transcript[dataset.split("/")[-1]].keys():
                        emb = np.load(f'{dataset}/se_embs/{emb_file}')

                        if ei==0 or ei%1000 or (ei+1)==len(emb_files):
                            print(f'\rGathering similar sounding samples to target voice from the multi-lingual priors datasets | Language: {lang} | Dataset {di+1}/{len(langs_datasets[lang])} ({round((ei+1)/len(emb_files)*100)}%)     ', end="", flush=True)

                        language_data_paths.append(f'{dataset}/wavs/{emb_file.replace(".npy", ".wav")}')
                        language_data_embs.append(emb)

            with open(cache_path, "wb+") as f:
                pickle.dump([language_data_transcript, language_data_paths, language_data_embs], f)


        language_data_embs = np.array(language_data_embs)
        print(f'\nBuilding faiss index over {language_data_embs.shape[0]} items...')
        language_data_embs = language_data_embs.astype(np.float32)
        index = faiss.IndexFlatL2(language_data_embs.shape[1])
        index.add(language_data_embs)

        target_num_samples = 2000

        query_emb = np.stack([target_emb]).astype(np.float32)
        D, I = index.search(query_emb, target_num_samples)

        results = I[0, :]
        # query_index = results[0]
        results_indexes = results[1:]

        for ri,res_ind in enumerate(results_indexes):
            dataset = language_data_paths[ri].split("/")[-3]
            fname = language_data_paths[ri].split("/")[-1]
            datalist.append(f'{language_data_transcript[dataset][fname]}|{language_data_paths[ri]}|{dataset}|{lang}')

    with open(cache_samples_path, "w+", encoding="utf8") as f:
        f.write("\n".join(datalist))





if __name__ == '__main__':
    get_emb(f'D:/DEBUG_JOKER/en_me_joker_M/se_embs')