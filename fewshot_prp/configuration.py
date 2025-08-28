###################################################################################
#                                                                                 #
#                          FEW-SHOT PRP CONFIGURATION                             #
#                                                                                 #
###################################################################################

import os

# Current working directory
cwd = os.getcwd()

# location of the datasets
data_path = f"{cwd}/datasets/"

# Location of the results
res_path = f"{cwd}/scores/"

# passage index location
index_path = os.path.join(os.path.expanduser("~"), "Datasets/index/")

# path to models cache
cache_path = os.path.join(os.path.expanduser("~"), "cache")

# parameters for BM25
bm25_b = 0.75
bm25_k1 = 1.2

# maximum fewshots examples
max_fewshot = 5

# dataset_configuration
datasets = {
    "msmarco_passage": {
        "name": "msmarco_passage",
        "topics": "train",
        "index": f"{index_path}trecdl_index/",
    },
    "dl19": {
        "name": "irds:msmarco-passage/trec-dl-2019/judged",
        "topics": "text",
        "index": f"{index_path}trecdl_index/",
    },
    "dl20": {
        "name": "irds:msmarco-passage/trec-dl-2020/judged",
        "topics": "text",
        "index": f"{index_path}trecdl_index/",
    },
    "covid": {
        "name": "irds:cord19/trec-covid",
        "topics": "title",
        "index": f"{index_path}beir_covid_index/",
    },
    "scifact": {
        "name": "irds:beir/scifact/test",
        "topics": "text",
        "index": f"{index_path}beir_scifact_index/",
    },
}

rerankers = {
    "zephyr": {
        "model_name": "HuggingFaceH4/zephyr-7b-beta",
        "tokenizer": "HuggingFaceH4/zephyr-7b-beta",
    },
    "flanxl": {"model_name": "google/flan-t5-xl", "tokenizer": "google/flan-t5-xl"},
}
