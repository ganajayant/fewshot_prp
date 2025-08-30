from pathlib import Path

import pyterrier as pt

pt.init()

datasets = [
    # {"name": "irds:beir/scifact/test", "fields": ["text", "title"]},
    {"name": "irds:cord19/trec-covid", "fields": ["title", "doi", "date", "abstract"]},
    # {"name": "irds:msmarco-passage/trec-dl-2020/judged", "fields": ["text"]},
    # {"name": "irds:msmarco-passage/trec-dl-2019/judged", "fields": ["text"]},
    # {"name": "msmarco_passage", "fields": ["text"]},
]

base_index_path = Path.home() / "Datasets"

for dataset_config in datasets:
    dataset_name = dataset_config["name"]
    fields_to_index = dataset_config["fields"]

    dataset_path_name = dataset_name.replace(":", "_").replace("/", "_")
    index_path = base_index_path / f"{dataset_path_name}_index"

    print(f"Indexing {dataset_name} to {index_path}...")

    dataset = pt.get_dataset(dataset_name)
    indexer = pt.IterDictIndexer(str(index_path), fields=fields_to_index)
    indexer.index(dataset.get_corpus_iter())
    print(f"Indexing complete for {dataset_name}.")
    print("-" * 30)
