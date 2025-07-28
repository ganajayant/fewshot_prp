# fewshot_prp
A few-shot implementation of Pairwise Ranking Prompting (PRP) method with an All-Pair re-ranking mechanism.
> Our Fewshot PRP paper has been accepted at EMNLP-2024 Findings!

# Installation

To use the fewshot_prp functions directly, install using pip
```
pip install dist/fewshot_prp-0.1-py3-none-any.whl
```

Currently, for research and development, clone this repo and install it as editable
```
git clone fewshot_prp.git
cd fewshot_prp
pip install -e .
```

# A sample execution

For proper functionality, it is expected that the configuration file is correctly configured. 
The phase-one retriever first retrieves the top-k documents (the index needs to be set correctly) for the selected datasets.
The LLM reranker then re-ranks the documents.
Evaluation can be manually performed or set the 'eval' parameter for default evaluation.

> **_NOTE:_** This package is a complete pipeline with phase-one retriever and re-ranker.

## Configuration
A configuration.py file contains all the key parameters for the phase-one retriever and the re-ranker. 

- The phase-one is a BM25 retriever with default parameters b=0.75 and k1=1.2. To change these parameters, modify bm25_b and bm25_k1.
- The index location needs to be specified in index_path; check and configure index locations under datasets.
- The cache location may be specified in cache_path (for transformer models).
- To set the maximum number of few-shot examples, consider changing the max_fewshot.

## Execution parameters
The following parameters are accepted by fewshot_prp
- The 'reranker' parameter sets the LLM currently supports 'zephyr' and 'flanxl'.
- The 'top-k' parameter specifies the re-ranking set, which is first retrieved using BM25 and later re-ranked using the selected LLM reranker.
- Currently supported datasets include 'dl19' (TREC DL'2019), 'dl20' (TREC DL'2020), 'covid' (TREC-COVID) and 'scifact' (SciFact).
- The 'k-shots' parameter accepts values '>=0'; 0 = zero-shot, 1 = one-shot, ... and so on.
- To get consistent output from random experiments, use the 'seed' parameter.
- Currently, we support examples samples collected from MS MARCO using BM25 'LEX' using bert 'SEM', and static examples 'STC.' Select any of these modes for few-shot.
- To try the relevant-document-only mode of re-ranking, use the 'RO' prefix with modes like 'LEXRO', 'SEMRO' and 'STCRO.'
- For direct evaluation after re-ranking, set the 'eval' parameter to True. (defaults outputs 'MAP(rel=2)@100' and 'nDCG@10' metrics)

> **_NOTE:_** The SEM, LEX, and STC example sets with 10 examples per query are already provided in this package for TREC DL'2019 and TREC DL'2020.  For TREC-Covid and SciFact, only SEM and LEX example sets are provided.

## Run
An example zero-shot and one-shot re-ranking (LLM = Zephyr) of top-100 documents from TREC DL'2019 and TREC DL'2020 in SEM and LEX mode and setting for direct evaluation (we consider a seed value of 42).
```ruby
python fewshot-prp reranker=zephyr top_k=100 datasets=dl19,dl20 kshots=0,1 seed=42 modes=SEM,LEX eval=True
```

# Evaluation

The output files must be present in the following location:

```
-- fewshot_prp
  -- datasets
     -- phase_one_retreival
        -- <baselines>
  -- scores
     -- <re-ranker outputs>
```

The evaluation model parameters:
```
ModelEvaluation.evaluate(<key>, <metrics>, <FS>, W1, W2, <FB>, <RR>)
```
- key : Either 'baseline' or 'bm25'
- Fusion Sum (FS): Linear combined with weights W1 and W2 (W2 = 1 - W1). Use either 'True' or 'False' along with weights, e.g. ( W1=0.7 and W2=1-W1 ).
- Fall Back (FB): Fall back to BM25 ranks in case of conflicts/ties. Use either 'True' or 'False'.
- Reciprocal Rank (RR): Uses rank reciprocals to re-rank the documents. Use either 'True' or 'False'

## Evaluate all the baselines phase-one retriever (BM25)

Use key as 'baselines'

```ruby
from fewshot_prp.evaluation.evaluation import ModelEvaluation
from pyterrier.measures import *

metrics = [AP(rel=2)@100, NDCG(cutoff=10)]

output = ModelEvaluation.evaluate('baselines', metrics, False, 0.7, 0.3, False, False)
display(output)
```

## Evaluate the re-rankers (Zephyr,FlanXL)

Use key as 'bm25'. For re-ranking conflicts and ties, we used Fusion Sum with W1=0.7 in all the experiments.

```ruby
from fewshot_prp.evaluation.evaluation import ModelEvaluation
from pyterrier.measures import *

metrics = [AP(rel=2)@100, NDCG(cutoff=10)]
w = 0.70

output = ModelEvaluation.evaluate('bm25', metrics, True, w, 1-w, False, False)
display(output)
```

> **_NOTE:_** All the metrics supported by pyterrier can be used.

---
## 🙏 Citation

```bibtex
@inproceedings{sinhababu-2024-fsprp,
    title = "Few-shot Prompting for Pairwise Ranking: An Effective Non-Parametric Retrieval Model",
    author = {Sinhababu, Nilanjan and Parry, Andrew and Ganguly, Debasis and Samanta, Debasis and Mitra, Pabitra},
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.720/",
    doi = "10.18653/v1/2024.findings-emnlp.720",
    pages = "12363--12377"
}
```
