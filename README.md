# fewshot_prp
A few-shot implementation of Pairwise Ranking Prompting (PRP) method with a All-Pair re-ranking mechanism.

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

## Evaluate all the baselines phase-one retriever (BM25)
```ruby
from fewshot_prp.evaluation.evaluation import ModelEvaluation
from pyterrier.measures import *

metrics = [AP(rel=2)@100, NDCG(cutoff=10)]

output = ModelEvaluation.evaluate('baselines', metrics, False, 1.0, 1.0, False, False)
display(output)
```

## Evaluate the re-rankers (Zephyr,FlanXL)

```ruby
from fewshot_prp.evaluation.evaluation import ModelEvaluation
from pyterrier.measures import *

metrics = [AP(rel=2)@100, NDCG(cutoff=10)]
w = 0.70

output = ModelEvaluation.evaluate('bm25', metrics, True, w, 1-w, False, False)
```

> **_NOTE:_** All the metrics supported by my terrier can be used. 
#                                                   FS     W1   W2    FB    RR
metrics = [AP(rel=2)@100, NDCG(cutoff=1), NDCG(cutoff=5), NDCG(cutoff=10), NDCG(cutoff=100), P@10]
