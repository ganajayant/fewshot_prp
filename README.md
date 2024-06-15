# fewshot_prp
A few-shot implementation of Pairwise Ranking Prompting (PRP) method with a All-Pair re-ranking mechanism.

# Installation
pip install dist/fewshot_prp-0.1-py3-none-any.whl

# A sample execution
python main.py reranker=flanxl top_k=20 datasets=dl19,dl20 kshots=0,1 seed=42 modes=SEM,LEX eval=False