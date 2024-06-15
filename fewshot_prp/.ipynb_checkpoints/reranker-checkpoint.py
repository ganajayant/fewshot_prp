import torch
#from torch.nn import functional as F
import pandas as pd
import os
import json
from tqdm.auto import tqdm
import numpy as np
import random as random
from difflib import SequenceMatcher
import pyterrier as pt
from pyterrier.model import add_ranks
from models.rankers import AllPair
from evaluation.evaluation import ModelEvaluation
import configuration
import gc

#from huggingface_hub import login
#login("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")

class RelevancyScorerLLM():
    global tokenizer
    global model

    def transform(self,
                  topics_or_res : pd.DataFrame,
                  few_shot_examples: list):
        res = {
            'qid': [],
            'query': [],
            'docid': [],
            'docno': [],
            'text': [],
            'score': [],
        }
        query_num = 1

        prediction_logs = []
        with torch.no_grad():
            for (qid, query), query_results in tqdm(topics_or_res.groupby(['qid', 'query']), unit='q'):
                # for all-pair processing:
                self.parameters['query_num'] = query_num
                doc_no, doc_ids, doc_texts, scores = AllPair(self.parameters, qid, query, query_results, few_shot_examples)
                query_num+=1
                res['qid'].extend([qid] * len(doc_ids))
                res['query'].extend([query] * len(doc_ids))
                res['docid'].extend(doc_ids)
                res['docno'].extend(doc_no)
                res['text'].extend(doc_texts)
                res['score'].extend(scores)
                
        return pd.DataFrame(res)

    def store_res(res, file_name):
        res = add_ranks(res)
        file_name=f'{file_name}.res'
        pt.io.write_results(res, file_name, format='trec', append=False)

    def __init__(self, model, tokenizer, reranker: str = 'zephyr', top_k: int = 100, ds: str = 'dl19', kshot: int = 0, random_seed: int = 42, mode: str = 'LEX', evaluate: bool  = False, r_model: str = 'bm25', retriever: str = 'AP'):
        
        self.random_seed = random_seed
        random.seed(self.random_seed)
        
        self.kshot = kshot
        self.retriever = retriever
        self.max_fewshot = configuration.max_fewshot
        self.duration = []
        
        if self.kshot != 0:
            self.mode = mode
        else:
            self.mode = 'ZER'

        data_path = configuration.data_path
        topics_or_res = pd.read_csv(data_path + f'phase_one_retrieval/ranked_{r_model}_{ds}_t{top_k}.csv')
        self.parameters = {}

        #true static random example
        self.example_static = []

        self.device = "cuda:0" if torch.cuda.is_available() else "CPU"
        self.model = model
        self.tokenizer = tokenizer

        if reranker=='zephyr':
            self.A = self.tokenizer.encode("1", return_tensors="pt", add_special_tokens=False)[0][1] 
            self.B = self.tokenizer.encode("2", return_tensors="pt", add_special_tokens=False)[0][1] 
        else:
            self.A = self.tokenizer.encode("1", return_tensors="pt", add_special_tokens=False)[0].item() 
            self.B = self.tokenizer.encode("2", return_tensors="pt", add_special_tokens=False)[0].item()

        if mode=='LEX' or mode=='LEXRO':
            sm = 'bm25'
        elif mode=='SEM' or mode=='SEMRO':
            sm= 'bert'
        elif mode=='STC' or mode=='STCRO':
            sm= 'static'

        # set the parameters
        self.parameters = {
            'device': self.device,
            'model': self.model,
            'tokenizer': self.tokenizer,
            'A': self.A,
            'B': self.B,
            'kshot':self.kshot,
            'mode': self.mode,
            'seed': self.random_seed,
            'query_num': 0,
            'dataset':ds,
            'duration': self.duration,
            'reranker': reranker
        }

        if kshot!=0:
            try:
                file = open(data_path + f'kshots/{sm}_fewshot-{ds}.json')
                print(f'{sm}_fewshot-{ds} File Selected')
                few_shot_examples = json.load(file)
                file.close()
            except:
                print('Warning: Fewshot example file missing!')
            
            scored_res = self.transform(topics_or_res, few_shot_examples)
        else:
            scored_res = self.transform(topics_or_res, [])

        file_name = f'scores/reranking_scores_{ds}_t{top_k}_{reranker}_{self.mode}{self.max_fewshot}_R{random_seed}_{self.kshot}-shot'

        scored_res.to_csv(f'{file_name}.csv')
        RelevancyScorerLLM.store_res(scored_res, file_name)

        if evaluate:
            # per-query execution time
            duration = self.parameters['duration']
            mean = sum(duration) / len(duration) 
            variance = sum([((x - mean) ** 2) for x in duration]) / len(duration) 
            res = variance ** 0.5
            pm = u"\u00B1"
            print(f"Inference time: {str(round(mean,4))} {pm} {str(round(res,4))}")

            # Print all evaluation results
            output = ModelEvaluation.evaluate(r_model)
            print(output)