import pandas as pd
import numpy as np
import time
import sys
from .llm_generator import LLMGenerator
sys.path.append("..")
from prompts.prompt import GeneratePrompt

class AllPair:

    def __init__(self, parameters):
        self.print_flag = -1 # set it 0 for per-query verbose

    def __new__(self, parameters, qid, query : str, query_results : pd.DataFrame, few_shot_examples:list):

        AllPair.__init__(self, parameters)

        doc_texts = query_results['text'].tolist()
        doc_ids = query_results['docid'].tolist()
        doc_no = query_results['docno'].tolist()
        
        doc_num = [(i, d) for i, d in enumerate(doc_texts)]
        score_matrix = np.zeros((len(doc_texts), len(doc_texts)))

        self.print_flag = -1 # set it 0 for per-query verbose
        #log_predictions = []

        for index_one in range(len(doc_texts)):
            doc_one = doc_num[index_one][1]
            for index_two in range(len(doc_texts)):
                if index_one != index_two and score_matrix[index_one][index_two]==0:
                    start_time = time.time()
                    doc_two = doc_num[index_two][1]
                    
                    prompt_a, prompt_b = GeneratePrompt(self.print_flag, parameters, qid, query, doc_one, doc_two, few_shot_examples)
                    if self.print_flag == 0:
                        self.print_flag = 1
                    
                    a_scores = LLMGenerator.generate(self, parameters, prompt_a)
                    b_scores = LLMGenerator.generate(self, parameters, prompt_b)
                    
                    # for verification
                    q1_output = 'A' if a_scores['A'] > a_scores['B'] else 'B'
                    q2_output = 'A' if b_scores['A'] > b_scores['B'] else 'B'
                    
                    #log_predictions.append([qid, (doc_ids[index_one], doc_ids[index_two]), q1_output, q2_output])
                    
                    # -861 is a flag value that needs resetting after the task is over
                    # formulation as per PRP-allpair scoring
                    
                    if a_scores['A'] > a_scores['B'] and b_scores['B'] > b_scores['A']:
                        score_matrix[index_one][index_two] = 1
                        score_matrix[index_two][index_one] = -861
                    elif a_scores['A'] < a_scores['B'] and b_scores['B'] < b_scores['A']:
                        score_matrix[index_two][index_one] = 1
                        score_matrix[index_one][index_two] = -861
                    elif a_scores['A'] == a_scores['B'] or b_scores['B'] == b_scores['A']:
                        score_matrix[index_one][index_two] = 0.5
                        score_matrix[index_two][index_one] = 0.5
                    else:
                        score_matrix[index_one][index_two] = -861
                        score_matrix[index_two][index_one] = -861
                    
                    per_query_time = time.time() - start_time
                    parameters['duration'].append(per_query_time)

        # reset flag to zero for scoring
        score_matrix[score_matrix==-861] = 0
        return doc_no, doc_ids, doc_texts, np.sum(score_matrix, axis=1).tolist()