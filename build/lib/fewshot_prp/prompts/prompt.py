from .struct import ZeroShotPrompt, MZeroShotPrompt, ROFewShotPrompt, FewShotPrompt, MFewShotPrompt
import random
import sys
sys.path.append("..")
from models.mode_selection import Local
import configuration

class GeneratePrompt:
    def __init__(self, print_flag, parameters):
        self.print_flag = print_flag
        self.query_num = parameters['query_num']
        self.kshot = parameters['kshot']
        self.mode = parameters['mode']
        self.random_seed = parameters['seed']
        self.query_num = parameters['query_num']
        self.reranker = parameters['reranker']

    def __new__(self,
                print_flag,
                parameters,
                qid,
                query:str,
                doc_one:str,
                doc_two:str,
                few_shot_examples:list = []) -> tuple:

        GeneratePrompt.__init__(self, print_flag, parameters)

        if len(few_shot_examples)>0:
            #do few-shot
            related_data = []

            # 2. map the query, and fetch corresoponding relataed data
            for ele in few_shot_examples:
                if int(ele['trecdl.query.id']) == int(qid):
                    related_data = ele.get('fewshots', [])
            
            # 3. Take a sub-set of fewshot examples
            related_data = related_data[0:configuration.max_fewshot]
            
            # 4. Few shot examples if there is data else zero shot

            if len(related_data) > 0 or self.mode=='TOR':
                if self.print_flag==0:
                    print('***Few Shot for ', query)
                    self.print_flag = 1
                examples = []
                current_top = 0
                for k in range(self.kshot):
                    example = {}
                    # 2. selecting example data based on mode
                    related_query, rel_doc, nrel_doc = Local(related_data)
                    # store all the examples in the dict
                    example = {f'rel_query_{k}': related_query, f'rel_doc_{k}': rel_doc, f'nrel_doc_{k}': nrel_doc}
                    examples.append(example)

                if self.mode=='SEM' or self.mode=='LEX' or self.mode=='STC':
                    if self.reranker=='zephyr':
                        prompt_a = FewShotPrompt(query, doc_one, doc_two, examples)
                        prompt_b = FewShotPrompt(query, doc_two, doc_one, examples)
                    else:
                        prompt_a = MFewShotPrompt(query, doc_one, doc_two, examples)
                        prompt_b = MFewShotPrompt(query, doc_two, doc_one, examples)
                elif self.mode=='SEMRO' or self.mode=='LEXRO' or self.mode=='STCRO':
                    prompt_a = ROFewShotPrompt(query, doc_one, doc_two, examples)
                    prompt_b = ROFewShotPrompt(query, doc_two, doc_one, examples)

                return (prompt_a, prompt_b)

            # Zero-Shot
            else:
                if self.print_flag==0:
                    print('***Zero Shot for ', query)
                    self.print_flag = 1
                if self.reranker=='zephyr':
                    prompt_a = ZeroShotPrompt(query, doc_one, doc_two)
                    prompt_b = ZeroShotPrompt(query, doc_two, doc_one)
                else:
                    prompt_a = MZeroShotPrompt(query, doc_one, doc_two)
                    prompt_b = MZeroShotPrompt(query, doc_two, doc_one)
                return (prompt_a, prompt_b)

        else:
            # Zero-Shot
            if self.print_flag==0:
                    print('***Zero Shot for ', query)
                    self.print_flag = 1
            if self.reranker=='zephyr':
                prompt_a = ZeroShotPrompt(query, doc_one, doc_two)
                prompt_b = ZeroShotPrompt(query, doc_two, doc_one)
            else:
                prompt_a = MZeroShotPrompt(query, doc_one, doc_two)
                prompt_b = MZeroShotPrompt(query, doc_two, doc_one)
            return (prompt_a, prompt_b)
