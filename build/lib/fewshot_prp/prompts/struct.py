from typing import List
from .text import zero_shot, few_shot, mini_zero_shot, mini_few_shot

class ZeroShotPrompt:
    def __new__(self, query : str, doc1 : str, doc2 : str) -> str:
        return zero_shot.format(query=query, doc_one=doc1, doc_two=doc2)

class ROFewShotPrompt:
    def __new__(self, query : str, doc1 : str, doc2 : str, examples : list) -> str:
        final_examples = ''
        counter = 0
        for example in examples:
            example_format = '''
            A passage relevant to the query "'''+str(example['rel_query_'+str(counter)])+'''" is given.
            Relevant Passage: "'''+str(example['rel_doc_'+str(counter)])+'''"
            '''
            final_examples = final_examples + example_format
           
        return few_shot.format(query=query, doc_one=doc1, doc_two=doc2, examples = final_examples)

class MROFewShotPrompt:
    def __new__(self, query : str, doc1 : str, doc2 : str, examples : list) -> str:
        final_examples = ''
        counter = 0
        for example in examples:
            example_format = '''
            A passage relevant to the query "'''+str(example['rel_query_'+str(counter)])+'''" is given.
            Relevant Passage: "'''+str(example['rel_doc_'+str(counter)])+'''"
            '''
            final_examples = final_examples + example_format
           
        return mini_few_shot.format(query=query, doc_one=doc1, doc_two=doc2, examples = final_examples)

class FewShotPrompt:
    def __new__(self, query : str, doc1 : str, doc2 : str, examples : list) -> str:
        
        counter = 0
        final_examples = ''
        for example in examples:
            if counter%2==0:
                example_format = '''
                which of the following two passages is more relevant to the query "'''+str(example['rel_query_'+str(counter)])+'''"?
                Passage 1 : "'''+str(example['rel_doc_'+str(counter)])+'''"
                Passage 2 : "'''+str(example['nrel_doc_'+str(counter)])+'''"
                Output: 1

                '''
            else:
                example_format = '''
                which of the following two passages is more relevant to the query "'''+str(example['rel_query_'+str(counter)])+'''"?
                Passage 1 : "'''+str(example['nrel_doc_'+str(counter)])+'''"
                Passage 2 : "'''+str(example['rel_doc_'+str(counter)])+'''"
                Output: 2

                '''
            final_examples = final_examples + example_format
            counter+=1
            
        return few_shot.format(query=query, doc_one=doc1, doc_two=doc2, examples = final_examples)

class MZeroShotPrompt:
    def __new__(self, query : str, doc1 : str, doc2 : str) -> str:
        return mini_zero_shot.format(query=query, doc_one=doc1, doc_two=doc2)
        
class MFewShotPrompt:
    def __new__(self, query : str, doc1 : str, doc2 : str, examples : list) -> str:
        
        counter = 0
        final_examples = ''
        for example in examples:
            if counter%2==0:
                example_format = '''
                Which of the following two passages is more relevant to "'''+str(example['rel_query_'+str(counter)])+'''"?
                1: "'''+str(example['rel_doc_'+str(counter)])+'''"
                2: "'''+str(example['nrel_doc_'+str(counter)])+'''"
                Output: 1

                '''
            else:
                example_format = '''
                Which of the following two passages is more relevant to "'''+str(example['rel_query_'+str(counter)])+'''"?
                1: "'''+str(example['nrel_doc_'+str(counter)])+'''"
                2: "'''+str(example['rel_doc_'+str(counter)])+'''"
                Output: 2

                '''
            final_examples = final_examples + example_format
            counter+=1
            
        return mini_few_shot.format(query=query, doc_one=doc1, doc_two=doc2, examples = final_examples)