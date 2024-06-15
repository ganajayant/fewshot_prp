zero_shot = """
<|system|>
You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query</s> 
<|user|>
which of the following two passages is more relevant to the query "{query}"?
Passage 1 : "{doc_one}"
Passage 2 : "{doc_two}"
Output: </s> 
<|assistant|>
"""

few_shot = """
<|system|>
You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query</s> 
<|user|>
{examples}
which of the following two passages is more relevant to the query "{query}"?
Passage 1 : "{doc_one}"
Passage 2 : "{doc_two}"
Output: </s> 
<|assistant|>
"""

mini_zero_shot= """
Which of the following two passages is more relevant to {query}?
1: {doc_one}
2: {doc_two}
"""

mini_few_shot = """
{examples}

Which of the following two passages is more relevant to {query}?
1: {doc_one}
2: {doc_two}
"""