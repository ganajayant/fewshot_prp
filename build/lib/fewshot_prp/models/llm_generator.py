import torch
from models.llm_tokenizer import LoadLLM_Model, LoadLLM_Tokenizer
import warnings
warnings.filterwarnings("ignore")

class LLMGenerator:
    def __init__(self, parameters):
        self.model = parameters['model']
        self.tokenizer = parameters['tokenizer']
        self.A = parameters['A']
        self.B = parameters['B']
        self.device = parameters['device']

    def generate(self, parameters, prompt : str):

        LLMGenerator.__init__(self, parameters)

        score_a = []
        score_b = []

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(**inputs, do_sample=False,temperature=0.0, top_p=None, top_k=2, return_dict_in_generate=True, output_scores=True, max_new_tokens=1, pad_token_id=2)
        
        score_stack = torch.stack(outputs.scores, dim=1)

        score_a = score_stack[0][0][self.A].item() 
        score_b = score_stack[0][0][self.B].item() 
        # same as above :: result = score_stack[:, 0, (self.A, self.B)]       
        
        score_a_b = {'A': round(score_a,2), 'B': round(score_b,2)}

        return score_a_b