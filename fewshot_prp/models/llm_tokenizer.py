import sys

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

sys.path.append("..")
import configuration


class LoadLLM_Model:
    def __init__(self, reranker):
        self.model_name = configuration.rerankers[reranker]["model_name"]

    def __new__(self, reranker):
        model_name = configuration.rerankers[reranker]["model_name"]
        # setup the llm model
        cache_path = configuration.cache_path
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # if self.model_name == "HuggingFaceH4/zephyr-7b-beta":
        if reranker == "zephyr":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            model.eval()
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                model_name, cache_dir=cache_path, device_map="auto"
            )
            model.eval()
        return model


class LoadLLM_Tokenizer:
    def __init__(self, reranker):
        self.tok_name = configuration.rerankers[reranker]["tokenizer"]

    def __new__(self, reranker):
        tok_name = configuration.rerankers[reranker]["tokenizer"]
        if reranker == "zephyr":
            tokenizer = AutoTokenizer.from_pretrained(tok_name)
        else:
            tokenizer = T5Tokenizer.from_pretrained(tok_name)
        return tokenizer
