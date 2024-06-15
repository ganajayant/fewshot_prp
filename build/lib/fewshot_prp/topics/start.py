import os
import sys
import pandas as pd
import pyterrier as pt
if not pt.started(): pt.init()
sys.path.append("..")
import configuration

class PhaseOneRetrieval():
    
    # load the index
    def load_index(index_path):
        index = pt.IndexFactory.of(index_path, memory=True)
        return index

    # load the topics
    def load_topics(ds):
        dataset = pt.datasets.get_dataset(configuration.datasets[ds]['name'])
        queries = dataset.get_topics(configuration.datasets[ds]['topics'])
        return queries
   
    def remove_symbols(text):
        # initializing bad_chars_list
        bad_chars = [';',':','!',"*","/","?","'",'"',"-","_",".","%"]
        for i in bad_chars:
            text = text.replace(i, ' ')
        return str(text)
    
    def clean(queries):
        for i in range(len(queries)):
            text = queries['query'][i]
            text = text.lower()
            text = PhaseOneRetrieval.remove_symbols(text)
            queries['query'][i] = text
        return queries

    # configure the retrieval model
    def retrieval(r_model, index, queries, top_cut):
        retr = pt.BatchRetrieve(index, controls={"wmodel": r_model.upper(), "bm25.b" : configuration.bm25_b, "bm25.k_1": configuration.bm25_k1}, metadata=["docno", "text"])%top_cut
        
        res=retr.transform(queries) # this has the ranked retrieved documents
        return res

    # save the ranked results
    def save_res(self, res):
        sys.path.append("..")
        res.to_csv(f'./datasets/phase_one_retrieval/ranked_{self.r_model.lower()}_{self.ds}_t{self.top_cut}.csv', index=False)

    def __init__(self, r_model: str, ds:str, top_cut: int):
        self.index_path = configuration.datasets[ds]['index']
        #self.collection_path = configuration.data_path+f'msmarco/collection.tsv'
        self.r_model = r_model
        self.ds = ds
        self.top_cut = top_cut
        self.res = pd.DataFrame()
        
    def __new__(self, r_model: str, ds:str, top_cut: int):
        
        self.__init__(self, r_model, ds, top_cut)

        print('>> Index loading. \n')
        index = self.load_index(self.index_path)

        print('>> Index loading successful\n')
        #print(index.getCollectionStatistics().toString())

        print('>> Loading Topics\n')
        queries = self.load_topics(ds)
        queries = self.clean(queries)

        print('>> Retrieval in progress using '+self.r_model+'\n')
        res = self.retrieval(self.r_model, index, queries, self.top_cut)

        self.save_res(self, res)
        print('>> Passages saved.')