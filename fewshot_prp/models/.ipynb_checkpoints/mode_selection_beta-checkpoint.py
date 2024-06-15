import random
import sys
sys.path.append("..")
import configuration
import pyterrier as pt
if not pt.started(): pt.init()
import pandas as pd

class TOR:
    data_path = configuration.data_path
    index_path = configuration.index_path
    collection = pd.read_csv(data_path + 'msmarco/collection.tsv', sep='\t', header=None) 
    index = pt.IndexFactory.of(index_path)
    BM25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.b" : 0.75, "bm25.k_1": 1.2})
    def_nrel_id = collection[0].sample(random_state=42).values[0]

    def remove_symbols(text):
        text = text.lower()
        # initializing bad_chars_list
        bad_chars = [';',':','!',"*","/","?","'",'"',"-","_",".","%"]
        for i in bad_chars:
            text = text.replace(i, ' ')
        return str(text)
        
    def retrieve_kth_bm25(query:str , kth:int):
        query = remove_symbols(query)
        
        try:
            res=BM25.search(query)
            nrel_id = res['docid'][kth]
        except:
            print('Exception for query: ', query)
            nrel_id = def_nrel_id
        return nrel_id
    
    def __init__(self, seed : int = 42):
        self.seed = seed
        
    def __new__(self, collection, train_queries, train_qrels, seed : int = 42):

        # selecting query and rel-doc
        #random.seed(seed)
        sel_qid = random.choice(train_qrels[0])
        sel_query = train_queries[1].loc[train_queries[0]==sel_qid]
        sel_query = sel_query.values[0]
        sel_rel_docid = train_qrels[2].loc[train_qrels[0]==sel_qid]
        sel_rel_docid = sel_rel_docid.values[0]        
        sel_rel_doc = collection[1].loc[collection[0]==sel_rel_docid]
        sel_rel_doc = sel_rel_doc.values[0]

        # selecting nrel-doc
        random.seed(seed+5)
        sel_nrel_doc = retrieve_kth_bm25(sel_query,50)
        
        return sel_query, sel_rel_doc, sel_nrel_doc