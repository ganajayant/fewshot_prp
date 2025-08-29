import math

import pandas as pd
import pyterrier as pt
from pyterrier.measures import *
from pyterrier.model import add_ranks

if not pt.started():
    pt.init()
import sys

sys.path.append("..")
import configuration


class Scoring:
    def fusion_sum(metrics, res_ori, res_zer, qrels, w1: float = 1.0, w2: float = 1.0):
        pd.options.mode.chained_assignment = None  # default='warn'

        # copy to avoid changing original variables
        res_zer = res_zer.copy()
        res_add = res_ori.copy()

        if "docno" not in res_zer.columns:
            res_zer["docno"] = res_zer["docid"]

        res_add["score"] = res_add["score"].mul(w1)
        res_zer["score"] = res_zer["score"].mul(w2)

        res_add["fallback_score"] = res_add["score"] + res_zer["score"]
        res_add = res_add.drop("score", axis=1)
        res_add["score"] = res_add["fallback_score"]
        res_add = res_add.drop("fallback_score", axis=1)
        res_add = res_add.drop("rank", axis=1)
        res_add.head()
        eval = pt.Evaluate(res_add, qrels, metrics)
        return eval

    def direct_fusion_sum(
        metrics, res_ori, res_zer, qrels, w1: float = 1.0, w2: float = 1.0
    ):
        pd.options.mode.chained_assignment = None  # default='warn'

        # copy to avoid changing original variables
        res_zer = res_zer.copy()
        res_add = res_ori.copy()

        if "docno" not in res_zer.columns:
            res_zer["docno"] = res_zer["docid"]

        res_zer = add_ranks(res_zer)
        res_add = add_ranks(res_add)

        res_zer["qid"] = res_zer["qid"].astype(int)

        if res_add["docno"].dtype != res_zer["docno"].dtype:
            res_zer["docno"] = res_zer["docno"].astype(res_add["docno"].dtype)

        # print(res_zer['docno'].dtype)
        # res_add['qid']=res_add['qid'].astype(int)

        # res_zer['docno']=res_zer['docno'].astype(object)
        # res_add['docno']=res_add['docno'].astype(object)

        # print(res_zer.dtypes)
        # print(res_add.dtypes)

        res = res_zer.merge(res_add, on=["qid", "docno"])
        res["score"] = w1 * res["score_x"] + w2 * res["score_y"]
        try:
            eval = pt.Evaluate(res, qrels, metrics)
        except:
            eval = "Null"
        return eval

    def fallback(metrics, res_ori, res_zer, qrels):
        pd.options.mode.chained_assignment = None  # default='warn'

        if "docno" not in res_zer.columns:
            res_zer["docno"] = res_zer["docid"]

        if "docno" not in res_ori.columns:
            res_ori["docno"] = res_ori["docid"]

        if "query_id" in res_zer.columns:
            res_zer["qid"] = res_zer["query_id"]

        if "query_id" in res_ori.columns:
            res_ori["qid"] = res_ori["query_id"]

        res_fall = res_zer
        res_fall_total = pd.DataFrame()
        res_fall_total["updated_score"] = 0
        res_fall_total["bm25_score"] = 0

        unique_qids = res_fall["qid"].unique()

        for qid in unique_qids:
            current_score = 1
            ranking_qid = res_fall.loc[res_fall["qid"] == qid]
            unique_scores = ranking_qid["score"].unique()
            unique_scores = sorted(unique_scores, reverse=True)[:]

            for score in unique_scores:
                ranking_qid_sub = ranking_qid.loc[ranking_qid["score"] == score]
                docids = list(ranking_qid_sub["docid"])
                original = res_ori.loc[res_ori["qid"] == qid]
                original = original.loc[original["docid"].isin(docids)]
                index = 0
                for docid in ranking_qid_sub.docid:
                    original_score = (
                        original["score"].loc[original["docid"] == docid].values[0]
                    )

                    rowIndex = ranking_qid_sub.index[index]
                    index += 1

                    ranking_qid_sub.at[rowIndex, "bm25_score"] = original_score
                    ranking_qid_sub.at[rowIndex, "updated_score"] = current_score
                    current_score += 1

                frames = [res_fall_total, ranking_qid_sub]
                res_fall_total = pd.concat(frames)

        res_fall_total = res_fall_total.drop("bm25_score", axis=1)
        res_fall_total = res_fall_total.drop("score", axis=1)
        res_fall_total["score"] = (
            max(res_fall_total["updated_score"]) - res_fall_total["updated_score"]
        )
        res_fall_total = res_fall_total.drop("updated_score", axis=1)
        res_fall_total = res_fall_total.drop("Unnamed: 0", axis=1)
        res_fall_total["qid"] = res_fall_total["qid"].astype(int)

        eval = pt.Evaluate(res_fall_total, qrels, metrics)
        return eval

    def reciprocal_rank_topics(metrics, topics, ds, top, res_zer, qrels):
        pd.options.mode.chained_assignment = None  # default='warn'

        # add ranks
        new_res_zer = res_zer
        # new_res_zer['docno'] = new_res_zer['docid']
        new_res_zer = add_ranks(new_res_zer)
        new_res_zer = new_res_zer.drop("Unnamed: 0", axis=1)
        new_res_zer[["qid", "docid"]] = new_res_zer[["qid", "docid"]].apply(
            pd.to_numeric
        )

        # scoring rr
        index = pt.IndexFactory.of(configuration.datasets[ds]["index"], memory=True)
        bm25 = (
            pt.BatchRetrieve(
                index,
                controls={
                    "wmodel": "BM25",
                    "bm25.b": configuration.bm25_b,
                    "bm25.k_1": configuration.bm25_k1,
                },
                metadata=["docno", "text"],
            )
            % top
        )
        res_ori = bm25.transform(topics)
        res_ori[["qid", "docid"]] = res_ori[["qid", "docid"]].apply(pd.to_numeric)
        res_rr = res_ori
        res_rr = res_rr.drop("score", axis=1)
        res_rr["score"] = 0

        res_rr = res_rr.reset_index()  # make sure indexes pair with number of rows

        for index, row in res_rr.iterrows():
            qid = row["qid"]
            docid = row["docid"]
            rank = row["rank"]
            print("------------------------------")
            print("qid ->", type(qid))
            print("resqid ->", new_res_zer["qid"].dtypes)
            print("did ->", type(docid))
            print("resdid ->", new_res_zer["docid"].dtypes)
            print("------------------------------")
            rank_llm = (
                new_res_zer["rank"]
                .loc[(new_res_zer["qid"] == qid) & (new_res_zer["docid"] == docid)]
                .values[0]
            )
            rr_score = math.log(1 + (1 / (rank + 1))) + math.log(
                1 + (1 / (rank_llm + 1))
            )
            res_rr.loc[index, "score"] = rr_score

        eval = pt.Evaluate(res_rr, qrels, metrics)
        return eval

    def reciprocal_rank(metrics, res_original, res_zer, qrels, ds):
        pd.options.mode.chained_assignment = None  # default='warn'

        # copy to avoid changing original variables
        new_res_zer = res_zer.copy()
        res_ori = res_original.copy()
        res_ori = add_ranks(res_ori)

        if "docno" not in new_res_zer.columns:
            new_res_zer["docno"] = new_res_zer["docid"]
        new_res_zer = add_ranks(new_res_zer)

        if "Unnamed: 0" in new_res_zer.columns:
            new_res_zer = new_res_zer.drop("Unnamed: 0", axis=1)

        # new_res_zer[['qid', 'docno']] = new_res_zer[['qid', 'docno']].apply(pd.to_numeric)

        # scoring rr
        # res_ori[['qid', 'docid']] = res_ori[['qid', 'docid']].apply(pd.to_numeric)
        res_rr = res_ori
        res_rr = res_rr.drop("score", axis=1)
        res_rr["score"] = 0

        res_rr = res_rr.reset_index()  # make sure indexes pair with number of rows

        for index, row in res_rr.iterrows():
            qid = row["qid"]
            docid = row["docno"]
            rank = row["rank"]
            current_res = new_res_zer.loc[new_res_zer["qid"].astype(str) == str(qid)]
            try:
                rank_llm = (
                    current_res["rank"]
                    .loc[new_res_zer["docno"].astype(str) == str(docid)]
                    .values[0]
                )
            except:
                pass
            rr_score = math.log(1 + (1 / (rank + 1))) + math.log(
                1 + (1 / (rank_llm + 1))
            )
            res_rr.loc[index, "score"] = rr_score

        eval = pt.Evaluate(res_rr, qrels, metrics)
        return eval
