import os
import re

import pandas as pd
import pyterrier as pt
from evaluation.scoring import Scoring
from pyterrier.measures import *

if not pt.started():
    pt.init()
import sys

sys.path.append("..")
import configuration


class ModelEvaluation:
    def evaluate(
        r_model,
        metrics: list = [AP(rel=2) @ 100, NDCG(cutoff=10)],
        fusion_sum: bool = False,
        w1: float = 0.7,
        w2: float = 0.3,
        fallback: bool = False,
        reciprocal_rank: bool = False,
    ):
        scores_df_sorted = pd.DataFrame()
        sys.path.append("..")
        if r_model == "baselines":
            files = [
                f
                for f in os.listdir(f"./datasets/phase_one_retrieval/")
                if re.match(r"ranked*", f)
            ]
        else:
            files = [
                f for f in os.listdir(f"./scores/") if re.match(r"reranking_scores*", f)
            ]

        file_found = False
        scores = []

        for file in files:
            # try:
            file_name = file
            file_info = file.split("_")

            try:
                ds = file_info[2]
            except:
                ds = ""

            try:
                top_k = file_info[3][1:]
            except:
                top_k = ""

            try:
                reranker = file_info[4]
            except:
                reranker = ""

            try:
                mode = file_info[5]
            except:
                mode = ""

            try:
                kshot = file_info[7][0]
            except:
                kshot = ""

            # evaluate
            if r_model == "baselines":
                res = pd.read_csv(f"./datasets/phase_one_retrieval/{file_name}")
                model_name = file_name[7:]
            else:
                res_raw = pd.read_csv(f"./scores/{file_name}")
                res = res_raw
                res = res.loc[:, ~res.columns.str.contains("^Unnamed")]
                model_name = file_name[17:]

            if "docno" not in res.columns:
                res["docno"] = res["docid"]

            if "query_id" in res.columns:
                res["qid"] = res["query_id"]

            dataset = pt.datasets.get_dataset(configuration.datasets[ds]["name"])
            qrels = dataset.get_qrels()
            topics = dataset.get_topics(configuration.datasets[ds]["topics"])

            # files found
            file_found = True

            # evaluation scores
            eval = pt.Evaluate(res, qrels, metrics=metrics)
            model_name = model_name[: len(model_name) - 4]
            eval["Model"] = model_name
            scores.append(eval)

            # Fusion mode
            if fusion_sum == True and r_model != "baselines":
                pd.options.mode.chained_assignment = None
                ori_name = f"ranked_bm25_{ds}_t{top_k}.csv"
                ori = pd.read_csv(f"./datasets/phase_one_retrieval/{ori_name}")
                # eval_fusion_sum = Scoring.fusion_sum(metrics, ori, res, qrels, w1, w2)
                eval_dfusion_sum = Scoring.direct_fusion_sum(
                    metrics, ori, res, qrels, w1, w2
                )
                # eval['Fusion Sum'] = eval_fusion_sum
                eval["DFusion Sum"] = eval_dfusion_sum

            # Fallback mode
            if fallback == True and r_model != "baselines":
                if (
                    ds == "covid"
                    or ds == "touche"
                    or ds == "toucheV2"
                    or ds == "scifact"
                ):
                    pd.options.mode.chained_assignment = None
                    ori_name = f"ranked_bm25_{ds}_t{top_k}.csv"
                    ori = pd.read_csv(f"./datasets/phase_one_retrieval/{ori_name}")
                    eval_fallback = Scoring.fallback(metrics, ori, res_raw, qrels)
                    eval["Fallback"] = eval_fallback

            if reciprocal_rank == True and r_model != "baselines":
                pd.options.mode.chained_assignment = None
                ori_name = f"ranked_bm25_{ds}_t{top_k}.csv"
                ori = pd.read_csv(f"./datasets/phase_one_retrieval/{ori_name}")
                eval_rr = Scoring.reciprocal_rank(metrics, ori, res_raw, qrels, ds)
                eval["RR-BM25-LLM"] = eval_rr

        if file_found:
            scores_df = pd.DataFrame(scores)
            cols = scores_df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            scores_df = scores_df[cols]

            output = scores_df
            scores_df_sorted = output.sort_values(["Model"])
            scores_df_sorted.reset_index(inplace=True)
            scores_df_sorted = scores_df_sorted.drop("index", axis=1)
        else:
            print("No res files found. Please check the location.")

        return scores_df_sorted
