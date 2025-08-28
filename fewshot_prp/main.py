import os
import re
import sys

import configuration
from models.llm_tokenizer import LoadLLM_Model, LoadLLM_Tokenizer
from reranker import RelevancyScorerLLM
from topics.start import PhaseOneRetrieval


def check_file(reranker, ds, top_k, mode, kshot, random_seed):
    print("this is the fodler ", os.getcwd())
    res_files = [
        f for f in os.listdir("./scores/") if re.match(r"reranking_scores*", f)
    ]
    current_file = f"reranking_scores_{ds}_t{top_k}_{reranker}_{mode}{configuration.max_fewshot}_R{random_seed}_{kshot}-shot.csv"
    if current_file in res_files:
        print(
            f"SKIPPING EXPERIMENT : {ds}_t{top_k}_{reranker}_{mode}{configuration.max_fewshot}_R{random_seed}_{kshot}-shot"
        )
        return True
    else:
        print(
            f"RUNNING EXPERIMENT : {ds}_t{top_k}_{reranker}_{mode}{configuration.max_fewshot}_R{random_seed}_{kshot}-shot"
        )
        return False


def check_res(r_model, ds, top_k):
    res_files = [
        f
        for f in os.listdir("./datasets/phase_one_retrieval/")
        if re.match(r"ranked_*", f)
    ]

    current_file = f"ranked_{r_model}_{ds}_t{top_k}.csv"

    if current_file not in res_files:
        PhaseOneRetrieval(r_model, ds, top_k)


def main():
    """Main entry point for the fewshot_prp command line tool."""
    # Parse command line arguments in key=value format
    kwargs = dict(arg.split("=") for arg in sys.argv[1:])

    # Call the actual main logic
    return __main__(**kwargs)


def __main__(**kwargs):
    # rerankers, top_ks, datasets, kshots, random_seed, modes, retrievers, parameters_for_sws, eval

    for k, v in kwargs.items():
        print("keyword argument: {} = {}".format(k, v))
        if k == "reranker":
            rerankers = v.split(",")
        if k == "top_k":
            top_ks = v.split(",")
            top_ks = [int(i) for i in top_ks]
        if k == "datasets":
            dss = v.split(",")
        if k == "kshots":
            kshots = v.split(",")
            kshots = [int(i) for i in kshots]
        if k == "seed":
            random_seeds = v.split(",")
        if k == "modes":
            modes = v.split(",")
        if k == "eval":
            if v == "True":
                eval = True
            else:
                eval = False

    for reranker in rerankers:
        model = LoadLLM_Model(reranker)
        tokenizer = LoadLLM_Tokenizer(reranker)
        r_model = "bm25"
        for ds in dss:
            for top_k in top_ks:
                for kshot in kshots:
                    if kshot == 0:
                        # zero-shot
                        mode = "ZER"
                        # Check if the same experiments are already present or not
                        random_seed = 42
                        if not check_file(
                            reranker, ds, top_k, mode, kshot, random_seed
                        ):
                            # Run the RelevancyScorerLLM class with all the parameters
                            check_res(r_model, ds, top_k)
                            RelevancyScorerLLM(
                                model,
                                tokenizer,
                                reranker,
                                top_k,
                                ds,
                                kshot,
                                random_seed,
                                mode,
                                eval,
                            )
                    else:
                        # few-shot
                        for random_seed in random_seeds:
                            for mode in modes:
                                # Check if the same experiments are already present or not
                                if not check_file(
                                    reranker, ds, top_k, mode, kshot, random_seed
                                ):
                                    # Run the RelevancyScorerLLM class with all the parameters
                                    check_res(r_model, ds, top_k)
                                    RelevancyScorerLLM(
                                        model,
                                        tokenizer,
                                        reranker,
                                        top_k,
                                        ds,
                                        kshot,
                                        random_seed,
                                        mode,
                                        eval,
                                    )


if __name__ == "__main__":
    main()
