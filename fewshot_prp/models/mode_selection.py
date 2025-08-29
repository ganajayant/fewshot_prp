import random
import sys

sys.path.append("..")
import pyterrier as pt

if not pt.started():
    pt.init()


class Local:
    def remove_symbols(text):
        text = text.lower()
        # initializing bad_chars_list
        bad_chars = [";", ":", "!", "*", "/", "?", "'", '"', "-", "_", ".", "%"]
        for i in bad_chars:
            text = text.replace(i, " ")
        return str(text)

    def __new__(self, related_data):
        # picking a localized sample from fewshots
        localized = random.choice(related_data)

        # get the similar query
        related_query = localized["msmarco.query.text"]
        sel_query = Local.remove_symbols(related_query)

        # get the rel docs
        rel_doc = localized["msmarco.qrel.info"][0]["reldoc.text"]
        sel_rel_doc = Local.remove_symbols(rel_doc)

        nrel_doc = localized["msmarco.qrel.info"][0]["nreldoc.text"]
        sel_nrel_doc = Local.remove_symbols(nrel_doc)

        return sel_query, sel_rel_doc, sel_nrel_doc
