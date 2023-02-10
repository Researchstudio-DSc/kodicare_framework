import os
import hydra
from hydra.utils import instantiate
import json
import math


@hydra.main(version_base=None, config_path="../../conf", config_name=None)
def main(cfg):
    reader = instantiate(cfg.indexing.collection_reader, data_dir=cfg.config.data_dir)
    tokenizer = instantiate(cfg.evaluation.tokenizer)

    ## count terms
    term_counts = {}
    total_terms = 0
    term_df = {}
    doc_count = 0
    for batch in reader.iterate():
        passage_text_batch = [document_obj["passage_text"] for document_id, document_obj in batch]
        passage_text_batch = tokenizer.batch_tokenize(passage_text_batch)
        for doc in passage_text_batch:
            doc_terms = set()
            for token in doc:
                doc_terms.add(token)
                if token not in term_counts:
                    term_counts[token] = 0
                term_counts[token] += 1
                total_terms += 1
            # idf
            for term in doc_terms:
                if term not in term_df:
                    term_df[term] = 0
                term_df[term] += 1
            doc_count += 1
    ## calulate probabilities
    term_probs = {term: count/total_terms for term, count in term_counts.items()}
    ## calculate idf
    term_idf = {term: math.log10(doc_count / df) for term, df in term_df.items()}
    idf_unknonw = math.log10(doc_count / 1)
    with open(os.path.join(cfg.config.out_dir, "term_p.json"), "w") as fp:
        d = {
            "total_terms": total_terms,
            "probabilities": term_probs,
            "idf": term_idf,
            "idf_unknown": idf_unknonw
        }
        json.dump(d, fp)


if __name__ == '__main__':
    main()