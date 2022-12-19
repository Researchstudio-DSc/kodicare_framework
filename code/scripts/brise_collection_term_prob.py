import os
import hydra
from hydra.utils import instantiate
import json


@hydra.main(version_base=None, config_path="../../conf", config_name=None)
def main(cfg):
    reader = instantiate(cfg.indexing.collection_reader, data_dir=cfg.config.data_dir)
    tokenizer = instantiate(cfg.evaluation.tokenizer)

    ## count terms
    term_counts = {}
    total_terms = 0
    for batch in reader.iterate():
        passage_text_batch = [document_obj["passage_text"] for document_id, document_obj in batch]
        passage_text_batch = tokenizer.batch_tokenize(passage_text_batch)
        for doc in passage_text_batch:
            for token in doc:
                if token not in term_counts:
                    term_counts[token] = 0
                term_counts[token] += 1
                total_terms += 1
    ## calulate probabilities
    term_probs = {term: count/total_terms for term, count in term_counts.items()}
    with open(os.path.join(cfg.config.out_dir, "term_p.json"), "w") as fp:
        d = {
            "total_terms": total_terms,
            "probabilities": term_probs
        }
        json.dump(d, fp)


if __name__ == '__main__':
    main()