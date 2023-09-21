from tqdm import tqdm
import hydra
import kenlm

def score_corpus(model, corpus_path):
    # compute the score for the corpus
    # the corpus is expected to consist of a tokenized sentence per line

    words = 0
    score = 0
    with open(corpus_path, "r") as fp:
        lines = 0
        for line in fp:
            lines += 1
    
    with open(corpus_path, "r") as fp:
        for line in tqdm(fp, desc="lines", total=lines):
            sent = line.strip()
            # sum the probabilities for each sentence and count words
            # word count will be len(sentence) + 1 for each sentence (due to added </s> token)
            for prob, _, _ in model.full_scores(sent):
                score += prob
                words += 1
    # equivalent to log10(perplexity)
    # perplexity = 10.0**(-score/words)
    log_perplexity = (-score / words)
    return log_perplexity


@hydra.main(version_base=None, config_path="./conf", config_name=None)
def main(cfg):

    model = kenlm.Model(cfg.lm.model_path)

    training_score = score_corpus(model, cfg.train_corpus)
    print(f"{cfg.train_corpus}, {training_score:.4f}")

    for corpus in cfg.comp_corpora:
        comp_score = score_corpus(model, corpus_path=corpus)
        print(f"{corpus}, {comp_score:.4f}, {comp_score - training_score:.4f}")


if __name__ == '__main__':
    main()
