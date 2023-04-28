from tqdm import tqdm
import hydra
from pathlib import Path
import subprocess
import kenlm
import time
import multiprocessing

def score_corpus(binary_model_path, corpus_path, position):
    # compute the score for the corpus
    # the corpus is expected to consist of a tokenized sentence per line
    # alternatively, run bin/query -v summary
    model = kenlm.Model(binary_model_path)

    words = 0
    score = 0
    with open(corpus_path, "r") as fp:
        lines = 0
        for line in fp:
            lines += 1
    
    with open(corpus_path, "r") as fp:
        for line in tqdm(fp, desc="lines", total=lines, position=position):
            sent = line.strip()
            # sum the probabilities for each sentence and count words
            # word count will be len(sentence) + 1 for each sentence (due to added </s> token)
            for prob, _, _ in model.full_scores(sent):
                score += prob
                words += 1
    # equivalent to log10(perplexity)
    # perplexity = 10.0**(-score/words)
    cross_entropy = (-score / words)
    return corpus_path, cross_entropy


def get_scores(cfg, binary_model_path):
    with multiprocessing.Pool(processes=cfg.processes) as pool:
        args = [(binary_model_path, comp_corpus, idx) for idx, comp_corpus in enumerate(cfg.corpora)]
        scores = pool.starmap(score_corpus, args)

    return scores


def get_training_score(corpus, scores):
    for comp_corpus, score in scores:
        if comp_corpus == corpus:
            return score

def calculate_deltas(corpus, scores, out_fp):
    training_score = get_training_score(corpus, scores)
    for comp_corpus, comp_score in scores:
        out_fp.write(f"{corpus}, {comp_corpus}, {training_score:.4f}, {comp_score:.4f}, {comp_score - training_score:.4f}\n")


@hydra.main(version_base=None, config_path="./conf", config_name=None)
def main(cfg):

    lm_command = Path(cfg.lm.lib_path).joinpath("bin/lmplz")
    binary_command = Path(cfg.lm.lib_path).joinpath("bin/build_binary")
    t00 = time.time()

    for corpus in cfg.corpora:
        print(f'### CORPUS: {corpus}')
        arap_model_path = Path(corpus).with_suffix('.arpa')
        binary_model_path = Path(corpus).with_suffix('.binary')
        t0 = time.time()
        # create lm model
        if not binary_model_path.is_file():
            with open(corpus, "r") as fp, open(arap_model_path, "w") as out_fp:
                subprocess.run(
                    [lm_command, 
                    '-o', str(cfg.lm.order),
                    '-S', cfg.lm.memory,
                    '-T', cfg.lm.tmp_folder,
                    '--discount_fallback'],
                    stdin=fp,
                    stdout=out_fp,
                    check=True,
                    text=True
                )

        t1 = time.time()
        print('#'*80)
        print(f"LM creation: {t1-t0}")
        # convert lm arpa to binary
        if not binary_model_path.is_file():
            subprocess.run(
                [binary_command, 
                arap_model_path,
                str(binary_model_path)],
                check=True,
                text=True
            )
            if not cfg.lm.keep_arpa:
                arap_model_path.unlink()

        t2 = time.time()
        print('#'*80)
        print(f"LM binary: {t2-t1}")
        # compare corpus to all other corpora and calculate deltas
        scores = get_scores(cfg, str(binary_model_path))
        with open(cfg.results_file, "a") as fp:
            calculate_deltas(corpus, scores, out_fp=fp)
        
        t3 = time.time()
        print('#'*80)
        print(f"KD scoring: {t3-t2}")
        print('#'*80)
        print(f"Iteration time: {t3-t0}")
    print('#'*80)
    print('#'*80)
    print(f"Total time: {t3-t00}")



if __name__ == '__main__':
    main()
