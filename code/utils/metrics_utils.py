"""
A script contains different functions to do some calculations for metrics
"""
from scipy import spatial
from scipy import stats
import math


def calculate_similarity_between_vectors(metric_key, vector1, vector2):
    if metric_key == 'cosine':
        return 1 - spatial.distance.cosine(vector1, vector2)
    if metric_key == 'pearson':
        return stats.pearsonr(vector1, vector2)[0]
    return 0


def get_token_counts(tokens: list):
    # get the token count of each token in a list of tokens
    token_counts = {}
    for token in tokens:
        if token not in token_counts:
            token_counts[token] = 0
        token_counts[token] += 1
    return token_counts


def kli(p_t_d, p_t_c):
    # p_t_d = P(t|D)
    # p_t_c = P(t|C)
    # KLI(t) = P(t|D) * log (P(t|D) / P(t|C))
    return p_t_d * math.log10(p_t_d / p_t_c)


def kli_divergence(terms: set, document_counts: dict, collection_prob_dict: dict):
    document_total_terms = sum(document_counts.values())
    document_prob = {term: count/document_total_terms for term, count in document_counts.items()}
    collection_total_terms = collection_prob_dict["total_terms"]
    collection_prob = collection_prob_dict["probabilities"]
    term_probs = []
    for term in terms:
        if term in collection_prob:
            p_t_c = collection_prob[term]
        else:
            p_t_c = 1 / collection_total_terms
        p_t_d = document_prob[term]
        term_probs.append((term, kli(p_t_d, p_t_c)))
    return sorted(term_probs, key=lambda x: x[1], reverse=True)

