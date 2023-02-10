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


def kli_divergence(terms: set, document_term_counts: dict, collection_prob_dict: dict):
    document_total_terms = sum(document_term_counts.values())
    document_prob = {term: count/document_total_terms for term, count in document_term_counts.items()}
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


def idf_scores(terms: set, idf_dict: dict):
    term_idf = idf_dict["idf"]
    term_scores = []
    for term in terms:
        if term in term_idf:
            term_scores.append((term, term_idf[term]))
        else:
            term_scores.append((term, idf_dict["idf_unknown"]))
    return sorted(term_scores, key=lambda x: x[1], reverse=True)


class PLMTerm:
    def __init__(self, term: str, tf, p_t_d, p_t_c):
        self.term = term
        self.tf = tf
        self.p_t_d = p_t_d
        self.p_t_c = p_t_c
        self.e = 0
        self.converged = False

    def e_step(self, smooth_factor):
        self.e = self.tf * ((smooth_factor * self.p_t_d)
            / (((1 - smooth_factor) * self.p_t_c) + (smooth_factor * self.p_t_d)))

    def m_step(self, terms, inverse_d):
        sum_esteps = sum(term.e for term in terms if term.term != self.term)

        new_p_t_d = self.e / sum_esteps

        # stop if the change is less than 5% or the new probability is less than 1 in |D|
        if (new_p_t_d < inverse_d) or ((abs(new_p_t_d - self.p_t_d) / self.p_t_d) < 0.05):
            self.converged = True

        self.p_t_d = new_p_t_d


# take list of terms and return list of terms sorted by score ..
def plm(terms: set, document_term_counts: dict, collection_prob_dict: dict, smooth_factor: float, max_steps: int):
    document_total_terms = sum(document_term_counts.values())
    document_prob = {term: count/document_total_terms for term, count in document_term_counts.items()}
    collection_total_terms = collection_prob_dict["total_terms"]
    collection_prob = collection_prob_dict["probabilities"]
    term_probs = []
    plm_terms = []
    for term in terms:
        tf = document_term_counts[term]
        if term in collection_prob:
            p_t_c = collection_prob[term]
        else:
            p_t_c = 1 / collection_total_terms
        p_t_d = document_prob[term]
        plm_terms.append(PLMTerm(term=term, tf=tf, p_t_d=p_t_d, p_t_c=p_t_c))
    inverse_d = 1 / document_total_terms
    for i in range(max_steps):
        
        for term in plm_terms:
            if not term.converged:
                term.e_step(smooth_factor)

        for term in plm_terms:
            if not term.converged:
                term.m_step(plm_terms, inverse_d)

        if all([t.converged for t in plm_terms]):
            break

    term_probs = []
    for term in plm_terms:
        if term.p_t_d > 0.0001:
            term_probs.append((term.term, term.p_t_d))

    return sorted(term_probs, key=lambda x: x[1], reverse=True)