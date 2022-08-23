"""
A script contains different functions to do some calculations for metrics
"""
from scipy import spatial
from scipy import stats


def calculate_similarity_between_vectors(metric_key, vector1, vector2):
    if metric_key == 'cosine':
        return 1 - spatial.distance.cosine(vector1, vector2)
    if metric_key == 'pearson':
        return stats.pearsonr(vector1, vector2)[0]
    return 0
