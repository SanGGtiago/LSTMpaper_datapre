import numpy as np

from sklearn.metrics import ndcg_score
# we have groud-truth relevance of some answers to a query:
true_relevance = np.asarray([[10, 3, 0, 1, 5, 8, 9, 7]])
# we predict some scores (relevance) for the answers
scores = np.asarray([[.1, .2, .3, 4, 70, 8, 9, 6]])
ndcg_score(true_relevance, scores)
print(ndcg_score(true_relevance, scores))
print(ndcg_score(true_relevance, scores, 5))