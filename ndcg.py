import numpy as np

from sklearn.metrics import ndcg_score
# we have groud-truth relevance of some answers to a query:
true_relevance = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]])
# we predict some scores (relevance) for the answers
scores = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]])
ndcg_score(true_relevance, scores)
print(ndcg_score(true_relevance, scores))
print(ndcg_score(true_relevance, scores, 5))