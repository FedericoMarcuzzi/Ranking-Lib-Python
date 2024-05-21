import numpy as np
from ..misc import stable_argsort
from .metrics import Metric

class DCG(Metric):
    def __init__(self, cutoff, n_jobs=1):
        super().__init__("DCG", cutoff, True, n_jobs)

    def init_normalizer(self, y_true, qlen, set_name):
        print("DCG metric does not require normalization.")

    def eval_query(self, y_true, y_score, set_name=None, id=None):
        return self.DCG_query(y_true, y_score, self.cutoff)

    def DCG_query(self, y_true, y_score, cutoff):
        cutoff = min(cutoff, y_score.shape[0])
        idx_sorted = stable_argsort(y_score)[:cutoff]

        discount = np.log2(np.arange(2, cutoff + 2))
        gain = np.exp2(y_true[idx_sorted]) - 1.0
        return np.sum(gain / discount)
    
class NDCG(DCG):
    def __init__(self, cutoff, n_jobs=1):
        super().__init__(cutoff, n_jobs)
        self.metric_name = "NDCG"

    def init_normalizer(self, y_true, qlen, set_name):
        self.sets_norm[set_name] = self._per_queries(y_true, y_true, qlen, self._max_DCG_query_norm)

    def eval_query(self, y_true, y_score, set_name=None, id=None):
        norm = -1 if set_name is None or id is None else self.sets_norm[set_name][id]
        return self.NDCG_query(y_true, y_score, self.cutoff, idcg_score=norm)
    
    def _max_DCG_query_norm(self, y_true, y_score, set_name=None, id=None):
        return super().DCG_query(y_true, y_score, self.cutoff)
    
    def NDCG_query(self, y_true, y_pred, cutoff, idcg_score=None):
        if idcg_score is not None:
            idcg_score = super().DCG_query(y_true, y_true, cutoff)

        dcg_score = super().DCG_query(y_true, y_pred, cutoff)
        return 1. if idcg_score == 0 else dcg_score / idcg_score

    