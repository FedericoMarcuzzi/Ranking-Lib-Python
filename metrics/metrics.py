import numpy as np
from joblib import Parallel, delayed
from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self, metric_name, cutoff, best_higher, n_jobs=1):
        self.metric_name = metric_name
        self.cutoff = cutoff
        self.best_higher = best_higher
        self.n_jobs = n_jobs

        self.sets_norm = {}

    @abstractmethod
    def init_normalizer(self):
        pass
    
    @abstractmethod
    def eval_query(self, y_true, scores, set_name=None, id=None):
        pass

    def eval(self, y_true, y_score, qlen, set_name=None):
        return np.mean(self._per_queries(y_true, y_score, qlen, self.eval_query, set_name))
    
    def _per_queries(self, y_true, y_score, qlen, eval_foo, set_name=None):
        qlen = np.array(qlen).astype(int)
        cumsum = np.cumsum(qlen)[:-1]

        res_queries = []
        y_true_query = np.array_split(y_true, cumsum)
        y_score_query = np.array_split(y_score, cumsum)

        for i in np.arange(len(y_true_query)):
            res_queries.append(eval_foo(y_true_query[i], y_score_query[i], set_name, i))

        if self.n_jobs == 1:
            for i in np.arange(len(y_true_query)):
                res_queries.append(eval_foo(y_true_query[i], y_score_query[i], set_name, i))
        else:
            res_queries = Parallel(n_jobs=self.n_jobs)(delayed(eval_foo)
                (y_true_query[i], y_score_query[i], set_name, i)
                    for i in np.arange(len(y_true_query)))


        return np.asarray(res_queries)
    
    def eval_lgbm(self, y_score, data):
        y_true = data.get_label()
        qlen = data.get_group()
        name = self.metric_name + f"@{self.cutoff}"
        set_name = None
        if "name" in data:
            set_name = data["name"]

        return name, np.mean(self._per_queries(y_true, y_score, qlen, self.eval_query, set_name)), self.best_higher
    
    def get_cutoff(self):
        return self.cutoff

    def get_metric_name(self):  
        return self.metric_name