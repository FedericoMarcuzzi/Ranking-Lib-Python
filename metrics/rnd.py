import numpy as np
from ..misc import stable_argsort
from .metrics import Metric

class rND(Metric):
    def __init__(self, cutoff, step, n_jobs=1):
        super().__init__("rND", cutoff, False, n_jobs),

        self.step = step

    def init_normalizer(self, y_true, qlen, set_name):
        self.sets_norm[set_name] = self._per_queries(y_true, y_true, qlen, self._max_rND_query_norm)

    def eval_query(self, y_true, y_score, set_name=None, id=None):
        norm = -1 if set_name is None or id is None else self.sets_norm[set_name][id]
        return self.rND_query(y_true, y_score, self.step, self.cutoff, norm=norm)
    
    def _max_rND_query_norm(self, y_true, y_score, set_name=None, id=None):
        return self.max_rND_query(y_true, self.step, self.cutoff)

    def rD_query(self, y_true, step, cutoff, skip_sort=True, y_score=None):
        if not skip_sort:
            y_true = y_true[stable_argsort(y_score)]

        n_itms = len(y_true)
        cutoff = n_itms if cutoff <= 0 else cutoff
        
        n_prot = np.sum(y_true)
        if n_prot == 0 or n_prot == n_itms: return 0.
        
        end_ = min(cutoff, n_itms)
        slice_list = [step] * (end_ // step)
        if end_ % step != 0:
            slice_list += [end_ % step]

        r_cum = np.cumsum(y_true)
        cum_slice_list = np.cumsum(slice_list)

        rD_query = np.sum(np.abs(r_cum[cum_slice_list - 1] / cum_slice_list - n_prot / n_itms) * (1 / np.log2(cum_slice_list + 1)))
        return 0. if rD_query < 1e-7 else rD_query

    def max_rND_query(self, y_true, step, cutoff):
        n_itms = len(y_true)
        cutoff = n_itms if cutoff <= 0 else cutoff
        end_ = min(cutoff, n_itms)

        if end_ <= step: return 0.

        data_ = np.zeros((n_itms, ))
        data_[:np.sum(y_true)] = 1

        max_1 = self.rD_query(data_, step, cutoff, skip_sort=True)
        max_2 = self.rD_query(np.flip(data_), step, cutoff, skip_sort=True)
        return np.maximum(max_1, max_2)

    def rND_query(self, y_true, y_score, step, cutoff, norm=-1):
        cutoff = len(y_true) if cutoff <= 0 else cutoff
        max_ = self.max_rND_query(y_true, step, cutoff) if norm < 0 else norm

        if max_ <= 0.: return 0.

        rnk = stable_argsort(y_score)
        print(y_true[rnk])
        return self.rD_query(y_true[rnk], step, cutoff) / max_