import numpy as np
from joblib import Parallel, delayed

class Metric():
    def __init__(self, metric="ndcg", cutoff=0):
        self.cutoff = cutoff
        self.metric_name = metric

        if metric == "ndcg":
            self.metric = self._ndcg_query

        if metric == "dcg":
            self.metric = self._dcg_query

    def _dcg_query(self, y, y_score, cutoff):
        cutoff = min(cutoff, y_score.shape[0])
        idx_sorted = len(y_score) - 1 - np.argsort(y_score[::-1], kind='stable')[::-1][:cutoff]

        discount = np.log2(np.arange(2, cutoff + 2))
        gain = np.exp2(y[idx_sorted]) - 1.0

        dcg = (gain / discount).sum()
        return dcg

    def _ndcg_query(self, y, y_pred, cutoff):
        idcg_score = self._dcg_query(y, y, cutoff)
        dcg_score = self._dcg_query(y, y_pred, cutoff)

        if idcg_score != 0:
            ndcg = dcg_score / idcg_score
        else:
            ndcg = 1

        return ndcg

    def _per_queries(self, y, y_score, qlen, cutoff, n_jobs=1):
        qlen = np.array(qlen).astype(int)
        cum = np.cumsum(qlen)[:-1]

        ndcg_queries = []

        if n_jobs == 1:
            for labels, scores in zip(np.array_split(y, cum), np.array_split(y_score, cum)):
                ndcg_queries.append(self.metric(labels, scores, cutoff))
        else:
            ndcg_queries = Parallel(n_jobs=n_jobs)(delayed(self.metric)(labels, scores, cutoff)
                                        for labels, scores in zip(np.array_split(y, cum), np.array_split(y_score, cum)))

        return np.asarray(ndcg_queries)

    def eval(self, y, y_score, qlen, cutoff, n_jobs=1):
        return np.mean(self._per_queries(y, y_score, qlen, cutoff, n_jobs))
    
    def eval_query(self, y_true, scores):
        return self.metric(y_true, scores, self.cutoff)

    def set_cutoff(self, cutoff):
        self.cutoff = cutoff

    def get_cutoff(self):
        return self.cutoff

    def get_metric_name(self):  
        return self.metric_name

    def eval_lgbm(self, y_score, data):
        assert self.cutoff != 0, "call set_cutoff first! cutoff must be greater than 0!"
        y = data.get_label()
        qlen = data.get_group()
        name = self.metric_name + f"@{self.cutoff}"

        return name, np.mean(self._per_queries(y, y_score, qlen, self.cutoff)), True


class PL_Rank_Metric(Metric):
    def __init__(self, metric="ndcg", cutoff=0):
        super.__init__(metric, cutoff)

        if metric == "fair":  
            self.metric = None

    def eval(self, y, y_score, qlen, cutoff, rank_weights=None, num_samples=None, n_jobs=1):   
        if self.metric_name != "fair":  
            return np.mean(self._per_queries(y, y_score, qlen, cutoff, n_jobs)) 
        else:   
            return self._eval_fairness(y, y_score, qlen, cutoff, rank_weights, num_samples)

    def multiple_cutoff_rankings(self, scores, cutoff, invert=True):
        n_samples = scores.shape[0]
        n_docs = scores.shape[1]
        cutoff = min(n_docs, cutoff)

        ind = np.arange(n_samples)
        partition = np.argpartition(scores, cutoff-1, axis=1)[:,:cutoff]
        sorted_partition = np.argsort(scores[ind[:, None], partition], axis=1)
        rankings = partition[ind[:, None], sorted_partition]

        if not invert:
            return rankings, None
        else:
            inverted = np.full((n_samples, n_docs), cutoff, dtype=rankings.dtype)
            inverted[ind[:, None], rankings] = np.arange(cutoff)[None, :]
            return rankings, inverted
    
    def gumbel_sample_rankings(self, log_scores, n_samples, cutoff=None, inverted=False, doc_prob=False, return_gumbel=False):  
        np.random.seed(7)
        n_docs = log_scores.shape[0]

        if cutoff:
            ranking_len = min(n_docs, cutoff)
        else:
            ranking_len = n_docs

        gumbel_samples = np.random.gumbel(size=(n_samples, n_docs))
        gumbel_scores = -(log_scores[None,:]+gumbel_samples)

        rankings, inv_rankings = self.multiple_cutoff_rankings(gumbel_scores, ranking_len, invert=inverted)

        if not doc_prob:
            if not return_gumbel:
                return rankings, inv_rankings, None, None, None
            else:
                return rankings, inv_rankings, None, None, gumbel_scores
    
    def _eval_fairness(self, y, y_score, qlen, cutoff, rank_weights, num_samples):  
        state = np.random.get_state()
        np.random.seed(7)
        y = 2**y-1
        cutoff = self.cutoff
        qlen = np.array(qlen).astype(int)
        cum = np.cumsum(qlen)[:-1]

        result = 0.
        squared_result = 0.
        for q_labels, q_scores in zip(np.array_split(y, cum), np.array_split(y_score, cum)):
            if np.sum(q_labels) > 0 and q_labels.size > 1:
                sampled_rankings = self.gumbel_sample_rankings(q_scores, num_samples, cutoff=cutoff)[0]
                
                q_n_docs = q_labels.shape[0]
                q_cutoff = min(cutoff, q_n_docs)
                doc_exposure = np.zeros(q_n_docs, dtype=np.float64)
                np.add.at(doc_exposure, sampled_rankings, rank_weights[:q_cutoff])
                doc_exposure /= num_samples
                
                swap_reward = doc_exposure[:,None]*q_labels[None,:]
                
                q_result = np.mean((swap_reward-swap_reward.T)**2.)
                q_result *= q_n_docs/(q_n_docs-1.)

                q_squared = np.mean(np.abs(swap_reward-swap_reward.T))
                q_squared *= q_n_docs/(q_n_docs-1.)
                
                result += q_result
                squared_result += q_squared

        result /= qlen.shape[0]
        squared_result /= qlen.shape[0]
        np.random.set_state(state)
        return result, squared_result
