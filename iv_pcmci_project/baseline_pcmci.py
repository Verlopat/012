import pandas as pd
from utils import make_lagged_dataframe, partial_corr_pvalue, powerset_limited

class BaselinePCMCI:
    def __init__(self, tau_max=1, alpha=0.01, max_cond_dim=2, allowed_nodes=None):
        self.tau_max = tau_max
        self.alpha = alpha
        self.max_cond_dim = max_cond_dim
        self.allowed_nodes = allowed_nodes
        self.graph_ = set()
        self.summary_ = None

    def fit(self, df):
        cols = self.allowed_nodes if self.allowed_nodes is not None else list(df.columns)
        lagged = make_lagged_dataframe(df[cols], self.tau_max)

        rows = []
        edges = set()

        for target in cols:
            y = lagged[target]
            for source in cols:
                for lag in range(1, self.tau_max + 1):
                    xname = f"{source}_lag{lag}"
                    x = lagged[xname]

                    cond_candidates = [
                        f"{c}_lag1" for c in cols
                        if c != source
                    ]
                    cond_sets = powerset_limited(cond_candidates, self.max_cond_dim)

                    best = None
                    for cond_names in cond_sets:
                        cond = lagged[cond_names] if cond_names else pd.DataFrame(index=lagged.index)
                        r, p = partial_corr_pvalue(y, x, cond)
                        rec = {
                            "source": source,
                            "target": target,
                            "lag": lag,
                            "cond_set": cond_names,
                            "std_r": r,
                            "std_p": p,
                            "decision": p < self.alpha,
                            "method": "standard"
                        }
                        if best is None or p > best["std_p"]:
                            best = rec

                    rows.append(best)
                    if best["decision"]:
                        edges.add((source, target, lag))

        self.graph_ = edges
        self.summary_ = pd.DataFrame(rows)
        return self

    def get_graph(self):
        return self.graph_

    def get_summary(self):
        return self.summary_.copy()
