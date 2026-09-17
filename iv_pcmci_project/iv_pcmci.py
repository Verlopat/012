import pandas as pd
from utils import make_lagged_dataframe, partial_corr_pvalue, first_stage_fstat, iv_test, proxy_test, powerset_limited

class IVPCMCI:
    def __init__(self, tau_max=1, alpha=0.01, max_cond_dim=2, min_fstat=10.0, allowed_nodes=None):
        self.tau_max = tau_max
        self.alpha = alpha
        self.max_cond_dim = max_cond_dim
        self.min_fstat = min_fstat
        self.allowed_nodes = allowed_nodes
        self.graph_ = set()
        self.summary_ = None

    def fit(self, df, instrument_map=None, proxy_map=None, protected_pairs=None, forbidden_pairs=None):
        instrument_map = instrument_map or {}
        proxy_map = proxy_map or {}
        protected_pairs = protected_pairs or set()
        forbidden_pairs = forbidden_pairs or set()

        cols = self.allowed_nodes if self.allowed_nodes is not None else list(df.columns)
        lagged = make_lagged_dataframe(df, self.tau_max)

        rows = []
        edges = set()

        for target in cols:
            y = lagged[target]
            for source in cols:
                for lag in range(1, self.tau_max + 1):
                    if (source, target) in forbidden_pairs:
                        rows.append({
                            "source": source,
                            "target": target,
                            "lag": lag,
                            "cond_set": [],
                            "std_r": None,
                            "std_p": None,
                            "decision": False,
                            "method": "forbidden",
                            "fstat": None,
                            "iv_coef": None,
                            "iv_p": None,
                            "proxy_r": None,
                            "proxy_p": None,
                        })
                        continue

                    xname = f"{source}_lag{lag}"
                    x = lagged[xname]

                    cond_candidates = [f"{c}_lag1" for c in cols if c != source and c != target]
                    cond_sets = powerset_limited(cond_candidates, self.max_cond_dim)

                    best = None
                    best_p = None

                    for cond_names in cond_sets:
                        cond = lagged[cond_names] if cond_names else pd.DataFrame(index=lagged.index)
                        std_r, std_p = partial_corr_pvalue(y, x, cond)

                        record = {
                            "source": source,
                            "target": target,
                            "lag": lag,
                            "cond_set": cond_names,
                            "std_r": std_r,
                            "std_p": std_p,
                            "decision": False,
                            "method": "standard",
                            "fstat": None,
                            "iv_coef": None,
                            "iv_p": None,
                            "proxy_r": None,
                            "proxy_p": None,
                        }

                        is_protected = (source, target) in protected_pairs

                        if not is_protected:
                            record["decision"] = std_p < self.alpha
                            record["method"] = "standard"
                        else:
                            passed_standard = std_p < (self.alpha * 5)
                            passed_iv = False
                            passed_proxy = False

                            if source in instrument_map:
                                valid_instr = []
                                for instr in instrument_map[source]:
                                    zname = f"{instr}_lag{lag}"
                                    if zname in lagged.columns:
                                        valid_instr.append(zname)
                                if valid_instr:
                                    z = lagged[valid_instr]
                                    fstat, _ = first_stage_fstat(x, z, cond)
                                    record["fstat"] = fstat
                                    if fstat >= self.min_fstat:
                                        iv_coef, iv_p = iv_test(y, x, z, cond)
                                        record["iv_coef"] = iv_coef
                                        record["iv_p"] = iv_p
                                        passed_iv = iv_p < (self.alpha * 5)

                            if (source, target) in proxy_map:
                                p1, p2 = proxy_map[(source, target)]
                                p1name = f"{p1}_lag{lag}"
                                p2name = f"{p2}_lag{lag}"
                                if p1name in lagged.columns and p2name in lagged.columns:
                                    pr, pp = proxy_test(y, x, lagged[p1name], lagged[p2name], cond)
                                    record["proxy_r"] = pr
                                    record["proxy_p"] = pp
                                    passed_proxy = pp < (self.alpha * 5)

                            record["decision"] = passed_standard and (passed_iv or passed_proxy)
                            if passed_iv:
                                record["method"] = "iv"
                            elif passed_proxy:
                                record["method"] = "proxy"
                            else:
                                record["method"] = "protected_reject"

                        select_p = min(
                            record["std_p"] if record["std_p"] is not None else 1.0,
                            record["iv_p"] if record["iv_p"] is not None else 1.0,
                            record["proxy_p"] if record["proxy_p"] is not None else 1.0,
                        )

                        if best is None or select_p < best_p:
                            best = record
                            best_p = select_p

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
