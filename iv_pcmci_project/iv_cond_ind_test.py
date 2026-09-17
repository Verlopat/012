import numpy as np
import pandas as pd
from scipy import stats
from tigramite.independence_tests.parcorr import ParCorr
from utils import first_stage_fstat, iv_test, proxy_test

class IVCondIndTest(ParCorr):
    def __init__(
        self,
        series_df=None,
        var_names=None,
        instrument_map=None,
        proxy_map=None,
        protected_pairs=None,
        forbidden_pairs=None,
        min_fstat=5.0,
        alpha=0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.series_df = series_df
        self.var_names = var_names or []
        self.instrument_map = instrument_map or {}
        self.proxy_map = proxy_map or {}
        self.protected_pairs = set(protected_pairs or [])
        self.forbidden_pairs = set(forbidden_pairs or [])
        self.min_fstat = min_fstat
        self.alpha = alpha
        self.last_info = {}

    def set_context(self, source=None, target=None, lag=None):
        self._source = source
        self._target = target
        self._lag = lag

    def run_test_raw(self, x, y, z=None, x_type=None, y_type=None, z_type=None):
        z = np.empty((0, x.shape[1])) if z is None else z

        x_series = pd.Series(np.asarray(x).reshape(-1))
        y_series = pd.Series(np.asarray(y).reshape(-1))
        if z.size == 0:
            z_df = pd.DataFrame()
        else:
            z_df = pd.DataFrame(np.asarray(z).T, columns=[f"z{i}" for i in range(z.shape[0])])

        std_r, std_p = stats.pearsonr(x_series, y_series)

        source = getattr(self, "_source", None)
        target = getattr(self, "_target", None)
        lag = getattr(self, "_lag", None)
        pair = (source, target)

        info = {
            "source": source,
            "target": target,
            "lag": lag,
            "std_r": float(std_r),
            "std_p": float(std_p),
            "decision": False,
            "method": "standard",
            "fstat": None,
            "iv_coef": None,
            "iv_p": None,
            "proxy_r": None,
            "proxy_p": None,
        }

        if pair in self.forbidden_pairs:
            info["method"] = "forbidden"
            self.last_info = info
            return 0.0, 1.0

        if pair not in self.protected_pairs:
            info["decision"] = std_p < self.alpha
            self.last_info = info
            return abs(float(std_r)), float(std_p)

        passed_standard = std_p < (self.alpha * 5)
        passed_iv = False
        passed_proxy = False
        out_val = abs(float(std_r))
        out_p = float(std_p)

        if source in self.instrument_map and self.series_df is not None:
            valid_instr = [c for c in self.instrument_map[source] if c in self.series_df.columns]
            if len(valid_instr) > 0:
                try:
                    z_instr = self.series_df[valid_instr].iloc[-len(x_series):].reset_index(drop=True)
                    fstat, _ = first_stage_fstat(x_series, z_instr, z_df if len(z_df.columns) else None)
                    info["fstat"] = float(fstat)
                    if fstat >= self.min_fstat:
                        iv_coef, iv_p = iv_test(y_series, x_series, z_instr, z_df if len(z_df.columns) else None)
                        info["iv_coef"] = float(iv_coef)
                        info["iv_p"] = float(iv_p)
                        if iv_p < (self.alpha * 5):
                            passed_iv = True
                            out_val = max(out_val, abs(float(iv_coef)))
                            out_p = min(out_p, float(iv_p))
                except Exception:
                    pass

        if pair in self.proxy_map and self.series_df is not None:
            p1, p2 = self.proxy_map[pair]
            if p1 in self.series_df.columns and p2 in self.series_df.columns:
                try:
                    p1s = self.series_df[p1].iloc[-len(y_series):].reset_index(drop=True)
                    p2s = self.series_df[p2].iloc[-len(y_series):].reset_index(drop=True)
                    proxy_r, proxy_p = proxy_test(y_series, x_series, p1s, p2s, z_df if len(z_df.columns) else None)
                    info["proxy_r"] = float(proxy_r)
                    info["proxy_p"] = float(proxy_p)
                    if proxy_p < (self.alpha * 5):
                        passed_proxy = True
                        out_val = max(out_val, abs(float(proxy_r)))
                        out_p = min(out_p, float(proxy_p))
                except Exception:
                    pass

        info["decision"] = bool(passed_standard and (passed_iv or passed_proxy))
        if passed_iv:
            info["method"] = "iv"
        elif passed_proxy:
            info["method"] = "proxy"
        else:
            info["method"] = "protected_reject"

        self.last_info = info

        if info["decision"]:
            return float(out_val), float(out_p)
        return 0.0, 1.0
