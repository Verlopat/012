import pandas as pd
from scipy import stats
import statsmodels.api as sm

def regress_residual(target, regressors):
    X = sm.add_constant(regressors.copy(), has_constant="add")
    model = sm.OLS(target, X).fit()
    return model.resid, model

def partial_corr_pvalue(y, x, cond=None):
    if cond is None or cond.shape[1] == 0:
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    ry, _ = regress_residual(y, cond)
    rx, _ = regress_residual(x, cond)
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)

def first_stage_fstat(x, z, cond=None):
    zdf = z.to_frame() if isinstance(z, pd.Series) else z.copy()
    X = zdf if cond is None or cond.shape[1] == 0 else pd.concat([zdf, cond], axis=1)
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(x, X).fit()
    return float(model.fvalue) if model.fvalue is not None else 0.0, model

def iv_test(y, x, z, cond=None):
    zdf = z.to_frame() if isinstance(z, pd.Series) else z.copy()

    first_X = zdf if cond is None or cond.shape[1] == 0 else pd.concat([zdf, cond], axis=1)
    first_X = sm.add_constant(first_X, has_constant="add")
    first_stage = sm.OLS(x, first_X).fit()
    x_hat = first_stage.fittedvalues.rename("x_hat")

    second_X = x_hat.to_frame()
    if cond is not None and cond.shape[1] > 0:
        second_X = pd.concat([second_X, cond], axis=1)
    second_X = sm.add_constant(second_X, has_constant="add")
    second_stage = sm.OLS(y, second_X).fit()

    coef = float(second_stage.params.get("x_hat", 0.0))
    pval = float(second_stage.pvalues.get("x_hat", 1.0))
    return coef, pval

def proxy_test(y, x, proxy1, proxy2, cond=None):
    regs = pd.concat([x.rename("x"), proxy1.rename("proxy1")], axis=1)
    if cond is not None and cond.shape[1] > 0:
        regs = pd.concat([regs, cond], axis=1)
    resid_y, _ = regress_residual(y, regs)
    r, p = stats.pearsonr(resid_y, proxy2)
    return float(r), float(p)

def edge_metrics(true_edges, pred_edges):
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    shd = fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": shd,
    }
