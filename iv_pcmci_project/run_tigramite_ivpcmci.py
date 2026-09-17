import json
import numpy as np
import pandas as pd

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from config import PROJECT_NAME, SEED, N_SAMPLES, TAU_MAX, ALPHA, PC_ALPHA, MIN_FSTAT
from generate_data import generate_synthetic_iv_proxy_data
from iv_cond_ind_test import IVCondIndTest
from utils import edge_metrics

def extract_edges_from_pmatrix(p_matrix, val_matrix, var_names, alpha_level=0.01, tau_max=1):
    edges = set()
    rows = []
    N = len(var_names)

    for i in range(N):
        for j in range(N):
            for tau in range(1, tau_max + 1):
                pval = float(p_matrix[i, j, tau])
                val = float(val_matrix[i, j, tau])
                src = var_names[i]
                tgt = var_names[j]
                if pval < alpha_level:
                    edges.add((src, tgt, tau))
                rows.append({
                    "source": src,
                    "target": tgt,
                    "lag": tau,
                    "pval": pval,
                    "val": val,
                    "decision": pval < alpha_level
                })
    return edges, pd.DataFrame(rows)

def apply_post_constraints(edges, protected_pairs, forbidden_pairs):
    filtered = set()
    for edge in edges:
        s, t, lag = edge
        if (s, t) in forbidden_pairs:
            continue
        filtered.add(edge)
    return filtered

def build_custom_iv_edges(df, metadata):
    observed = metadata["observed_core_nodes"]
    protected_pairs = metadata["protected_pairs"]
    forbidden_pairs = metadata["forbidden_pairs"]

    test = IVCondIndTest(
        series_df=df,
        var_names=observed,
        instrument_map=metadata["instrument_map"],
        proxy_map=metadata["proxy_map"],
        protected_pairs=protected_pairs,
        forbidden_pairs=forbidden_pairs,
        min_fstat=MIN_FSTAT,
        alpha=ALPHA,
        significance="analytic",
        verbosity=0
    )

    rows = []
    edges = set()

    for source in observed:
        for target in observed:
            for lag in range(1, TAU_MAX + 1):
                x = df[source].shift(lag).dropna().reset_index(drop=True)
                y = df[target].iloc[lag:].reset_index(drop=True)

                if len(x) != len(y):
                    m = min(len(x), len(y))
                    x = x.iloc[:m]
                    y = y.iloc[:m]

                test.set_context(source=source, target=target, lag=lag)
                val, pval = test.run_test_raw(
                    x.to_numpy().reshape(1, -1),
                    y.to_numpy().reshape(1, -1),
                    np.empty((0, len(x)))
                )
                info = test.last_info.copy()
                info["val"] = val
                info["pval"] = pval
                rows.append(info)

                if pval < ALPHA and info["decision"]:
                    edges.add((source, target, lag))

    return edges, pd.DataFrame(rows)

def main():
    print(f"=== {PROJECT_NAME} ===")

    df, true_edges, metadata = generate_synthetic_iv_proxy_data(
        n_samples=N_SAMPLES,
        seed=SEED
    )

    observed = metadata["observed_core_nodes"]
    observed_df = df[observed].copy()
    observed_df.to_csv("synthetic_data.csv", index=False)

    data_array = observed_df.values
    dataframe = pp.DataFrame(data_array, var_names=observed)

    baseline_test = ParCorr(significance="analytic", verbosity=0)
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=baseline_test,
        verbosity=0
    )

    baseline_results = pcmci.run_pcmci(
        tau_max=TAU_MAX,
        pc_alpha=PC_ALPHA,
        alpha_level=ALPHA
    )

    baseline_edges, baseline_summary = extract_edges_from_pmatrix(
        baseline_results["p_matrix"],
        baseline_results["val_matrix"],
        observed,
        alpha_level=ALPHA,
        tau_max=TAU_MAX
    )
    baseline_edges = apply_post_constraints(
        baseline_edges,
        metadata["protected_pairs"],
        metadata["forbidden_pairs"]
    )

    iv_edges, iv_summary = build_custom_iv_edges(df, metadata)

    baseline_metrics = edge_metrics(true_edges, baseline_edges)
    iv_metrics = edge_metrics(true_edges, iv_edges)

    baseline_summary.to_csv("tigramite_baseline_summary.csv", index=False)
    iv_summary.to_csv("tigramite_iv_summary.csv", index=False)

    metrics_df = pd.DataFrame([
        {"model": "TigramiteBaselinePCMCI", **baseline_metrics},
        {"model": "TigramiteIVPCMCI", **iv_metrics}
    ])
    metrics_df.to_csv("tigramite_metrics.csv", index=False)

    with open("true_edges.json", "w") as f:
        json.dump(sorted(list(true_edges)), f, indent=2)
    with open("tigramite_baseline_edges.json", "w") as f:
        json.dump(sorted(list(baseline_edges)), f, indent=2)
    with open("tigramite_iv_edges.json", "w") as f:
        json.dump(sorted(list(iv_edges)), f, indent=2)

    print("\nTrue edges:")
    for e in sorted(true_edges):
        print(" ", e)

    print("\nTigramite baseline edges:")
    for e in sorted(baseline_edges):
        print(" ", e)

    print("\nTigramite IV edges:")
    for e in sorted(iv_edges):
        print(" ", e)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print("\nSaved files:")
    print(" synthetic_data.csv")
    print(" tigramite_baseline_summary.csv")
    print(" tigramite_iv_summary.csv")
    print(" tigramite_metrics.csv")
    print(" true_edges.json")
    print(" tigramite_baseline_edges.json")
    print(" tigramite_iv_edges.json")

if __name__ == "__main__":
    main()
