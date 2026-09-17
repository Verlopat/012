import json
import pandas as pd
from config import PROJECT_NAME, SEED, N_SAMPLES, TAU_MAX, ALPHA, MAX_COND_DIM, MIN_FSTAT
from generate_data import generate_synthetic_iv_proxy_data
from baseline_pcmci import BaselinePCMCI
from iv_pcmci import IVPCMCI
from utils import edge_metrics

def main():
    print(f"=== {PROJECT_NAME} ===")

    df, true_edges, metadata = generate_synthetic_iv_proxy_data(
        n_samples=N_SAMPLES,
        seed=SEED
    )

    observed_nodes = metadata["observed_core_nodes"]

    df.to_csv("synthetic_data.csv", index=False)

    baseline = BaselinePCMCI(
        tau_max=TAU_MAX,
        alpha=ALPHA,
        max_cond_dim=MAX_COND_DIM,
        allowed_nodes=observed_nodes
    ).fit(df)

    protected_pairs = {("X", "Y")}
    forbidden_pairs = {("Y", "X"), ("W", "W"), ("Z", "Z")}

    ivpcmci = IVPCMCI(
        tau_max=TAU_MAX,
        alpha=ALPHA,
        max_cond_dim=MAX_COND_DIM,
        min_fstat=MIN_FSTAT,
        allowed_nodes=observed_nodes
    ).fit(
        df,
        instrument_map=metadata["instrument_map"],
        proxy_map=metadata["proxy_map"],
        protected_pairs=protected_pairs,
        forbidden_pairs=forbidden_pairs
    )

    baseline_edges = baseline.get_graph()
    iv_edges = ivpcmci.get_graph()

    baseline.get_summary().to_csv("baseline_summary.csv", index=False)
    ivpcmci.get_summary().to_csv("ivpcmci_summary.csv", index=False)

    baseline_metrics = edge_metrics(true_edges, baseline_edges)
    iv_metrics = edge_metrics(true_edges, iv_edges)

    metrics_df = pd.DataFrame([
        {"model": "BaselinePCMCI", **baseline_metrics},
        {"model": "IVPCMCI", **iv_metrics}
    ])
    metrics_df.to_csv("metrics.csv", index=False)

    with open("true_edges.json", "w") as f:
        json.dump(sorted(list(true_edges)), f, indent=2)
    with open("baseline_edges.json", "w") as f:
        json.dump(sorted(list(baseline_edges)), f, indent=2)
    with open("ivpcmci_edges.json", "w") as f:
        json.dump(sorted(list(iv_edges)), f, indent=2)

    print("\nTrue edges:")
    for e in sorted(true_edges):
        print(" ", e)

    print("\nBaseline edges:")
    for e in sorted(baseline_edges):
        print(" ", e)

    print("\nIV-PCMCI edges:")
    for e in sorted(iv_edges):
        print(" ", e)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print("\nSaved files:")
    print(" synthetic_data.csv")
    print(" baseline_summary.csv")
    print(" ivpcmci_summary.csv")
    print(" metrics.csv")
    print(" true_edges.json")
    print(" baseline_edges.json")
    print(" ivpcmci_edges.json")

if __name__ == "__main__":
    main()
