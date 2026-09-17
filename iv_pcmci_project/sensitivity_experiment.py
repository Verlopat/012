#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ALPHA, MIN_FSTAT
from iv_cond_ind_test import IVCondIndTest
from utils import edge_metrics

def generate_synthetic_iv_proxy_data_strength(n_samples=1200, seed=42, z_strength=2.0):
    rng = np.random.default_rng(seed)

    U = np.zeros(n_samples)
    Z = np.zeros(n_samples)
    X = np.zeros(n_samples)
    Y = np.zeros(n_samples)
    W = np.zeros(n_samples)
    P1 = np.zeros(n_samples)
    P2 = np.zeros(n_samples)

    for t in range(1, n_samples):
        U[t] = 0.75 * U[t - 1] + rng.normal(0, 0.45)
        Z[t] = rng.normal(0, 1.0)
        W[t] = 0.55 * W[t - 1] + rng.normal(0, 0.50)

        X[t] = (
            0.45 * X[t - 1]
            + z_strength * Z[t - 1]
            + 0.45 * U[t - 1]
            + 0.15 * W[t - 1]
            + rng.normal(0, 0.20)
        )

        Y[t] = (
            0.45 * Y[t - 1]
            + 0.85 * X[t - 1]
            + 0.45 * U[t - 1]
            + 0.15 * W[t - 1]
            + rng.normal(0, 0.20)
        )

        P1[t] = 0.95 * U[t] + rng.normal(0, 0.15)
        P2[t] = 0.90 * U[t] + rng.normal(0, 0.15)

    df = pd.DataFrame({
        "Z": Z,
        "X": X,
        "Y": Y,
        "W": W,
        "P1": P1,
        "P2": P2
    })

    true_edges = {
        ("Z", "X", 1),
        ("W", "X", 1),
        ("W", "Y", 1),
        ("X", "X", 1),
        ("Y", "Y", 1),
        ("X", "Y", 1),
    }

    metadata = {
        "instrument_map": {"X": ["Z"]},
        "proxy_map": {("X", "Y"): ("P1", "P2")},
        "observed_core_nodes": ["Z", "X", "Y", "W"],
        "proxy_nodes": ["P1", "P2"],
        "protected_pairs": {("X", "Y")},
        "forbidden_pairs": {("Y", "X"), ("W", "W"), ("Z", "Z")}
    }

    return df, true_edges, metadata

def run_iv_screen(df, metadata, alpha=0.01, min_fstat=5.0):
    observed = metadata["observed_core_nodes"]

    test = IVCondIndTest(
        series_df=df,
        var_names=observed,
        instrument_map=metadata["instrument_map"],
        proxy_map=metadata["proxy_map"],
        protected_pairs=metadata["protected_pairs"],
        forbidden_pairs=metadata["forbidden_pairs"],
        min_fstat=min_fstat,
        alpha=alpha,
        significance="analytic",
        verbosity=0
    )

    edges = set()
    rows = []

    for source in observed:
        for target in observed:
            lag = 1

            x = df[source].shift(lag).dropna().reset_index(drop=True)
            y = df[target].iloc[lag:].reset_index(drop=True)

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

            if pval < alpha and info["decision"]:
                edges.add((source, target, lag))

    return edges, pd.DataFrame(rows)

def main():
    strengths = [0.5, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0]
    results = []
    diag_rows = []

    for strength in strengths:
        print(f"Running strength = {strength}")

        df, true_edges, metadata = generate_synthetic_iv_proxy_data_strength(
            n_samples=1200,
            seed=42,
            z_strength=strength
        )

        pred_edges, summary_df = run_iv_screen(
            df,
            metadata,
            alpha=ALPHA,
            min_fstat=MIN_FSTAT
        )

        metrics = edge_metrics(true_edges, pred_edges)

        xy_row = summary_df[
            (summary_df["source"] == "X") &
            (summary_df["target"] == "Y") &
            (summary_df["lag"] == 1)
        ].copy()

        fstat = float(xy_row["fstat"].iloc[0]) if not xy_row.empty and pd.notna(xy_row["fstat"].iloc[0]) else np.nan
        proxy_p = float(xy_row["proxy_p"].iloc[0]) if not xy_row.empty and pd.notna(xy_row["proxy_p"].iloc[0]) else np.nan
        method = str(xy_row["method"].iloc[0]) if not xy_row.empty else "missing"
        decision = bool(xy_row["decision"].iloc[0]) if not xy_row.empty else False

        result_row = {
            "strength": strength,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "shd": metrics["shd"],
            "xy_fstat": fstat,
            "xy_proxy_p": proxy_p,
            "xy_method": method,
            "xy_decision": decision
        }
        results.append(result_row)

        summary_df["strength"] = strength
        diag_rows.append(summary_df)

    results_df = pd.DataFrame(results)
    diagnostics_df = pd.concat(diag_rows, ignore_index=True)

    results_df.to_csv("sensitivity_results.csv", index=False)
    diagnostics_df.to_csv("sensitivity_diagnostics.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["strength"], results_df["shd"], marker="o", linewidth=2, label="SHD")
    plt.plot(results_df["strength"], results_df["recall"], marker="s", linewidth=2, label="Recall")
    plt.plot(results_df["strength"], results_df["precision"], marker="^", linewidth=2, label="Precision")
    plt.xlabel("Instrument strength (Z -> X coefficient)")
    plt.ylabel("Metric value")
    plt.title("IV-PCMCI sensitivity to instrument strength")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sensitivity_plot.png", dpi=200)

    print("\nSensitivity results:")
    print(results_df.to_string(index=False))

    print("\nSaved files:")
    print(" sensitivity_results.csv")
    print(" sensitivity_diagnostics.csv")
    print(" sensitivity_plot.png")

if __name__ == "__main__":
    main()
