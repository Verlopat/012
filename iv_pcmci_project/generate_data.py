import numpy as np
import pandas as pd

def generate_synthetic_iv_proxy_data(n_samples=1200, seed=42):
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
            + 2.00 * Z[t - 1]
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

if __name__ == "__main__":
    df, true_edges, metadata = generate_synthetic_iv_proxy_data()
    print(df.head())
    print("\nTrue edges:")
    for edge in sorted(true_edges):
        print(edge)
    print("\nMetadata:")
    print(metadata)
