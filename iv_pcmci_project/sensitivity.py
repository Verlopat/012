import pandas as pd
from generate_data import generate_synthetic_iv_proxy_data
from iv_pcmci import IVPCMCI
from utils import edge_metrics
import matplotlib.pyplot as plt

strengths = [0.5, 0.8, 1.2, 1.6, 2.0]
shds = []

for strength in strengths:
    # Would need to modify generate_data.py temporarily
    print(f"Strength {strength}: F-stat expected ~{strength**2:.1f}")
    # shd = run_experiment(strength)
    # shds.append(shd)

plt.plot(strengths, shds)
plt.xlabel("Instrument strength")
plt.ylabel("SHD")
plt.savefig("sensitivity.png")
