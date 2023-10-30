import pandas as pd
from matplotlib import pyplot as plt
import os
import sys

mat = [
  ["mc", "mc", "data"],
  ["tmp_dfs/weightedsys", "dfs_sys/split1", "tmp_dfs/weightedsys"],
  ["", "_corr", ""]
]

keys1 = ["mc_uncorr", "mc_corr", "data"]
keys2 = ["type", "folder", "suffix"]

dct = {keys1[i]: {keys2[j]: mat[j][i] for j in range(3)} for i in range(3)}

filegen = "df_{}_{}_train_split1{}.h5"

variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']

eos_path = "../eos/"
plot_folder = f"{eos_path}/www/qrnn_plots/comparison_try2"
os.system(f"mkdir -p {plot_folder}")
os.system(f"cp {eos_path}/www/index.php {plot_folder}")

for var in variables:
  for ebee in ["EE", "EB"]:
    for key1 in ["data", "mc_uncorr", "mc_corr"]:
      subdct = dct[key1]
      file = filegen.format(subdct["type"], ebee, subdct["suffix"])
      df = pd.read_hdf(f"{subdct['folder']}/{file}")
      if key1 == "data":
        _, bins, __ = plt.hist(df[f"{var}{subdct['suffix']}"], bins=100, histtype="step", label=key1)
      else:
        plt.hist(df[f"{var}{subdct['suffix']}"], bins=bins[:-1], histtype="step", label=key1)
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{plot_folder}/{var}.png")
    plt.close()
