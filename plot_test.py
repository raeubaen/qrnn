import pandas as pd
from matplotlib import pyplot as plt
import os

folders = ["tmp_dfs/sys", "tmp_dfs/weightedsys", "dfs_sys/split1"]

filegen = "df_{}_{}_{}train_split1{}.h5"

for folder in folders:
  eos_path = "../eos/"
  plot_folder = f"{eos_path}/www/qrnn_plots/{folder.replace('/', '_')}"
  os.system(f"mkdir -p {plot_folder}")
  os.system(f"cp {eos_path}/www/index.php {plot_folder}")
  for t in ["mc", "data"]:
    for v in ["Iso_", ""]:
      for s in ["", "_corr", "_corr_final"]:
        for ebee in ["EE", "EB"]:
          file = filegen.format(t,  ebee, v, s)
          try:
            df = pd.read_hdf(f"{folder}/{file}")
          except FileNotFoundError:
            print(f"{folder}/{file} not found")
            continue
          else:
            for key in df.columns:
              try:
                plt.hist(df[key], bins=100)
              except ValueError:
                print(f"ERROR in {folder}/{file} - key: {key}")
              else:
                plt.yscale("log")
                plt.savefig(f"{plot_folder}/{file.replace('.h5', f'_{key}.png')}")
                plt.close()

