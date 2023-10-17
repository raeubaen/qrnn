import pandas as pd
import uproot
import sys

infile = sys.argv[1]
outfile = sys.argv[2]
df = pd.read_hdf(infile)
data = {key: df[key] for key in df.columns}
f = uproot.recreate(outfile)
f["tree"] = data
f.close()

