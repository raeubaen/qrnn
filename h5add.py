import pandas as pd
import sys

outfile = sys.argv[1]
infiles = sys.argv[2:]

df_out = pd.concat([pd.read_hdf(infile) for infile in infiles])
df_out.to_hdf(outfile, "tree")

