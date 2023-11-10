import argparse
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn import preprocessing 
import pickle
import gzip

from qrnn import get_compiled_model

if __name__ == "__main__":

    num_hidden_layers = 5
    num_connected_layers = 3
    num_units = [8, 8, 8, 16, 16]
    act = ['relu' for _ in range(num_hidden_layers)]
    dropout = [0.4, 0.4, 0.4, 0.4]
    gauss_std = [0.2, 0.2, 0.2, 0.2, 0.2]

    model = get_compiled_model(
      np.linspace(0, 1, 21), np.ones((21)), 
      input_dim=4, 
      num_hidden_layers=num_hidden_layers, num_units=num_units, act=act, 
      num_connected_layers=num_connected_layers
    )

    model.summary()

