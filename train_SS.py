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
import os 

from qrnn import trainQuantile, predict, scale
from mylib.transformer import transform, inverse_transform



def compute_qweights(sr, qs, weights=None):
    quantiles = np.quantile(sr, qs)
    es = np.array(sr)[:,None] - quantiles
    huber_e = Hubber(es, 1.e-4, signed=True)
    loss = np.maximum(qs*huber_e, (qs-1.)*huber_e)
    qweights = 1./np.average(loss, axis=0, weights=weights)
    return qweights/np.min(qweights)

def Hubber(e, delta=0.1, signed=False):
    is_small_e = np.abs(e) < delta
    small_e = np.square(e) / (2.*delta)
    big_e = np.abs(e) - delta/2.
    if signed:
        return np.sign(e)*np.where(is_small_e, small_e, big_e) 
    else: 
        return np.where(is_small_e, small_e, big_e)
    



def main(options):
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    weight = ['ml_weight']

    data_key = 'data' 
    EBEE = options.EBEE 

    spl = options.split
    if spl in [1, 2]: 
        inputtrain = 'tmp_dfs/weightedsys/df_{}_{}_train_split{}.h5'.format(data_key, EBEE, spl)
    else: 
        inputtrain = 'weighted_dfs/df_{}_{}_train.h5'.format(data_key, EBEE)
        print(f"Wrong argument '-s' ('--split'), argument must have value 1 or 2. Now using defalt dataframe {inputtrain}")
#    inputtest = 'weighted_dfs/df_{}_{}_test.h5'.format(data_key, EBEE)
   
    #load dataframe
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables+weight]).sample(frac=1).reset_index(drop=True)
    
    #transform features and targets
    transformer_file = 'data_{}'.format(EBEE)
    df_train.loc[:,kinrho+variables] = transform(df_train.loc[:,kinrho+variables], transformer_file, kinrho+variables)

    batch_size = pow(2, 10)
#    num_hidden_layers = 5
#    num_units_from = 50
#    shrink_rate = 0.8
#    num_units = [int((num_units_from*shrink_rate**i)/len(qs)) for i in range(num_hidden_layers)]
#    act = ['tanh' for _ in range(num_hidden_layers)]
#    dropout = [0.1 for _ in range(num_hidden_layers)]
#    gauss_std = None 

#    num_hidden_layers = 6
#    num_connected_layers = 3
#    num_units = [30, 25, 20, 30, 25, 10]

    '''
    num_hidden_layers = 5
    num_connected_layers = 3
    num_units = [8, 8, 8, 16, 16]
    act = ['relu' for _ in range(num_hidden_layers)]
    act[-1] = 'tanh'
    dropout = [0.4, 0.4, 0.4, 0.4]
    gauss_std = [0.2, 0.2, 0.2, 0.2, 0.2]
    '''

    num_hidden_layers = 5
    num_connected_layers = 2
    num_units = [30, 15, 20, 15, 10]
    act = ['tanh' for _ in range(num_hidden_layers)]
#    act = ['tanh','exponential', 'softplus', 'elu', 'tanh']
    dropout = [0.1, 0.1, 0.1, 0.1, 0.1]
    gauss_std = [0.2, 0.2, 0.2, 0.2, 0.2]

    #train
    
    train_start = time.time()

    if spl in [1, 2]: 
        modeldir = f'models/split{spl}'
        plotsdir = f'plots/split{spl}'
    else:
        modeldir = 'chained_models'
        plotsdir = 'plots'

#    target = variables[options.ith_var]
    for target in variables:
        features = kinrho + variables[:variables.index(target)] 
#        features = kinrho + variables[:min(variables.index(target),3)] 
#        features = kinrho 
        print('>>>>>>>>> train for variable {} with features {}'.format(target, features))

        X = df_train.loc[:,features]
        Y = df_train.loc[:,target]
        sample_weight = df_train.loc[:,'ml_weight']

        qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
        qweights = compute_qweights(Y, qs, sample_weight)
        print('quantile loss weights: {}'.format(qweights))

        model_file = '{}/{}_{}_{}'.format(modeldir, data_key, EBEE, target)
        history, eval_results = trainQuantile(
            X, Y, 
            qs, qweights, 
            num_hidden_layers, num_units, act, 
            num_connected_layers = num_connected_layers,
            sample_weight = sample_weight,
            l2lam = 1.e-3, 
            opt = 'Adadelta', lr = 0.5, 
            batch_size = batch_size, 
            epochs = 300, 
            save_file = model_file, 
            )

        train_end = time.time()
        print('evaluation results: ', eval_results)
        print('time spent in training: {} s'.format(train_end-train_start))
        
        # plot training history
        history_fig = plt.figure(tight_layout=True)
        plt.plot(history.history['loss'], label='training')
        plt.plot(history.history['val_loss'], label='validation')
        plt.yscale('log')
        plt.title('Training history')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        history_fig.savefig('{}/training_histories/{}_{}_{}.png'.format(plotsdir, data_key, EBEE, target))

   
   



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-s','--split', action='store', type=int)
    options = parser.parse_args()
    main(options)
