from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import xgboost as xgb

def trainQuantile(X, Y, qs, sample_weight=None, epochs=10, checkpoint_dir='./ckpt', save_file=None):

    input_dim = len(X.keys())
    
    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    alpha = np.array(qs)
    evals_result: Dict[str, Dict] = {}

    X_train, X_val, y_train, y_val, W_train, W_val= train_test_split(X, Y, sample_weight, test_size=0.1)
    Xy_train = xgb.QuantileDMatrix(X_train, y_train, weight=W_train)
    Xy_val = xgb.QuantileDMatrix(X_val, y_val, ref=Xy_train, weight=W_val)

    watchlist  = [(Xy_train,'qloss'), (Xy_val, 'qloss')]

    booster = xgb.train(
        {
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "quantile_alpha": alpha,
            "learning_rate": 0.04,
            "max_depth": 10,
            "device": "cuda:0"
        },
        Xy_train,
        1000,
        watchlist
        num_boost_round=128,
        early_stopping_rounds=3,
        evals=[(Xy_train, "Train"), (Xy_val, "Val")],
        evals_result=evals_result,
    )

    if save_file is not None:
        booster.save_model(save_file)

    return history, eval_results


def predict(X, qs, qweights, model_from=None, scale_par=None):

    booster = xgboost.load_model(model_from)

    predY = model.predict(X)

    if scale_par is not None: 
        logger.info('target is scaled, now mapping it back!')
        predY = predY*scale_par['sigma'] + scale_par['mu']

    return predY


def scale(df, scale_file):

    df = pd.DataFrame(df)

    par = pd.read_hdf(scale_file).loc[:,df.keys()] 

    df_scaled = (df - par.loc['mu',:])/par.loc['sigma',:]
    return df_scaled


def load_or_restore_model(checkpoint_dir, qs, qweights, *args, **kwargs):

    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return xgboost.load_model(latest_checkpoint)
    print("Creating a new model")




