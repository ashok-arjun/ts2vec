##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Zhihan Yue
## Modified by: Arjun Ashok
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import numpy as np
import pandas as pd
import math
import wandb
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def _get_time_features_beijing(dt):
    return np.stack([
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)

def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_BeijingAirQuality(loader, dataset, target_col_indices, include_target, \
    train_slice_start=None, train_slice_end=None, valid_slice_end=None, cols=None, transform=False, task_type='regression'):
    data = pd.read_csv(f'datasets/{loader}/{dataset}.csv', index_col='date', parse_dates=True)
    if cols: data = data[cols]

    start_date = data.index[0]
    end_date = data.index[-1]
    print("Startdate: {} Enddate: {}".format(start_date, end_date))
    wandb.log({"dataset/start_date":start_date, "dataset/end_date":end_date})

    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]

    df_cols = data.columns
    print("Columns in the dataset:", df_cols)
    if target_col_indices: 
        target_col_indices_positive = [x if x >= 0 else len(data.columns)+x for x in target_col_indices]
        source_cols = [x for i,x in enumerate(df_cols) if i not in target_col_indices_positive]
        target_cols = [x for i,x in enumerate(df_cols) if i in target_col_indices_positive]

        print("Source columns:", source_cols)
        print("Target columns:", target_cols)

    data_pd = data.copy()        
    data = data.to_numpy()

    # Should we shuffle data before doing this?
    train_slice = slice(int(train_slice_start * len(data)), int(train_slice_end * len(data)))
    valid_slice = slice(int(train_slice_end * len(data)), int(valid_slice_end * len(data)))
    test_slice = slice(int(valid_slice_end * len(data)), None)
    
    train_slice_pd = data_pd.iloc[train_slice]
    valid_slice_pd = data_pd.iloc[valid_slice]
    test_slice_pd = data_pd.iloc[test_slice]
    
    print("Start and end dates of splits:")
    print("Train: Start: {} End: {}".format(train_slice_pd.index[0], train_slice_pd.index[-1]))
    print("Valid: Start: {} End: {}".format(valid_slice_pd.index[0], valid_slice_pd.index[-1]))
    print("Test: Start: {} End: {}".format(test_slice_pd.index[0], test_slice_pd.index[-1]))
    wandb.log({"dataset/modified_start_date_train":train_slice_pd.index[0], \
        "dataset/modified_end_date_train":train_slice_pd.index[-1], "dataset/modified_length_train": len(train_slice_pd)})
    wandb.log({"dataset/modified_start_date_valid":valid_slice_pd.index[0], \
        "dataset/modified_end_date_valid":valid_slice_pd.index[-1], "dataset/modified_length_valid": len(valid_slice_pd)})
    wandb.log({"dataset/modified_start_date_test":test_slice_pd.index[0], \
        "dataset/modified_end_date_test":test_slice_pd.index[-1], "dataset/modified_length_test": len(test_slice_pd)})

    if task_type.startswith("classification"):
        scale_data = data[:, :-1]
        labels = data[:, [-1]].copy()
    else:
        scale_data = data
    scaler = StandardScaler().fit(scale_data[train_slice])
    data = scaler.transform(scale_data)
    if task_type.startswith("classification"): data = np.concatenate([data, labels], axis=1)
    print("After transform:", data.shape)

    data = np.expand_dims(data, 0)
    data_full = data.copy()
    print("Shape of full_data", data_full.shape)
    if target_col_indices and not include_target:
        target_col_indices_positive = [x if x >= 0 else data.shape[2]+x for x in target_col_indices]
        data = data[:, :, [x for x in range(len(df_cols)) if x not in target_col_indices_positive]]
    print("Shape of data", data.shape)
    if n_covariate_cols > 0:
        print("Fitting StandardScaler to dt_embed[train_slice]...")
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        print(dt_embed.shape, data.shape)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=2)
        data_full = np.concatenate([np.repeat(dt_embed, data_full.shape[0], axis=0), data_full], axis=2)
        print("Done.")

    """
    if task_type == 'forecasting' or task_type == 'regression_as_forecasting':
        data = np.expand_dims(data, 0)
        data_full = data.copy()
        print("Shape of full_data", data_full.shape)
        if target_col_indices and not include_target:
            target_col_indices_positive = [x if x >= 0 else data.shape[2]+x for x in target_col_indices]
            data = data[:, :, [x for x in range(len(df_cols)) if x not in target_col_indices_positive]]
        print("Shape of data", data.shape)
        if n_covariate_cols > 0:
            print("Fitting StandardScaler to dt_embed[train_slice]...")
            dt_scaler = StandardScaler().fit(dt_embed[train_slice])
            dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
            print(dt_embed.shape, data.shape)
            data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=2)
            data_full = np.concatenate([np.repeat(dt_embed, data_full.shape[0], axis=0), data_full], axis=2)
            print("Done.")
    else:
        data = np.expand_dims(data, 1)
        data_full = data.copy()
        print("Shape of full_data", data_full.shape) # Shape of data_full: (32681, 1, 13) 
        if target_col_indices and not include_target:
            target_col_indices_positive = [x if x >= 0 else data.shape[2]+x for x in target_col_indices]
            data = data[:, :, [x for x in range(len(df_cols)) if x not in target_col_indices_positive]]
        print("Shape of data", data.shape) # Shape of data: (32681, 1, 12) 
        if n_covariate_cols > 0:
            print("Fitting StandardScaler to dt_embed[train_slice]...")
            dt_scaler = StandardScaler().fit(dt_embed[train_slice])
            dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 1)
            data = np.concatenate([np.repeat(dt_embed, data.shape[1], axis=0), data], axis=2)
            data_full = np.concatenate([np.repeat(dt_embed, data_full.shape[1], axis=0), data_full], axis=2)
            print("Done.")
    """
    
    if task_type.startswith("classification"):
        targets = np.array(data_full[0, :, -1])
        labels = np.unique(targets)
        transform = { k : i for i, k in enumerate(labels)}
        targets_int = np.vectorize(transform.get)(targets)
        data_full[0, :, -1] = targets_int

        print("Labels", targets_int)

    print("Added covariate columns.")
    print("Shape of data_full:", data_full.shape)
    print("Shape of data:", data.shape)

    return data_full, data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols


def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    print(train_X.shape, test_X.shape)
    print(train_y)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    # print(train_X.shape, test_X.shape)
    print(train_y)
    return train_X, train_y, test_X, test_y
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0

def load_forecast_csv(name, univar=False, load_feats=False, start_date=None, end_date=None, \
                    train_slice_start=None, train_slice_end=None, \
                    valid_slice_end=None):
    filename = name if not load_feats else name + "_feats"
    data = pd.read_csv(f'datasets/{filename}.csv', index_col='date', parse_dates=True)
    print("Dataset starts at {} and ends at {}".format(data.index[0], data.index[-1]))
    wandb.log({"dataset/start_date":data.index[0], "dataset/end_date":data.index[-1], "dataset/length": len(data)})

    if not start_date:
        start_date = data.index[0]
    if not end_date:
        end_date = data.index[-1]
    print("Startdate: {} Enddate: {}".format(start_date, end_date))
    wandb.log({"dataset/given_start_date":start_date, "dataset/given_end_date":end_date})

    data = data.loc[start_date:end_date]
    print("Modified: Dataset starts at {} and ends at {}".format(data.index[0], data.index[-1]))
    wandb.log({"dataset/modified_start_date":data.index[0], "dataset/modified_end_date":data.index[-1], \
        "dataset/modified_length": len(data)})

    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]

    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        elif name == 'WTH':
            data = data[['WetBulbCelsius']]
        else:
            data = data.iloc[:, -1:]

    data_pd = data.copy()        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(int(train_slice_start * len(data)), int(train_slice_end * len(data)))
        valid_slice = slice(int(train_slice_end * len(data)), int(valid_slice_end * len(data)))
        test_slice = slice(int(valid_slice_end * len(data)), None)
    
    train_slice_pd = data_pd.iloc[train_slice]
    valid_slice_pd = data_pd.iloc[valid_slice]
    test_slice_pd = data_pd.iloc[test_slice]
    
    print("Start and end dates of splits:")
    print("Train: Start: {} End: {}".format(train_slice_pd.index[0], train_slice_pd.index[-1]))
    print("Valid: Start: {} End: {}".format(valid_slice_pd.index[0], valid_slice_pd.index[-1]))
    print("Test: Start: {} End: {}".format(test_slice_pd.index[0], test_slice_pd.index[-1]))
    wandb.log({"dataset/modified_start_date_train":train_slice_pd.index[0], \
        "dataset/modified_end_date_train":train_slice_pd.index[-1], "dataset/modified_length_train": len(train_slice_pd)})
    wandb.log({"dataset/modified_start_date_valid":valid_slice_pd.index[0], \
        "dataset/modified_end_date_valid":valid_slice_pd.index[-1], "dataset/modified_length_valid": len(valid_slice_pd)})
    wandb.log({"dataset/modified_start_date_test":test_slice_pd.index[0], \
        "dataset/modified_end_date_test":test_slice_pd.index[-1], "dataset/modified_length_test": len(test_slice_pd)})

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    pred_lens = [24, 48, 168, 336, 720]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data
