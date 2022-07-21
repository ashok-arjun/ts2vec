import numpy as np
import time
import wandb
from . import _eval_protocols as eval_protocols

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:] # (1, feat.shape[1], pred_len, data.shape[2])
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean(),
        'MAPE': np.mean(np.abs((pred - target) / target)) * 100
    }
    
def eval_forecasting(method, model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, target_col_indices, \
    padding=200, include_target=False):

    if target_col_indices:
        target_cols = target_col_indices
        if not include_target:
            target_col_indices_positive = [x if x >= 0 else data.shape[2]+x for x in target_col_indices]
            source_cols = [x for x in list(range(data.shape[2])) if x not in target_col_indices_positive]
        else:
            source_cols = list(range(0, data.shape[2]))
    else:
        target_cols = list(range(0, data.shape[2]))
        target_cols = target_cols[n_covariate_cols:]
        source_cols = list(range(0, data.shape[2]))

    encoding_data = data[:, :, source_cols]
    print("Encoding data shape:", encoding_data.shape)

    t = time.time()

    if method == 'ts2vec':
        all_repr = model.encode(
            encoding_data,
            casual=True,
            sliding_length=1,
            sliding_padding=padding,
            batch_size=256
        )
    else:
        all_repr = model.encode(
            encoding_data,
            mode='forecasting',
            casual=True,
            sliding_length=1,
            sliding_padding=padding,
            batch_size=256
        )
    ts2vec_infer_time = time.time() - t
    
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    train_data = data[:, train_slice, target_cols]
    valid_data = data[:, valid_slice, target_cols]
    test_data = data[:, test_slice, target_cols]
    
    print("Target columns:", target_cols)
    print("Shape of train_data", train_data.shape)

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        
        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        
        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)
        
        # if test_data.shape[0] > 1:
        #     test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
        #     test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        # else:
        #     test_pred_inv = scaler.inverse_transform(test_pred)
        #     test_labels_inv = scaler.inverse_transform(test_labels)
            
        out_log[pred_len] = {
            'norm': test_pred,
            # 'raw': test_pred_inv,
            'norm_gt': test_labels,
            # 'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            # 'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }

        for metric, value in ours_result[pred_len]['norm'].items():
            wandb.log({"eval/{}/{}".format(pred_len, metric): value})
        
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res
