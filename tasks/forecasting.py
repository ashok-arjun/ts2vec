##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Zhihan Yue
## Modified by: Arjun Ashok
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import time
import wandb
from . import _eval_protocols as eval_protocols
import matplotlib.pyplot as plt

# def make_window

def generate_pred_samples(features, data, pred_len, drop=0, regression=False):
    n = data.shape[1]
    if not regression: features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:n+i-pred_len+1] for i in range(pred_len)], axis=2) # (1, feat.shape[1], pred_len, data.shape[2])
    if not regression: labels = labels[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def smape(A, F):
    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    if len_ == 0 and np.nansum(tmp) == 0: # Deals with a special case
        return 100
    return 100 / len_ * np.nansum(tmp)

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean(),
        'MAPE': np.mean(np.abs((pred - target) / target)) * 100,
        'SMAPE': smape(target, pred)
    }
    
def eval_forecasting(args, method, model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, target_col_indices, \
    padding=200, include_target=False, protocol="ridge"):

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
    # Encoding data shape: (1, 32681, 12)

    t = time.time()

    # if not args.train and not args.load_ckpt:
    #     print("Using data as representations")
    #     train_repr = encoding_data[:, train_slice]
    #     valid_repr = encoding_data[:, valid_slice]
    #     test_repr = encoding_data[:, test_slice]
    #     repr = encoding_data.reshape(encoding_data.shape[0], encoding_data.shape[1])
    # else:
    #     repr = model.encode(encoding_data, encoding_window='full_series' if encoding_targets.ndim == 1 else None)
    
    if not args.train and not args.load_ckpt:
        print("Using data as representations")
        all_repr = encoding_data
    else:
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
    
    train_targets = data[:, train_slice, target_cols]
    valid_targets = data[:, valid_slice, target_cols]
    test_targets = data[:, test_slice, target_cols]
    
    print("Target columns:", target_cols)
    print("data:{}".format(data.shape))
    print("train_repr:{}. train_targets:{}".format(train_repr.shape, train_targets.shape))
    print("valid_repr:{}. valid_targets:{}".format(valid_repr.shape, valid_targets.shape))
    print("test_repr:{}. test_targets:{}".format(test_repr.shape, test_targets.shape))

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_targets, pred_len, drop=padding, \
                                                            regression='regression' in args.task_type)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_targets, pred_len, regression='regression' in args.task_type)
        test_features, test_labels = generate_pred_samples(test_repr, test_targets, pred_len, regression='regression' in args.task_type)
        
        print("train_features:{}. train_labels:{}".format(train_features.shape, train_labels.shape))
        print("valid_features:{}. valid_labels:{}".format(valid_features.shape, valid_labels.shape))
        print("test_features:{}. test_labels:{}".format(test_features.shape, test_labels.shape))

        t = time.time()

        print("Protocol:",protocol)
        if protocol == "ridge":
            lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        elif protocol == "neural_network":
            lr = eval_protocols.fit_neural_network(train_features, train_labels, valid_features, valid_labels)

        lr_train_time[pred_len] = time.time() - t
        
        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_targets.shape[0], -1, pred_len, test_targets.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        print("test_pred:", test_pred.shape)
        print("test_labels:", test_labels.shape)

        """
        test_pred: (1, 7721, 24, 1)                                                                                                                                      
        test_labels: (1, 7721, 24, 1) 
        """

        if args.plot_preds:
            for i in range(pred_len):
                i_ahead_forecasts = test_pred[0, :, i].reshape(-1)
                i_ahead_gt = test_labels[:, :, i].reshape(-1)

                plt.clf()

                plt.plot(list(range(i_ahead_forecasts.shape[0])), i_ahead_forecasts, label = "pred")
                plt.plot(list(range(i_ahead_gt.shape[0])), i_ahead_gt, label = "gt")

                plt.legend()

                wandb.log({"forecast_plots/pred_len_{}/{}_hour_ahead".format(pred_len, i): plt})

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
