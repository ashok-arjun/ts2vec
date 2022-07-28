import numpy as np
import time
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
import wandb

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

def eval_forecasting(method, model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, target_col_indices, \
    padding=200, include_target=False):
    pass

def eval_regression(args, model, data, train_slice, valid_slice, test_slice, target_col_indices, include_target=False):

    print("data shape:", data.shape)
    if target_col_indices:
        target_cols = target_col_indices
        if not include_target:
            target_col_indices_positive = [x if x >= 0 else data.shape[1]+x for x in target_col_indices]
            source_cols = [x for x in list(range(data.shape[1])) if x not in target_col_indices_positive]
        else:
            source_cols = list(range(0, data.shape[1]))
    else:
        target_cols = list(range(0, data.shape[1]))
        source_cols = list(range(0, data.shape[1]))

    print(source_cols, target_cols)
    encoding_data = data[:, source_cols, :]
    encoding_targets = data[:, target_cols, :].reshape(data.shape[0])

    print("Encoding data shape:", encoding_data.shape)
    print("Encoding targets shape:", encoding_targets.shape)

    """Encoding data shape: (32681, 12, 1)
    Encoding targets shape: (32681,)"""

    t = time.time()

    if not args.train and not args.load_ckpt:
        print("Using data as representations")
        repr = encoding_data.reshape(encoding_data.shape[0], encoding_data.shape[1])
    else:
        repr = model.encode(encoding_data, encoding_window='full_series' if encoding_targets.ndim == 1 else None)
    if len(repr.shape) > 2:
        repr = repr.reshape(repr.shape[0], repr.shape[2])
    print("repr:", repr.shape)

    """repr: (32681, 1, 320)"""

    train_repr = repr[train_slice]
    valid_repr = repr[valid_slice]
    test_repr = repr[test_slice]

    train_targets = encoding_targets[train_slice]
    valid_targets = encoding_targets[valid_slice]
    test_targets = encoding_targets[test_slice]

    lr = eval_protocols.fit_ridge(train_repr, train_targets, valid_repr, valid_targets)
    
    test_pred = lr.predict(test_repr)
    print("test_pred:", test_pred.shape)

    log = {
        'norm': test_pred,
        'norm_gt': test_targets,
    }
    result = {
        'norm': cal_metrics(test_pred, test_targets),
    }

    for metric, value in result['norm'].items():
        wandb.log({"eval/{}".format(metric): value})

    return log, result