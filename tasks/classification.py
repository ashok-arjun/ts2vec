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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    y_score = clf.predict_proba(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    return y_score, { 'acc': acc, 'auprc': auprc }

def eval_classification_custom(args, method, model, data, train_slice, valid_slice, test_slice, target_col_indices, include_target=False, padding=200):
    print("data shape:", data.shape)

    if target_col_indices:
        target_cols = target_col_indices
        if not include_target:
            target_col_indices_positive = [x if x >= 0 else data.shape[2]+x for x in target_col_indices]
            source_cols = [x for x in list(range(data.shape[2])) if x not in target_col_indices_positive]
        else:
            source_cols = list(range(0, data.shape[2]))
    else:
        target_cols = list(range(0, data.shape[2]))
        source_cols = list(range(0, data.shape[2]))

    print(source_cols, target_cols)
    encoding_data = data[:, :, source_cols]
    encoding_targets = data[:, :, target_cols].reshape(data.shape[1])

    print("Encoding data shape:", encoding_data.shape)
    print("Encoding targets shape:", encoding_targets.shape)

    if encoding_data.shape[0] != 1:
        encoding_data = encoding_data.reshape(1, encoding_data.shape[1], encoding_data.shape[2]) 
    print("Encoding data shape:", encoding_data.shape)

    """Encoding data shape: (1, 32681, 12)
    Encoding targets shape: (32681,)"""

    if not args.train and not args.load_ckpt:
        print("Using data as representations")
        repr = encoding_data.reshape(encoding_data.shape[1], encoding_data.shape[2])
    else:
        if method == 'ts2vec':
            repr = model.encode(
                encoding_data,
                casual=True,
                sliding_length=1,
                sliding_padding=padding,
                batch_size=256
            )
        else:
            repr = model.encode(
                encoding_data,
                mode='forecasting',
                casual=True,
                sliding_length=1,
                sliding_padding=padding,
                batch_size=256
            )
        # # encoding_window='full_series' if encoding_targets.ndim == 1 else None
        # repr = model.encode(encoding_data, encoding_window=None, \
        #     casual=True, sliding_length=1, sliding_padding=23)
    if len(repr.shape) > 2:
        repr = repr.reshape(repr.shape[1], repr.shape[2])
    print("repr:", repr.shape)

    """repr: (32681, 320)"""

    train_repr = repr[train_slice]
    valid_repr = repr[valid_slice]
    test_repr = repr[test_slice]

    train_targets = encoding_targets[train_slice]
    valid_targets = encoding_targets[valid_slice]
    test_targets = encoding_targets[test_slice]

    if not args.train and not args.load_ckpt:
        train_repr, train_targets = train_repr[padding:], train_targets[padding:]

    print("Target columns:", target_cols)
    print("data:{}".format(data.shape))
    print("train_repr:{}. train_targets:{}".format(train_repr.shape, train_targets.shape))
    print("valid_repr:{}. valid_targets:{}".format(valid_repr.shape, valid_targets.shape))
    print("test_repr:{}. test_targets:{}".format(test_repr.shape, test_targets.shape))

    # methods = [eval_protocols.fit_lr, eval_protocols.fit_knn, eval_protocols.fit_svm]
    # names = ["logistic_regression", "knn", "svm"]
    methods = [eval_protocols.fit_lr, eval_protocols.fit_knn, eval_protocols.fit_neural_network]
    names = ["logistic_regression", "knn", "neural_network"]
    # methods = [eval_protocols.fit_neural_network]
    # names = ["neural_network"]
    val_auprc = []
    trained_methods = []
    for method, name in zip(methods, names):
        print("Fitting {}...".format(name))
        if name == "neural_network":
            clf = method(train_repr, train_targets, valid_repr, valid_targets, task='classification')
        else:
            clf = method(train_repr, train_targets)
        if name == "svm":
            y_score = clf.decision_function(valid_repr)
        else:
            y_score = clf.predict_proba(valid_repr)
        valid_labels_onehot = label_binarize(valid_targets, classes=np.arange(train_targets.max()+1))
        auprc = average_precision_score(valid_labels_onehot, y_score)
        val_auprc.append(auprc)
        trained_methods.append(clf)

    print("Scores")
    print(names)
    print(val_auprc)
    
    best_clf = np.argsort(val_auprc)[-1]
    trained_method = trained_methods[best_clf]
    method = methods[best_clf]
    name = names[best_clf]

    acc = trained_method.score(test_repr, test_targets)

    if name == "svm":
        y_score = trained_method.decision_function(test_repr)
    else:
        y_score = trained_method.predict_proba(test_repr)

    test_labels_onehot = label_binarize(test_targets, classes=np.arange(train_targets.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    result = { 'acc': acc, 'auprc': auprc }

    for metric, value in result.items():
        wandb.log({"eval/{}".format(metric): value})

    return y_score, { 'acc': acc, 'auprc': auprc }
