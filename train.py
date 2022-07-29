import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import numpy as np
import argparse
import os
import sys
import json
import time
import datetime
import tasks
import wandb
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout, set_seed
from pathlib import Path

from ts2vec import TS2Vec
from cost import CoST

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--train', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    
    # Method
    parser.add_argument('--method', type=str, default="ts2vec", choices=["ts2vec", "cost"])

    # CoST args
    parser.add_argument('--kernels', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument('--alpha', type=float, default=0.0005)

    # Custom
    parser.add_argument('--wandb_run_name', type=str, help='device ids of multile gpus')
    parser.add_argument('--wandb_resume_id', type=str, help='device ids of multile gpus')
    parser.add_argument('--tags', nargs='+')

    parser.add_argument('--step_lrs', action='store_true', help='device ids of multile gpus')
    parser.add_argument('--step_lrs_patience', type=int, default=5, help='device ids of multile gpus')
    parser.add_argument('--step_lrs_alpha', type=float, default=0.1, help='device ids of multile gpus')
    parser.add_argument('--step_lrs_cutoff', type=float, default=1e-9, help='device ids of multile gpus')
    parser.add_argument('--load_feats', action='store_true', help='device ids of multile gpus')
    parser.add_argument('--target_col_indices', nargs='+', type=int, default=[])
    parser.add_argument('--include_target', action='store_true')
    parser.add_argument('--load_ckpt', type=str, help='device ids of multile gpus')

    # Start and end date
    parser.add_argument('--start_date', type=str, help='device ids of multile gpus')
    parser.add_argument('--end_date', type=str, help='device ids of multile gpus')

    # Slices
    parser.add_argument('--train_slice_start', type=float, default=0., help='device ids of multile gpus')
    parser.add_argument('--train_slice_end', type=float, default=0.6, help='device ids of multile gpus')
    parser.add_argument('--valid_slice_end', type=float, default=0.8, help='device ids of multile gpus')

    # Specify checkpoint location to continue training / finetuning
    parser.add_argument('--ckpt_location', type=str)

    # Plot preds
    parser.add_argument('--plot_preds', action='store_true')

    args = parser.parse_args()
    
    run_name = args.method + '__' + args.dataset + '__' + name_with_datetime(args.run_name)
    
    if not args.wandb_run_name:
        args.wandb_run_name = args.run_name

    args.run_name = run_name

    if args.wandb_resume_id:
        wandb.init(entity="arjunashok", project="ts2vec", config=vars(args), resume=True, id=args.wandb_resume_id, tags=args.tags)
    else:
        wandb.init(entity="arjunashok", project="ts2vec", config=vars(args), name=args.wandb_run_name, tags=args.tags)

    print("Method:", args.method)
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    set_seed(args.seed)

    method = args.method
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    pred_lens = [24, 48, 168, 336, 720]
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        print("Shape of train_data:", train_data.shape)
        print("Shape of train_labels:", train_labels.shape)
        print("Shape of test_data:", test_data.shape)
        print("Shape of test_labels:", test_labels.shape)

    elif args.loader.startswith('PM2.5') or args.loader.startswith('PM10') or args.loader == 'BeijingWD':
        if args.loader.endswith('forecasting'):
            task_type = 'forecasting'
        elif args.loader == 'BeijingWD':
            task_type = 'classification_custom'
        else:
            task_type = 'regression'

        if args.loader.startswith('PM2.5'):
            args.loader = 'PM2.5'
        elif args.loader.startswith('PM10'):
            args.loader = 'PM10'
        elif args.loader == 'BeijingWD':
            args.loader = 'WD'

        data_full, data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols = datautils.load_BeijingAirQuality(args.loader, args.dataset, \
        args.target_col_indices, args.include_target, train_slice_start=args.train_slice_start, \
        train_slice_end=args.train_slice_end, valid_slice_end=args.valid_slice_end, task_type=task_type)
        train_data = data[:, train_slice]
        train_data = data[:, train_slice]
        print("Shape of data:", data.shape)
        print("Shape of train data:", train_data.shape)

    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        print("Shape of train_data:", train_data.shape)
        print("Shape of train_labels:", train_labels.shape)
        print("Shape of test_data:", test_data.shape)
        print("Shape of test_labels:", test_labels.shape)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, load_feats=args.load_feats, start_date=args.start_date, end_date=args.end_date, \
        train_slice_start=args.train_slice_start, train_slice_end=args.train_slice_end, \
        valid_slice_end=args.valid_slice_end)
        train_data = data[:, train_slice]
        print("Shape of data:", data.shape)
        print("Shape of train data:", train_data.shape)

    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')
        
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
    
    print("task type:", task_type)
    wandb.run.summary["task_type"] = task_type

    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + run_name
    os.makedirs(run_dir, exist_ok=True)
    
    with open(f'{run_dir}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    t = time.time()
    
    input_dims = train_data.shape[-1]
    if not args.include_target and task_type == 'forecasting':
        if not (args.loader.startswith('PM2.5') or args.loader.startswith('PM10')):
            input_dims -= len(args.target_col_indices)
    print("Total input_dims:", input_dims)
    if method == 'ts2vec':
        model = TS2Vec(
            input_dims=input_dims,
            device=device,
            **config
        )
    elif method == 'cost':
        model = CoST(
            input_dims=input_dims,
            kernels=args.kernels,
            alpha=args.alpha,
            device=device,
            **config
        )

    if args.load_ckpt:
        model.load(args.load_ckpt)

    if args.train:
        loss_log = model.fit(
            train_data,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=True
        )
        ckpt_location = f'{run_dir}/model.pkl'
        if args.ckpt_location: ckpt_location = args.ckpt_location
        ckpt_location_path = Path(ckpt_location)
        os.makedirs(ckpt_location_path.parent.absolute(), exist_ok=True)
        model.save(ckpt_location)
        wandb.save(ckpt_location)
        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == "classification_custom":
            out, eval_res = tasks.eval_classification_custom(args, model, data_full, train_slice, valid_slice, test_slice, \
                target_col_indices=args.target_col_indices, include_target=args.include_target)
        elif task_type == 'forecasting':
            padding = 200 if method == 'ts2vec' else args.max_train_length - 1
            out, eval_res = tasks.eval_forecasting(args, method, model, data_full, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, target_col_indices=args.target_col_indices, padding=padding, include_target=args.include_target)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'regression':
            out, eval_res = tasks.eval_regression(args, model, data_full, train_slice, valid_slice, test_slice, target_col_indices=args.target_col_indices, include_target=args.include_target)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

    wandb.finish()
    print("Finished.")

