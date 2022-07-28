# python -u train.py neighbour 'neighbour_feats_cost_pretrain_dates' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 \
# --train --eval --load_feats --method cost --tags 'pretraining' 'tests' \
# --start_date '2010-08-15 15:00:00' --end_date '2015-08-15 15:00:00' \
# --train_slice_end 0.85 --valid_slice_end 0.95

# python -u train.py Changping_PM2.5 Changping_2.5_weather_20k_0.9_train --loader Monash \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
# --method cost --tags 'training' 'beijing' 'tests' --target_col_indices -1 --iters 20000 \
# --ckpt_location "trained_models/Changping_2.5_weather_20k_0.9_train_cost/model_test.pkl" --train \
# --train_slice_end 0.9 --valid_slice_end 0.95 --max-train-length 12 --iters 2

python -u train.py Changping_PM2.5 Changping_2.5_weather_20k_0.9_eval --loader Monash \
--repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
--eval --method ts2vec --tags 'evaluation' 'beijing' 'tests' --target_col_indices -1 \
--load_ckpt "trained_models/Changping_2.5_weather_20k_0.9_train/model_test.pkl" --max-train-length 12

python -u train.py Changping_PM2.5 Changping_2.5_weather_20k_0.9_eval --loader Monash \
--repr-dims 320 --max-threads 8 --seed 42 --batch-size 32  \
--eval --method cost --tags 'evaluation' 'beijing' 'tests' --target_col_indices -1 \
--load_ckpt "trained_models/Changping_2.5_weather_20k_0.9_train_cost/model_test.pkl" --max-train-length 12