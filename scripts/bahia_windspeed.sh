# Pretraining: 2004-12-31 17:00:00 and ends at 2020-01-14 23:00:00
# Dataset: 2015-08-15 15:00:00 and ends at 2020-01-14 23:00:00

python -u train.py neighbour 'neighbour_feats_cost_pretrain_dates' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 \
--train --eval --load_feats --method cost --tags 'pretraining' \
--start_date '2010-08-15 15:00:00' --end_date '2015-08-15 15:00:00' \
--train_slice_end 0.85 --valid_slice_end 0.95

# python -u train.py bahia_windspeed 'bahia_windspeed_feats_cost' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 \
# --train --eval --load_feats --method cost --tags 'pretraining'

# python -u train.py bahia_windspeed bahia_windspeed_feats_load_and_eval_S --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 \
# --load_ckpt "training/bahia_windspeed__bahia_windspeed_feats_20220717_151849/model.pkl" \
# --eval --target_col_indices -1
