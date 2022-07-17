# python -u train.py bahia_windspeed bahia_windspeed_feats --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 \
# --train --eval --load_feats

python -u train.py bahia_windspeed bahia_windspeed_feats_load_and_eval_S --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 \
--load_ckpt "training/bahia_windspeed__bahia_windspeed_feats_20220717_151849/model.pkl" \
--eval --target_col_indices -1