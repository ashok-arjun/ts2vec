# Train

python -u train.py neighbour 'neighbour_feats_ts2vec_2008_2015' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 \
--train --eval --load_feats --method ts2vec --tags 'pretraining' \
--start_date '2007-12-31 19:00:00' --end_date '2015-08-15 15:00:00' \
--train_slice_end 0.85 --valid_slice_end 0.95 --batch-size 32 \
--ckpt_location "trained_models/neighbour_feats_ts2vec_2008_2015/model.pkl"

# # python -u train.py bahia_windspeed 'bahia_windspeed_feats_ts2vec' --loader forecast_csv \
# # --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32 \
# # --train --eval --load_feats --method ts2vec --tags 'training' \
# # --ckpt_location "trained_models/bahia_windspeed_feats_ts2vec/model.pkl"

python -u train.py bahia_windspeed 'bahia_windspeed_feats_ts2vec_continuing_neighbour_2008_2015' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 --batch-size 32 \
--train --eval --load_feats --method ts2vec --tags 'training' \
--load_ckpt "trained_models/neighbour_feats_ts2vec_2008_2015/model.pkl" \
--ckpt_location "trained_models/bahia_windspeed_feats_ts2vec_2008_2015_continuing_neighbour/model.pkl"

# # Evaluate 1 (ZS) and 2 (SUP) on bahia

python -u train.py bahia_windspeed 'neighbour_feats_ts2vec_bahia_windspeed_eval_2008_2015' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 \
--eval --method ts2vec --tags 'evaluation' \
--batch-size 32 \
--load_ckpt "trained_models/neighbour_feats_ts2vec_2008_2015/model.pkl" \
 --target_col_indices -1

# python -u train.py bahia_windspeed 'bahia_windspeed_feats_ts2vec_bahia_windspeed_eval' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 \
# --eval --method ts2vec --tags 'evaluation' \
# --batch-size 32 \
# --load_ckpt "trained_models/bahia_windspeed_feats_ts2vec/model.pkl" \
#  --target_col_indices -1

python -u train.py bahia_windspeed 'bahia_windspeed_feats_ts2vec_continuing_neighbour_bahia_windspeed_eval_2008_2015' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 \
--eval --method ts2vec --tags 'evaluation' \
--batch-size 32 \
--load_ckpt "trained_models/bahia_windspeed_feats_ts2vec_2008_2015_continuing_neighbour/model.pkl" \
 --target_col_indices -1

 # Evaluate 1 (ZS) and 2 (SUP) on bahia MS

python -u train.py bahia_windspeed 'neighbour_feats_ts2vec_bahia_windspeed_eval_2008_2015' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 \
--eval --method ts2vec --tags 'evaluation-MV' \
--batch-size 32 --load_feats \
--load_ckpt "trained_models/neighbour_feats_ts2vec_2008_2015/model.pkl" 

# python -u train.py bahia_windspeed 'bahia_windspeed_feats_ts2vec_bahia_windspeed_eval' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 \
# --eval --method ts2vec --tags 'evaluation-MV' \
# --batch-size 32 --load_feats \
# --load_ckpt "trained_models/bahia_windspeed_feats_ts2vec/model.pkl"

python -u train.py bahia_windspeed 'bahia_windspeed_feats_ts2vec_continuing_neighbour_bahia_windspeed_eval_2008_2015' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 \
--eval --method ts2vec --tags 'evaluation-MV' \
--batch-size 32 --load_feats \
--load_ckpt "trained_models/bahia_windspeed_feats_ts2vec_2008_2015_continuing_neighbour/model.pkl"