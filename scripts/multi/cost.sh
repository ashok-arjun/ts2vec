# Train

# python -u train.py neighbour 'neighbour_feats_cost' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
# --train --eval --load_feats --method cost --tags 'pretraining' \
# --start_date '2010-08-15 15:00:00' --end_date '2015-08-15 15:00:00' \
# --train_slice_end 0.85 --valid_slice_end 0.95 --batch-size 32 \
# --ckpt_location "trained_models/neighbour_feats_cost/model.pkl"

# python -u train.py neighbour 'neighbour_feats_cost_2008_2015' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 --max-train-length 201  \
# --train --eval --load_feats --method cost --tags 'pretraining' \
# --start_date '2007-12-31 19:00:00' --end_date '2015-08-15 15:00:00' \
# --train_slice_end 0.85 --valid_slice_end 0.95 --batch-size 32 \
# --ckpt_location "trained_models/neighbour_feats_cost_2008_2015/model.pkl"

python -u train.py bahia_windspeed 'bahia_windspeed_feats_cost' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 --batch-size 32 \
--train --eval --load_feats --method cost --tags 'training' \
--ckpt_location "trained_models/bahia_windspeed_feats_cost/model.pkl"

# python -u train.py bahia_windspeed 'bahia_windspeed_feats_cost_continuing_neighbour' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32 --max-train-length 201 \
# --train --eval --load_feats --method cost --tags 'training' \
# --load_ckpt "trained_models/neighbour_feats_cost/model.pkl" \
# --ckpt_location "trained_models/bahia_windspeed_feats_cost_continuing_neighbour/model.pkl"

# python -u train.py bahia_windspeed 'bahia_windspeed_feats_cost_continuing_neighbour_2008_2015' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 --batch-size 32 --max-train-length 201 \
# --train --eval --load_feats --method cost --tags 'training' \
# --load_ckpt "trained_models/neighbour_feats_cost_2008_2015/model.pkl" \
# --ckpt_location "trained_models/bahia_windspeed_feats_cost_continuing_neighbour_2008_2015/model.pkl"

# # # Evaluate 1 (ZS) and 2 (SUP) on bahia 
# To evaluate on MS, add --load_feats --tags 'evaluation-MV' and remove --target_col_indices -1

python -u train.py bahia_windspeed 'bahia_windspeed_feats_cost_bahia_windspeed_eval' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 \
--eval --method cost --tags 'evaluation' \
--batch-size 32 --max-train-length 201 \
--load_ckpt "trained_models/bahia_windspeed_feats_cost/model.pkl" \
 --target_col_indices -1 

 python -u train.py bahia_windspeed 'bahia_windspeed_feats_cost_bahia_windspeed_eval' --loader forecast_csv \
--repr-dims 320 --max-threads 8 --seed 42 \
--eval --method cost --tags 'evaluation' \
--batch-size 32 --max-train-length 201 \
--load_ckpt "trained_models/bahia_windspeed_feats_cost/model.pkl" \
--load_feats --tags 'evaluation-MV'

# python -u train.py bahia_windspeed 'neighbour_feats_cost_bahia_windspeed_eval' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 \
# --eval --method cost --tags 'evaluation' \
# --batch-size 32 --max-train-length 201 \
# --load_ckpt "trained_models/neighbour_feats_cost/model.pkl" \
# --load_feats --tags 'evaluation-MV'

# python -u train.py bahia_windspeed 'neighbour_feats_cost_2008_2015_windspeed_eval' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 \
# --eval --method cost --tags 'evaluation' \
# --batch-size 32 --max-train-length 201 \
# --load_ckpt "trained_models/neighbour_feats_cost_2008_2015/model.pkl" \
# --load_feats --tags 'evaluation-MV'

# python -u train.py bahia_windspeed 'bahia_windspeed_feats_ts2vec_continuing_neighbour_bahia_windspeed_eval' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 \
# --eval --method cost --tags 'evaluation' \
# --batch-size 32 --max-train-length 201 \
# --load_ckpt "trained_models/bahia_windspeed_feats_cost_continuing_neighbour/model.pkl" \
# --load_feats --tags 'evaluation-MV'

# python -u train.py bahia_windspeed 'bahia_windspeed_feats_ts2vec_continuing_neighbour_bahia_windspeed_eval_2008_2015' --loader forecast_csv \
# --repr-dims 320 --max-threads 8 --seed 42 \
# --eval --method cost --tags 'evaluation' \
# --batch-size 32 --max-train-length 201 \
# --load_ckpt "trained_models/bahia_windspeed_feats_cost_continuing_neighbour_2008_2015/model.pkl" \
# --load_feats --tags 'evaluation-MV'